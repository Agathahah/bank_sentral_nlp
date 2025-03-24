#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fine_tuning.py - Script untuk fine-tuning model IndoBERT untuk analisis sentimen Bank Sentral
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AdamW,
    get_linear_schedule_with_warmup
)
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import logging
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import random
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentimentDataset(Dataset):
    """Dataset untuk fine-tuning IndoBERT dengan data sentimen"""
    
    def __init__(self, texts, labels, tokenizer, max_length=256):
        """
        Inisialisasi dataset
        
        Parameters:
        -----------
        texts : list
            Daftar teks
        labels : list
            Daftar label sentimen (0: Negatif, 1: Netral, 2: Positif)
        tokenizer : transformers.PreTrainedTokenizer
            Tokenizer IndoBERT
        max_length : int, default 256
            Panjang maksimum token (dikurangi dari 512 default untuk dataset kecil)
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Tokenisasi teks
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Hilangkan dimensi batch yang ditambahkan oleh tokenizer
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}
        
        # Tambahkan label
        encoding['labels'] = torch.tensor(label, dtype=torch.long)
        
        return encoding

class IndoBERTFineTuner:
    """Kelas untuk fine-tuning model IndoBERT untuk analisis sentimen Bank Sentral"""
    
    def __init__(self, model_name="indobenchmark/indobert-base-p1", num_labels=3, 
                output_dir="../models/fine-tuned-indobert"):
        """
        Inisialisasi fine-tuner
        
        Parameters:
        -----------
        model_name : str, default "indobenchmark/indobert-base-p1"
            Nama model IndoBERT dari Hugging Face
        num_labels : int, default 3
            Jumlah kelas sentimen (3 untuk Negatif, Netral, Positif)
        output_dir : str, default "../models/fine-tuned-indobert"
            Direktori untuk menyimpan model hasil fine-tuning
        """
        self.model_name = model_name
        self.num_labels = num_labels
        self.output_dir = output_dir
        
        # Periksa dan buat direktori output jika belum ada
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Cek apakah menggunakan GPU
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else 
                                 "mps" if torch.backends.mps.is_available() else 
                                 "cpu")
        
        logger.info(f"Menggunakan device: {self.device}")
        
        # Muat tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Muat model untuk klasifikasi sequence
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            output_attentions=False,
            output_hidden_states=False
        )
        
        # Pindahkan model ke device yang sesuai
        self.model.to(self.device)
        
        # Metrik pelatihan
        self.training_stats = []
    
    def prepare_data(self, annotated_data, text_col='Text', label_col='Sentiment',
                    train_size=0.8, random_state=42):
        """
        Menyiapkan data untuk fine-tuning
        
        Parameters:
        -----------
        annotated_data : pd.DataFrame
            DataFrame berisi teks yang telah dianotasi dengan sentimen
        text_col : str, default 'Text'
            Nama kolom yang berisi teks
        label_col : str, default 'Sentiment'
            Nama kolom yang berisi label sentimen
        train_size : float, default 0.8
            Proporsi data yang digunakan untuk pelatihan
        random_state : int, default 42
            Seed untuk random state
        
        Returns:
        --------
        tuple
            (train_dataset, val_dataset)
        """
        # Pastikan hanya mengambil data yang memiliki anotasi
        data = annotated_data.dropna(subset=[label_col])
        
        # Konversi label ke integer
        label_map = {'Negatif': 0, 'Netral': 1, 'Positif': 2}
        
        if data[label_col].dtype == 'object':
            # Jika label berupa string
            data['label_id'] = data[label_col].map(label_map)
        else:
            # Jika label sudah berupa integer
            data['label_id'] = data[label_col]
        
        # Split data menjadi train dan validation
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            data[text_col].values, 
            data['label_id'].values,
            train_size=train_size,
            stratify=data['label_id'].values,  # Stratify berdasarkan label
            random_state=random_state
        )
        
        # Membuat dataset
        train_dataset = SentimentDataset(train_texts, train_labels, self.tokenizer)
        val_dataset = SentimentDataset(val_texts, val_labels, self.tokenizer)
        
        logger.info(f"Jumlah data training: {len(train_dataset)}")
        logger.info(f"Jumlah data validasi: {len(val_dataset)}")
        
        # Beri tahu distribusi kelas
        train_label_dist = pd.Series(train_labels).value_counts().sort_index()
        val_label_dist = pd.Series(val_labels).value_counts().sort_index()
        
        logger.info(f"Distribusi label training: {train_label_dist.to_dict()}")
        logger.info(f"Distribusi label validasi: {val_label_dist.to_dict()}")
        
        return train_dataset, val_dataset
    
    def fine_tune(self, train_dataset, val_dataset, batch_size=2, 
                gradient_accumulation_steps=4, epochs=10, 
                learning_rate=3e-5, epsilon=1e-8, weight_decay=0.01,
                warmup_steps=0, seed=42):
        """
        Melakukan fine-tuning model
        
        Parameters:
        -----------
        train_dataset : SentimentDataset
            Dataset untuk pelatihan
        val_dataset : SentimentDataset
            Dataset untuk validasi
        batch_size : int, default 2
            Ukuran batch untuk pelatihan (dikurangi untuk dataset kecil)
        gradient_accumulation_steps : int, default 4
            Jumlah langkah untuk akumulasi gradien
        epochs : int, default 10
            Jumlah epoch untuk pelatihan (ditingkatkan untuk dataset kecil)
        learning_rate : float, default 3e-5
            Learning rate
        epsilon : float, default 1e-8
            Epsilon untuk optimizer
        weight_decay : float, default 0.01
            Weight decay untuk regularisasi
        warmup_steps : int, default 0
            Jumlah langkah warmup untuk scheduler
        seed : int, default 42
            Seed untuk reproducibility
        
        Returns:
        --------
        dict
            Metrik evaluasi pada data validasi
        """
        # Set seed untuk reproducibility
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        # DataLoader
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(
            train_dataset,
            sampler=train_sampler,
            batch_size=batch_size
        )
        
        val_sampler = SequentialSampler(val_dataset)
        val_dataloader = DataLoader(
            val_dataset,
            sampler=val_sampler,
            batch_size=batch_size
        )
        
        # Optimizer
        optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            eps=epsilon,
            weight_decay=weight_decay
        )
        
        # Jumlah langkah pelatihan total (dengan gradient accumulation)
        total_steps = len(train_dataloader) * epochs // gradient_accumulation_steps
        
        # Scheduler untuk learning rate
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # Metrik untuk tracking
        best_val_accuracy = 0.0
        best_epoch = 0
        
        # Training loop
        logger.info("Memulai fine-tuning...")
        logger.info(f"Dataset kecil: menggunakan batch_size={batch_size}, gradient_accumulation_steps={gradient_accumulation_steps}")
        
        for epoch in range(epochs):
            logger.info(f"Epoch {epoch+1}/{epochs}")
            
            # Training
            self.model.train()
            total_train_loss = 0
            
            # Progress bar untuk training
            progress_bar = tqdm(train_dataloader, desc="Training", unit="batch")
            
            optimizer.zero_grad()  # Reset gradien di awal epoch
            
            for step, batch in enumerate(progress_bar):
                # Pindahkan batch ke device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(**batch)
                loss = outputs.loss / gradient_accumulation_steps  # Normalisasi loss
                
                # Backward pass
                loss.backward()
                
                # Update parameters setiap gradient_accumulation_steps
                if (step + 1) % gradient_accumulation_steps == 0 or (step + 1 == len(train_dataloader)):
                    # Clip gradient norm untuk stabilitas
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    # Update parameters
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                
                # Update total loss
                total_train_loss += loss.item() * gradient_accumulation_steps
                
                # Update progress bar
                progress_bar.set_postfix({"loss": loss.item() * gradient_accumulation_steps})
            
            # Hitung rata-rata loss
            avg_train_loss = total_train_loss / len(train_dataloader)
            logger.info(f"Rata-rata loss training: {avg_train_loss:.4f}")
            
            # Validasi
            self.model.eval()
            val_loss = 0
            val_preds = []
            val_true = []
            
            # Progress bar untuk validasi
            progress_bar = tqdm(val_dataloader, desc="Validation", unit="batch")
            
            with torch.no_grad():
                for batch in progress_bar:
                    # Pindahkan batch ke device
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    
                    # Forward pass
                    outputs = self.model(**batch)
                    loss = outputs.loss
                    
                    # Update total loss
                    val_loss += loss.item()
                    
                    # Simpan prediksi dan label sebenarnya
                    logits = outputs.logits
                    preds = torch.argmax(logits, dim=1).cpu().numpy()
                    labels = batch['labels'].cpu().numpy()
                    
                    val_preds.extend(preds)
                    val_true.extend(labels)
            
            # Hitung rata-rata loss
            avg_val_loss = val_loss / len(val_dataloader)
            logger.info(f"Rata-rata loss validasi: {avg_val_loss:.4f}")
            
            # Hitung metrik evaluasi
            val_accuracy = accuracy_score(val_true, val_preds)
            precision, recall, f1, _ = precision_recall_fscore_support(
                val_true, val_preds, average='weighted'
            )
            
            logger.info(f"Akurasi validasi: {val_accuracy:.4f}")
            logger.info(f"Presisi validasi: {precision:.4f}")
            logger.info(f"Recall validasi: {recall:.4f}")
            logger.info(f"F1-score validasi: {f1:.4f}")
            
            # Confusion matrix
            cm = confusion_matrix(val_true, val_preds)
            logger.info(f"Confusion Matrix:\n{cm}")
            
            # Simpan metrik
            self.training_stats.append({
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'val_accuracy': val_accuracy,
                'val_precision': precision,
                'val_recall': recall,
                'val_f1': f1
            })
            
            # Simpan model terbaik berdasarkan akurasi
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_epoch = epoch + 1
                
                # Simpan model
                model_path = os.path.join(self.output_dir, "best_model")
                self.model.save_pretrained(model_path)
                self.tokenizer.save_pretrained(model_path)
                
                logger.info(f"Model terbaik disimpan (epoch {best_epoch}, akurasi {best_val_accuracy:.4f})")
        
        # Simpan model terakhir
        model_path = os.path.join(self.output_dir, "last_model")
        self.model.save_pretrained(model_path)
        self.tokenizer.save_pretrained(model_path)
        
        logger.info(f"Model terakhir disimpan (epoch {epochs})")
        logger.info(f"Fine-tuning selesai. Model terbaik dari epoch {best_epoch} dengan akurasi {best_val_accuracy:.4f}")
        
        # Plot hasil training
        self.plot_training_results()
        
        # Evaluasi final pada data validasi
        final_metrics = {
            'accuracy': best_val_accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'best_epoch': best_epoch
        }
        
        return final_metrics
    
    def cross_validate(self, annotated_data, text_col='Text', label_col='Sentiment', 
                      n_splits=5, batch_size=2, epochs=5, random_state=42):
        """
        Melakukan cross-validation untuk dataset kecil
        
        Parameters:
        -----------
        annotated_data : pd.DataFrame
            DataFrame berisi teks yang telah dianotasi dengan sentimen
        text_col : str, default 'Text'
            Nama kolom yang berisi teks
        label_col : str, default 'Sentiment'
            Nama kolom yang berisi label sentimen
        n_splits : int, default 5
            Jumlah fold untuk cross-validation
        batch_size : int, default 2
            Ukuran batch untuk pelatihan
        epochs : int, default 5
            Jumlah epoch untuk setiap fold
        random_state : int, default 42
            Seed untuk reproducibility
        
        Returns:
        --------
        dict
            Hasil cross-validation
        """
        # Pastikan hanya mengambil data yang memiliki anotasi
        data = annotated_data.dropna(subset=[label_col])
        
        # Konversi label ke integer
        label_map = {'Negatif': 0, 'Netral': 1, 'Positif': 2}
        
        if data[label_col].dtype == 'object':
            # Jika label berupa string
            data['label_id'] = data[label_col].map(label_map)
        else:
            # Jika label sudah berupa integer
            data['label_id'] = data[label_col]
        
        # Setup KFold
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        
        # Persiapkan data
        texts = data[text_col].values
        labels = data['label_id'].values
        
        # Tracking metrik untuk setiap fold
        fold_metrics = []
        all_val_preds = []
        all_val_true = []
        
        logger.info(f"Memulai {n_splits}-fold cross-validation...")
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(texts)):
            logger.info(f"Fold {fold+1}/{n_splits}")
            
            # Reset model untuk setiap fold
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=self.num_labels,
                output_attentions=False,
                output_hidden_states=False
            )
            self.model.to(self.device)
            
            # Split data
            train_texts, val_texts = texts[train_idx], texts[val_idx]
            train_labels, val_labels = labels[train_idx], labels[val_idx]
            
            # Buat dataset
            train_dataset = SentimentDataset(train_texts, train_labels, self.tokenizer)
            val_dataset = SentimentDataset(val_texts, val_labels, self.tokenizer)
            
            # Fine-tune untuk fold ini
            fold_output_dir = os.path.join(self.output_dir, f"fold_{fold+1}")
            os.makedirs(fold_output_dir, exist_ok=True)
            
            # Copy model asli untuk fold ini
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=self.num_labels,
                output_attentions=False,
                output_hidden_states=False
            )
            self.model.to(self.device)
            
            # Fine-tune
            metrics = self.fine_tune(
                train_dataset, 
                val_dataset,
                batch_size=batch_size,
                epochs=epochs,
                output_dir=fold_output_dir
            )
            
            # Simpan metrik
            metrics['fold'] = fold + 1
            fold_metrics.append(metrics)
            
            # Validasi dengan model terbaik
            best_model_path = os.path.join(fold_output_dir, "best_model")
            self.model = AutoModelForSequenceClassification.from_pretrained(best_model_path)
            self.model.to(self.device)
            
            # Predict pada data validasi
            val_dataset = SentimentDataset(val_texts, val_labels, self.tokenizer)
            val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
            
            self.model.eval()
            val_preds = []
            
            with torch.no_grad():
                for batch in val_dataloader:
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    outputs = self.model(**batch)
                    logits = outputs.logits
                    preds = torch.argmax(logits, dim=1).cpu().numpy()
                    val_preds.extend(preds)
            
            all_val_preds.extend(val_preds)
            all_val_true.extend(val_labels)
        
        # Hitung metrik overall
        overall_accuracy = accuracy_score(all_val_true, all_val_preds)
        overall_precision, overall_recall, overall_f1, _ = precision_recall_fscore_support(
            all_val_true, all_val_preds, average='weighted'
        )
        
        # Rata-rata metrik per fold
        avg_metrics = {
            'accuracy': np.mean([m['accuracy'] for m in fold_metrics]),
            'precision': np.mean([m['precision'] for m in fold_metrics]),
            'recall': np.mean([m['recall'] for m in fold_metrics]),
            'f1': np.mean([m['f1'] for m in fold_metrics])
        }
        
        logger.info("Cross-validation selesai")
        logger.info(f"Rata-rata metrik: {avg_metrics}")
        logger.info(f"Overall accuracy: {overall_accuracy:.4f}")
        
        # Buat ringkasan cross-validation
        cv_result = {
            'fold_metrics': fold_metrics,
            'avg_metrics': avg_metrics,
            'overall': {
                'accuracy': overall_accuracy,
                'precision': overall_precision,
                'recall': overall_recall,
                'f1': overall_f1
            }
        }
        
        # Simpan hasil cross-validation
        cv_stats_path = os.path.join(self.output_dir, "cv_stats.txt")
        with open(cv_stats_path, 'w') as f:
            f.write(f"Cross-Validation Results ({n_splits} folds)\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("Fold Metrics:\n")
            for i, metrics in enumerate(fold_metrics):
                f.write(f"Fold {i+1}:\n")
                for k, v in metrics.items():
                    if k != 'fold':
                        f.write(f"  {k}: {v:.4f}\n")
                f.write("\n")
            
            f.write("Average Metrics:\n")
            for k, v in avg_metrics.items():
                f.write(f"  {k}: {v:.4f}\n")
            f.write("\n")
            
            f.write("Overall Metrics:\n")
            for k, v in cv_result['overall'].items():
                f.write(f"  {k}: {v:.4f}\n")
        
        logger.info(f"Cross-validation stats saved to {cv_stats_path}")
        
        return cv_result
    
    def plot_training_results(self, output_path=None):
        """
        Memplot hasil pelatihan
        
        Parameters:
        -----------
        output_path : str, optional
            Path file untuk menyimpan plot
        """
        stats_df = pd.DataFrame(self.training_stats)
        
        # Plot losses
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(stats_df['epoch'], stats_df['train_loss'], 'b-o', label='Training')
        plt.plot(stats_df['epoch'], stats_df['val_loss'], 'r-o', label='Validation')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training & Validation Loss')
        plt.legend()
        plt.grid(True)
        
        # Plot metrics
        plt.subplot(1, 2, 2)
        plt.plot(stats_df['epoch'], stats_df['val_accuracy'], 'g-o', label='Accuracy')
        plt.plot(stats_df['epoch'], stats_df['val_precision'], 'b-o', label='Precision')
        plt.plot(stats_df['epoch'], stats_df['val_recall'], 'r-o', label='Recall')
        plt.plot(stats_df['epoch'], stats_df['val_f1'], 'y-o', label='F1-score')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.title('Validation Metrics')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        
        # Simpan plot jika path disediakan
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot hasil training disimpan ke {output_path}")
        
        plt.show()
    
    def load_trained_model(self, model_path=None):
        """
        Memuat model yang telah di-fine-tuning
        
        Parameters:
        -----------
        model_path : str, optional
            Path ke model yang akan dimuat (default: best_model di output_dir)
        """
        if model_path is None:
            model_path = os.path.join(self.output_dir, "best_model")
        
        logger.info(f"Memuat model dari {model_path}")
        
        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            # Pindahkan model ke device yang sesuai
            self.model.to(self.device)
            
            logger.info("Model berhasil dimuat")
            return True
        except Exception as e:
            logger.error(f"Error saat memuat model: {e}")
            return False
    
    def perform_data_augmentation(self, texts, labels, augmentation_factor=2):
        """
        Melakukan augmentasi data untuk dataset kecil
        
        Parameters:
        -----------
        texts : list
            Daftar teks
        labels : list
            Daftar label
        augmentation_factor : int, default 2
            Faktor pengali jumlah data
            
        Returns:
        --------
        tuple
            (augmented_texts, augmented_labels)
        """
        logger.info(f"Melakukan augmentasi data (faktor {augmentation_factor}x)...")
        
        augmented_texts = texts.copy()
        augmented_labels = labels.copy()
        
        # Metode augmentasi sederhana:
        # 1. Menghapus kata secara acak (random word deletion)
        # 2. Mengacak urutan kata dalam kalimat (random word shuffle)
        
        # Augmentasi untuk setiap teks
        for idx, text in enumerate(texts):
            for _ in range(augmentation_factor - 1):  # -1 karena sudah ada teks asli
                tokens = text.split()
                
                # Skip jika terlalu sedikit token
                if len(tokens) < 5:
                    continue
                
                # Pilih metode augmentasi secara acak
                augmentation_type = random.choice(['deletion', 'shuffle'])
                
                if augmentation_type == 'deletion':
                    # Hapus 10-20% kata secara acak
                    n_to_delete = max(1, int(len(tokens) * random.uniform(0.1, 0.2)))
                    indices_to_delete = random.sample(range(len(tokens)), n_to_delete)
                    augmented_text = ' '.join([t for i, t in enumerate(tokens) if i not in indices_to_delete])
                
                elif augmentation_type == 'shuffle':
                    # Bagi teks menjadi kalimat-kalimat
                    sentences = text.split('.')
                    augmented_sentences = []
                    
                    for sentence in sentences:
                        sent_tokens = sentence.split()
                        if len(sent_tokens) > 3:  # Minimal 4 kata untuk diacak
                            # Simpan kata pertama dan terakhir, acak sisanya
                            first, *middle, last = sent_tokens
                            random.shuffle(middle)
                            shuffled_sent = ' '.join([first] + middle + [last])
                            augmented_sentences.append(shuffled_sent)
                        else:
                            augmented_sentences.append(sentence)
                    
                    augmented_text = '.'.join(augmented_sentences)
                
                # Tambahkan teks yang telah diaugmentasi
                augmented_texts.append(augmented_text)
                augmented_labels.append(labels[idx])
        
        logger.info(f"Augmentasi selesai. Data asli: {len(texts)}, Data setelah augmentasi: {len(augmented_texts)}")
        return augmented_texts, augmented_labels

# ID untuk label
id_to_label = {0: 'Negatif', 1: 'Netral', 2: 'Positif'}

# Contoh penggunaan
if __name__ == "__main__":
    # Pastikan GPU/MPS tersedia
    device = torch.device("cuda" if torch.cuda.is_available() else 
                         "mps" if torch.backends.mps.is_available() else 
                         "cpu")
    print(f"Using device: {device}")
    
    # Contoh penggunaan untuk fine-tuning IndoBERT
    try:
        # Load data yang telah dianotasi
        annotated_data = pd.read_excel("../results/annotations/annotated_data.xlsx")
        
        # Inisialisasi fine-tuner
        fine_tuner = IndoBERTFineTuner()
        
        # Cross-validation untuk dataset kecil
        print("Melakukan cross-validation untuk dataset kecil...")
        cv_result = fine_tuner.cross_validate(
            annotated_data,
            text_col='Clean_Text',  # Sesuaikan dengan nama kolom teks Anda
            label_col='Sentiment',  # Sesuaikan dengan nama kolom label Anda
            n_splits=5,            # 5-fold cross-validation
            batch_size=2,          # Batch size kecil
            epochs=5               # Epoch minimal
        )
        
        # Fine-tune dengan seluruh dataset
        print("\nMelakukan fine-tuning dengan seluruh dataset...")
        train_dataset, val_dataset = fine_tuner.prepare_data(
            annotated_data,
            text_col='Clean_Text',
            label_col='Sentiment',
            train_size=0.9  # Gunakan lebih banyak data untuk training
        )
        
        # Data augmentation untuk dataset kecil
        texts = [train_dataset.texts[i] for i in range(len(train_dataset))]
        labels = [train_dataset.labels[i] for i in range(len(train_dataset))]
        
        # Augmentasi data jika sangat sedikit (kurang dari 50 sampel)
        if len(texts) < 50:
            print("Melakukan augmentasi data karena dataset sangat kecil...")
            augmented_texts, augmented_labels = fine_tuner.perform_data_augmentation(texts, labels)
            
            # Buat dataset baru dengan data yang telah diaugmentasi
            train_dataset = SentimentDataset(augmented_texts, augmented_labels, fine_tuner.tokenizer)
            print(f"Dataset setelah augmentasi: {len(train_dataset)} sampel")
        
        # Fine-tuning
        metrics = fine_tuner.fine_tune(
            train_dataset,
            val_dataset,
            batch_size=2,           # Ukuran batch kecil
            gradient_accumulation_steps=4,  # Akumulasi gradien
            epochs=10,              # Lebih banyak epoch untuk dataset kecil
            learning_rate=3e-5      # Learning rate sedikit lebih tinggi
        )
        
        print(f"Fine-tuning selesai dengan akurasi: {metrics['accuracy']:.4f}")
        
        # Simpan model
        output_path = "../results/fine_tuning_results.png"
        fine_tuner.plot_training_results(output_path)
        print(f"Plot hasil fine-tuning disimpan ke {output_path}")
        
    except Exception as e:
        print(f"Error dalam fine-tuning: {e}")