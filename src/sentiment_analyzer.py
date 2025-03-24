#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sentiment_analyzer.py - Script untuk menganalisis sentimen menggunakan IndoBERT
"""

import pandas as pd
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
import pickle

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IndoBERTSentimentAnalyzer:
    def __init__(self, model_path=None, lexicon_path=None):
        """
        Inisialisasi analyzer sentimen menggunakan IndoBERT
        
        Parameters:
        -----------
        model_path : str, optional
            Path ke model IndoBERT yang telah di-fine-tuning
            Jika None, akan menggunakan model IndoBERT dasar
        lexicon_path : str, optional
            Path ke kamus sentimen untuk analisis berbasis leksikon
        """
        self.model_path = model_path
        self.lexicon_path = lexicon_path
        
        # Lexicon sentimen jika disediakan
        self.sentiment_lexicon = None
        if lexicon_path and os.path.exists(lexicon_path):
            self.load_lexicon(lexicon_path)
        
        # Cek apakah menggunakan GPU
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else 
                                 "mps" if torch.backends.mps.is_available() else 
                                 "cpu")
        
        logger.info(f"Menggunakan device: {self.device}")
        
        # Muat model IndoBERT
        self.load_model(model_path)
        
        # Mapping label
        self.id_to_label = {0: 'Negatif', 1: 'Netral', 2: 'Positif'}
        self.label_to_id = {'Negatif': 0, 'Netral': 1, 'Positif': 2}
    
    def load_model(self, model_path=None):
        """
        Memuat model IndoBERT
        
        Parameters:
        -----------
        model_path : str, optional
            Path ke model IndoBERT
        """
        try:
            if model_path and os.path.exists(model_path):
                logger.info(f"Memuat model fine-tuned dari {model_path}")
                # Muat model yang telah di-fine-tuning
                self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            else:
                # Muat model dasar IndoBERT
                model_name = "indobenchmark/indobert-base-p1"
                logger.info(f"Memuat model dasar: {model_name}")
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    model_name,
                    num_labels=3  # Negatif, Netral, Positif
                )
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Pindahkan model ke device yang sesuai
            self.model.to(self.device)
            self.model.eval()  # Set model ke mode evaluasi
            
            logger.info("Model berhasil dimuat")
            return True
        
        except Exception as e:
            logger.error(f"Error saat memuat model: {e}")
            return False
    
    def load_lexicon(self, lexicon_path):
        """
        Memuat kamus sentimen
        
        Parameters:
        -----------
        lexicon_path : str
            Path ke file kamus sentimen
        """
        try:
            if lexicon_path.endswith('.pkl'):
                # Jika format pickle
                with open(lexicon_path, 'rb') as f:
                    self.sentiment_lexicon = pickle.load(f)
            elif lexicon_path.endswith('.xlsx'):
                # Jika format Excel
                lexicon_df = pd.read_excel(lexicon_path)
                
                self.sentiment_lexicon = {
                    'positif': [],
                    'negatif': [],
                    'netral': []
                }
                
                for _, row in lexicon_df.iterrows():
                    word = row['Kata']
                    sentiment = row['Sentimen'].lower() if isinstance(row['Sentimen'], str) else None
                    
                    if sentiment == 'positif':
                        self.sentiment_lexicon['positif'].append(word)
                    elif sentiment == 'negatif':
                        self.sentiment_lexicon['negatif'].append(word)
                    elif sentiment in ['netral', 'neutral']:
                        self.sentiment_lexicon['netral'].append(word)
            else:
                # Jika format teks
                with open(lexicon_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                self.sentiment_lexicon = {
                    'positif': [],
                    'negatif': [],
                    'netral': []
                }
                
                # Parse section dalam format teks
                positif_match = re.search(r'POSITIF:(.*?)(?=NEGATIF:|NETRAL:|$)', content, re.DOTALL)
                negatif_match = re.search(r'NEGATIF:(.*?)(?=POSITIF:|NETRAL:|$)', content, re.DOTALL)
                netral_match = re.search(r'NETRAL:(.*?)(?=POSITIF:|NEGATIF:|$)', content, re.DOTALL)
                
                if positif_match:
                    self.sentiment_lexicon['positif'] = [word.strip() for word in positif_match.group(1).strip().split('\n') if word.strip()]
                
                if negatif_match:
                    self.sentiment_lexicon['negatif'] = [word.strip() for word in negatif_match.group(1).strip().split('\n') if word.strip()]
                
                if netral_match:
                    self.sentiment_lexicon['netral'] = [word.strip() for word in netral_match.group(1).strip().split('\n') if word.strip()]
            
            # Buat set untuk pencarian yang lebih cepat
            self.positif_set = set(self.sentiment_lexicon['positif'])
            self.negatif_set = set(self.sentiment_lexicon['negatif'])
            self.netral_set = set(self.sentiment_lexicon['netral'])
            
            logger.info(f"Kamus sentimen berhasil dimuat dengan {len(self.positif_set)} kata positif, "
                      f"{len(self.negatif_set)} kata negatif, dan {len(self.netral_set)} kata netral")
            
            return True
        
        except Exception as e:
            logger.error(f"Error saat memuat kamus sentimen: {e}")
            self.sentiment_lexicon = None
            return False
    
    def analyze_with_bert(self, text, max_length=256):
        """
        Menganalisis sentimen menggunakan model IndoBERT
        
        Parameters:
        -----------
        text : str
            Teks yang akan dianalisis
        max_length : int, default 256
            Panjang maksimum token (optimal untuk dataset kecil)
        
        Returns:
        --------
        dict
            Hasil analisis sentimen
        """
        if not isinstance(text, str) or not text.strip():
            return {
                'label': 'Netral',
                'score': 0.0,
                'probabilities': {
                    'Negatif': 0.0,
                    'Netral': 1.0,
                    'Positif': 0.0
                }
            }
        
        try:
            # Tokenisasi
            inputs = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=max_length,
                return_tensors='pt'
            )
            
            # Pindahkan input ke device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Prediksi
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Ambil probabilitas kelas
            probs = torch.nn.functional.softmax(outputs.logits, dim=1).cpu().numpy()[0]
            
            # Ambil indeks kelas dengan probabilitas tertinggi
            pred_id = np.argmax(probs)
            
            # Konversi ke label
            pred_label = self.id_to_label[pred_id]
            
            # Hitung skor sentimen (0 = Negatif, 1 = Netral, 2 = Positif)
            # Normalisasi ke rentang -1 hingga 1
            sentiment_score = (pred_id - 1)  # -1, 0, 1
            
            # Tambahkan komponen dari probabilitas
            sentiment_score += (probs[2] - probs[0]) * 0.5  # Menambahkan skor berdasarkan perbedaan probabilitas
            
            # Pastikan skor dalam rentang -1 hingga 1
            sentiment_score = max(-1, min(1, sentiment_score))
            
            result = {
                'label': pred_label,
                'score': sentiment_score,
                'probabilities': {
                    'Negatif': float(probs[0]),
                    'Netral': float(probs[1]),
                    'Positif': float(probs[2])
                }
            }
            
            return result
        
        except Exception as e:
            logger.error(f"Error saat menganalisis sentimen dengan BERT: {e}")
            return {
                'label': 'Netral',
                'score': 0.0,
                'probabilities': {
                    'Negatif': 0.0,
                    'Netral': 1.0,
                    'Positif': 0.0
                },
                'error': str(e)
            }
    
    def analyze_with_lexicon(self, text):
        """
        Menganalisis sentimen menggunakan kamus sentimen
        
        Parameters:
        -----------
        text : str
            Teks yang akan dianalisis
        
        Returns:
        --------
        dict
            Hasil analisis sentimen berbasis leksikon
        """
        if not isinstance(text, str) or not text.strip() or self.sentiment_lexicon is None:
            return {
                'label': 'Netral',
                'score': 0.0,
                'counts': {
                    'positif': 0,
                    'negatif': 0,
                    'netral': 0
                }
            }
        
        try:
            # Preprocessing teks
            text = text.lower()
            
            # Tokenisasi
            tokens = word_tokenize(text)
            
            # Menghitung kata berdasarkan sentimen
            pos_count = sum(1 for token in tokens if token in self.positif_set)
            neg_count = sum(1 for token in tokens if token in self.negatif_set)
            neu_count = sum(1 for token in tokens if token in self.netral_set)
            
            total_sentiment_words = pos_count + neg_count + neu_count
            
            # Menghitung skor sentimen
            if total_sentiment_words > 0:
                score = (pos_count - neg_count) / total_sentiment_words
            else:
                score = 0.0
            
            # Menentukan label
            if score > 0.1:
                label = 'Positif'
            elif score < -0.1:
                label = 'Negatif'
            else:
                label = 'Netral'
            
            result = {
                'label': label,
                'score': score,
                'counts': {
                    'positif': pos_count,
                    'negatif': neg_count,
                    'netral': neu_count
                }
            }
            
            return result
        
        except Exception as e:
            logger.error(f"Error saat menganalisis sentimen dengan leksikon: {e}")
            return {
                'label': 'Netral',
                'score': 0.0,
                'counts': {
                    'positif': 0,
                    'negatif': 0,
                    'netral': 0
                },
                'error': str(e)
            }
    
    def analyze_sentiment(self, text, use_bert=True, use_lexicon=True):
        """
        Menganalisis sentimen menggunakan kombinasi model IndoBERT dan leksikon
        
        Parameters:
        -----------
        text : str
            Teks yang akan dianalisis
        use_bert : bool, default True
            Apakah akan menggunakan model IndoBERT
        use_lexicon : bool, default True
            Apakah akan menggunakan kamus sentimen
        
        Returns:
        --------
        dict
            Hasil analisis sentimen gabungan
        """
        if not isinstance(text, str) or not text.strip():
            return {
                'label': 'Netral',
                'score': 0.0,
                'method': 'default'
            }
        
        bert_result = None
        lexicon_result = None
        
        # Analisis dengan BERT jika diminta
        if use_bert:
            bert_result = self.analyze_with_bert(text)
        
        # Analisis dengan leksikon jika diminta dan tersedia
        if use_lexicon and self.sentiment_lexicon is not None:
            lexicon_result = self.analyze_with_lexicon(text)
        
        # Kombinasikan hasil (prioritaskan BERT jika tersedia)
        if bert_result and lexicon_result:
            # Bobot untuk setiap metode (BERT lebih diprioritaskan)
            bert_weight = 0.7
            lexicon_weight = 0.3
            
            combined_score = (bert_result['score'] * bert_weight) + (lexicon_result['score'] * lexicon_weight)
            
            # Tentukan label berdasarkan skor gabungan
            if combined_score > 0.1:
                label = 'Positif'
            elif combined_score < -0.1:
                label = 'Negatif'
            else:
                label = 'Netral'
            
            result = {
                'label': label,
                'score': combined_score,
                'method': 'combined',
                'bert_result': bert_result,
                'lexicon_result': lexicon_result
            }
        elif bert_result:
            result = {
                'label': bert_result['label'],
                'score': bert_result['score'],
                'method': 'bert',
                'bert_result': bert_result
            }
        elif lexicon_result:
            result = {
                'label': lexicon_result['label'],
                'score': lexicon_result['score'],
                'method': 'lexicon',
                'lexicon_result': lexicon_result
            }
        else:
            result = {
                'label': 'Netral',
                'score': 0.0,
                'method': 'default'
            }
        
        return result
    
    def analyze_paragraphs(self, text, use_bert=True, use_lexicon=True):
        """
        Menganalisis sentimen per paragraf dalam teks
        
        Parameters:
        -----------
        text : str
            Teks yang akan dianalisis
        use_bert : bool, default True
            Apakah akan menggunakan model IndoBERT
        use_lexicon : bool, default True
            Apakah akan menggunakan kamus sentimen
        
        Returns:
        --------
        dict
            Hasil analisis sentimen keseluruhan dan per paragraf
        """
        if not isinstance(text, str) or not text.strip():
            return {
                'overall': {
                    'label': 'Netral',
                    'score': 0.0,
                    'method': 'default'
                },
                'paragraphs': []
            }
        
        # Analisis sentimen keseluruhan
        overall_sentiment = self.analyze_sentiment(text, use_bert=use_bert, use_lexicon=use_lexicon)
        
        # Ekstrak paragraf
        paragraphs = re.split(r'\n\s*\n', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        # Analisis sentimen per paragraf
        paragraph_sentiments = []
        
        for para in paragraphs:
            sentiment = self.analyze_sentiment(para, use_bert=use_bert, use_lexicon=use_lexicon)
            
            paragraph_sentiments.append({
                'text': para,
                'sentiment': sentiment
            })
        
        return {
            'overall': overall_sentiment,
            'paragraphs': paragraph_sentiments
        }
    
    def batch_analyze(self, texts, use_bert=True, use_lexicon=True, batch_size=4):
        """
        Menganalisis sentimen untuk batch teks
        
        Parameters:
        -----------
        texts : list atau pd.Series
            Daftar teks yang akan dianalisis
        use_bert : bool, default True
            Apakah akan menggunakan model IndoBERT
        use_lexicon : bool, default True
            Apakah akan menggunakan kamus sentimen
        batch_size : int, default 4
            Ukuran batch untuk analisis BERT (optimal untuk MacBook M3)
        
        Returns:
        --------
        list
            Daftar hasil analisis sentimen
        """
        results = []
        
        # Jika kedua metode digunakan, lakukan secara terpisah untuk efisiensi
        if use_bert and use_lexicon and self.sentiment_lexicon is not None:
            # Analisis dengan BERT
            bert_results = self.batch_analyze_bert(texts, batch_size)
            
            # Analisis dengan leksikon
            lexicon_results = []
            for text in tqdm(texts, desc="Analyzing with lexicon"):
                lexicon_results.append(self.analyze_with_lexicon(text))
            
            # Kombinasikan hasil
            for bert_res, lex_res in zip(bert_results, lexicon_results):
                # Bobot untuk setiap metode
                bert_weight = 0.7
                lexicon_weight = 0.3
                
                combined_score = (bert_res['score'] * bert_weight) + (lex_res['score'] * lexicon_weight)
                
                # Tentukan label berdasarkan skor gabungan
                if combined_score > 0.1:
                    label = 'Positif'
                elif combined_score < -0.1:
                    label = 'Negatif'
                else:
                    label = 'Netral'
                
                results.append({
                    'label': label,
                    'score': combined_score,
                    'method': 'combined',
                    'bert_result': bert_res,
                    'lexicon_result': lex_res
                })
        else:
            # Analisis satu per satu
            for text in tqdm(texts, desc="Analyzing sentiment"):
                results.append(self.analyze_sentiment(text, use_bert=use_bert, use_lexicon=use_lexicon))
        
        return results
    
    def batch_analyze_bert(self, texts, batch_size=4):
        """
        Menganalisis sentimen untuk batch teks menggunakan IndoBERT
        
        Parameters:
        -----------
        texts : list atau pd.Series
            Daftar teks yang akan dianalisis
        batch_size : int, default 4
            Ukuran batch untuk analisis (optimal untuk MacBook M3)
        
        Returns:
        --------
        list
            Daftar hasil analisis sentimen dengan BERT
        """
        results = []
        
        # Pastikan model dalam mode evaluasi
        self.model.eval()
        
        # Progress bar
        for i in tqdm(range(0, len(texts), batch_size), desc="Analyzing with BERT"):
            batch_texts = texts[i:i+batch_size]
            
            # Filter teks kosong
            valid_indices = []
            valid_texts = []
            
            for j, text in enumerate(batch_texts):
                if isinstance(text, str) and text.strip():
                    valid_indices.append(j)
                    valid_texts.append(text)
            
            if not valid_texts:
                # Tambahkan hasil default untuk teks kosong
                for _ in range(len(batch_texts)):
                    results.append({
                        'label': 'Netral',
                        'score': 0.0,
                        'probabilities': {
                            'Negatif': 0.0,
                            'Netral': 1.0,
                            'Positif': 0.0
                        }
                    })
                continue
            
            try:
                # Tokenisasi batch
                inputs = self.tokenizer(
                    valid_texts,
                    truncation=True,
                    padding='max_length',
                    max_length=256,  # Optimal untuk dataset kecil
                    return_tensors='pt'
                )
                
                # Pindahkan input ke device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Prediksi
                with torch.no_grad():
                    outputs = self.model(**inputs)
                
                # Ambil probabilitas kelas
                probs = torch.nn.functional.softmax(outputs.logits, dim=1).cpu().numpy()
                
                # Ambil indeks kelas dengan probabilitas tertinggi
                pred_ids = np.argmax(probs, axis=1)
                
                # Buat daftar hasil untuk batch ini
                batch_results = []
                
                for j in range(len(batch_texts)):
                    if j in valid_indices:
                        # Indeks dalam daftar valid
                        valid_idx = valid_indices.index(j)
                        
                        # Prediksi untuk teks ini
                        pred_id = pred_ids[valid_idx]
                        prob = probs[valid_idx]
                        
                        # Konversi ke label
                        pred_label = self.id_to_label[pred_id]
                        
                        # Hitung skor sentimen
                        sentiment_score = (pred_id - 1)  # -1, 0, 1
                        
                        # Tambahkan komponen dari probabilitas
                        sentiment_score += (prob[2] - prob[0]) * 0.5
                        
                        # Pastikan skor dalam rentang -1 hingga 1
                        sentiment_score = max(-1, min(1, sentiment_score))
                        
                        batch_results.append({
                            'label': pred_label,
                            'score': float(sentiment_score),
                            'probabilities': {
                                'Negatif': float(prob[0]),
                                'Netral': float(prob[1]),
                                'Positif': float(prob[2])
                            }
                        })
                    else:
                        # Default untuk teks kosong
                        batch_results.append({
                            'label': 'Netral',
                            'score': 0.0,
                            'probabilities': {
                                'Negatif': 0.0,
                                'Netral': 1.0,
                                'Positif': 0.0
                            }
                        })
                
                results.extend(batch_results)
            
            except Exception as e:
                logger.error(f"Error saat batch processing: {e}")
                
                # Tambahkan hasil default jika terjadi error
                for _ in range(len(batch_texts)):
                    results.append({
                        'label': 'Netral',
                        'score': 0.0,
                        'probabilities': {
                            'Negatif': 0.0,
                            'Netral': 1.0,
                            'Positif': 0.0
                        },
                        'probabilities': {
                            'Negatif': 0.0,
                            'Netral': 1.0,
                            'Positif': 0.0
                        },
                        'error': str(e)
                    })
        
        return results
    
    def analyze_temporal_sentiment(self, df, text_col, date_col, time_freq='M', use_bert=True, use_lexicon=True):
        """
        Menganalisis sentimen berdasarkan waktu
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame yang berisi teks dan tanggal
        text_col : str
            Nama kolom yang berisi teks
        date_col : str
            Nama kolom yang berisi tanggal
        time_freq : str, default 'M'
            Frekuensi waktu untuk agregasi ('D', 'W', 'M', 'Q', 'Y')
        use_bert : bool, default True
            Apakah akan menggunakan model IndoBERT
        use_lexicon : bool, default True
            Apakah akan menggunakan kamus sentimen
        
        Returns:
        --------
        pd.DataFrame
            DataFrame berisi tren sentimen berdasarkan waktu
        """
        # Pastikan kolom tanggal dalam format datetime
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Analisis sentimen untuk semua teks
        logger.info("Menganalisis sentimen untuk semua teks...")
        sentiments = self.batch_analyze(df[text_col], use_bert=use_bert, use_lexicon=use_lexicon)
        
        # Tambahkan hasil ke DataFrame
        df_with_sentiment = df.copy()
        df_with_sentiment['sentiment_label'] = [s['label'] for s in sentiments]
        df_with_sentiment['sentiment_score'] = [s['score'] for s in sentiments]
        
        # Agregasi berdasarkan waktu
        logger.info(f"Agregasi sentimen berdasarkan frekuensi waktu: {time_freq}")
        
        # Buat kolom periode untuk agregasi
        df_with_sentiment['period'] = df_with_sentiment[date_col].dt.to_period(time_freq)
        
        # Agregasi
        sentiment_by_time = df_with_sentiment.groupby('period').agg(
            avg_sentiment_score=('sentiment_score', 'mean'),
            count=('sentiment_score', 'count'),
            positive_count=('sentiment_label', lambda x: (x == 'Positif').sum()),
            neutral_count=('sentiment_label', lambda x: (x == 'Netral').sum()),
            negative_count=('sentiment_label', lambda x: (x == 'Negatif').sum())
        ).reset_index()
        
        # Hitung persentase
        sentiment_by_time['positive_pct'] = sentiment_by_time['positive_count'] / sentiment_by_time['count'] * 100
        sentiment_by_time['neutral_pct'] = sentiment_by_time['neutral_count'] / sentiment_by_time['count'] * 100
        sentiment_by_time['negative_pct'] = sentiment_by_time['negative_count'] / sentiment_by_time['count'] * 100
        
        # Konversi period ke datetime untuk plotting
        sentiment_by_time['date'] = sentiment_by_time['period'].dt.to_timestamp()
        
        return sentiment_by_time
    
    def plot_sentiment_trends(self, sentiment_by_time, output_path=None):
        """
        Memplot tren sentimen berdasarkan waktu
        
        Parameters:
        -----------
        sentiment_by_time : pd.DataFrame
            DataFrame berisi tren sentimen hasil dari analyze_temporal_sentiment
        output_path : str, optional
            Path file untuk menyimpan plot
        
        Returns:
        --------
        matplotlib.figure.Figure
            Objek figure matplotlib
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        # Plot skor sentimen rata-rata
        ax1.plot(sentiment_by_time['date'], sentiment_by_time['avg_sentiment_score'], 'o-', color='blue')
        ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
        ax1.set_ylabel('Skor Sentimen Rata-rata')
        ax1.set_title('Tren Sentimen Komunikasi Bank Sentral')
        ax1.grid(True, alpha=0.3)
        
        # Plot persentase kategori sentimen
        ax2.bar(sentiment_by_time['date'], sentiment_by_time['positive_pct'], color='green', alpha=0.7, label='Positif')
        ax2.bar(sentiment_by_time['date'], sentiment_by_time['neutral_pct'], bottom=sentiment_by_time['positive_pct'], 
               color='blue', alpha=0.7, label='Netral')
        ax2.bar(sentiment_by_time['date'], sentiment_by_time['negative_pct'], 
               bottom=sentiment_by_time['positive_pct'] + sentiment_by_time['neutral_pct'], 
               color='red', alpha=0.7, label='Negatif')
        
        ax2.set_ylabel('Persentase (%)')
        ax2.set_xlabel('Waktu')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Simpan plot jika path disediakan
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot tren sentimen disimpan ke {output_path}")
        
        return fig
    
    def compare_models(self, texts, output_path=None):
        """
        Membandingkan hasil analisis sentimen antara model IndoBERT dan leksikon
        
        Parameters:
        -----------
        texts : list atau pd.Series
            Daftar teks yang akan dianalisis
        output_path : str, optional
            Path file untuk menyimpan hasil perbandingan
        
        Returns:
        --------
        tuple
            (comparison_df, agreement_stats)
        """
        if self.sentiment_lexicon is None:
            logger.warning("Kamus sentimen tidak tersedia, tidak dapat membandingkan model")
            return None, None
        
        # Analisis dengan kedua metode
        bert_results = []
        lexicon_results = []
        
        for text in tqdm(texts, desc="Comparing models"):
            bert_results.append(self.analyze_with_bert(text))
            lexicon_results.append(self.analyze_with_lexicon(text))
        
        # Buat DataFrame perbandingan
        comparison_df = pd.DataFrame({
            'Text': texts,
            'BERT_Label': [r['label'] for r in bert_results],
            'BERT_Score': [r['score'] for r in bert_results],
            'Lexicon_Label': [r['label'] for r in lexicon_results],
            'Lexicon_Score': [r['score'] for r in lexicon_results]
        })
        
        # Tambahkan kolom agreement
        comparison_df['Agreement'] = comparison_df['BERT_Label'] == comparison_df['Lexicon_Label']
        
        # Hitung statistik agreement
        total_samples = len(comparison_df)
        agreement_count = comparison_df['Agreement'].sum()
        agreement_pct = (agreement_count / total_samples) * 100
        
        # Hitung confusion matrix antara kedua model
        bert_labels = [self.label_to_id.get(label, 1) for label in comparison_df['BERT_Label']]
        lexicon_labels = [self.label_to_id.get(label, 1) for label in comparison_df['Lexicon_Label']]
        
        # Confusion matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(bert_labels, lexicon_labels)
        
        # Buat DataFrame confusion matrix
        cm_df = pd.DataFrame(
            cm, 
            index=[self.id_to_label[i] for i in range(3)],
            columns=[self.id_to_label[i] for i in range(3)]
        )
        
        agreement_stats = {
            'total_samples': total_samples,
            'agreement_count': agreement_count,
            'agreement_pct': agreement_pct,
            'confusion_matrix': cm_df
        }
        
        # Simpan hasil jika path disediakan
        if output_path:
            # Simpan perbandingan
            comparison_df.to_excel(output_path, index=False)
            
            # Simpan confusion matrix
            cm_path = output_path.replace('.xlsx', '_confusion_matrix.xlsx')
            cm_df.to_excel(cm_path)
            
            logger.info(f"Perbandingan model disimpan ke {output_path}")
            logger.info(f"Confusion matrix disimpan ke {cm_path}")
        
        return comparison_df, agreement_stats

# Contoh penggunaan
if __name__ == "__main__":
    import pandas as pd
    import os
    from datetime import datetime
    
    # Inisialisasi analyzer
    analyzer = IndoBERTSentimentAnalyzer(
        model_path="../models/fine-tuned-indobert/best_model" if os.path.exists("../models/fine-tuned-indobert/best_model") else None,
        lexicon_path="../results/lexicon/sentiment_lexicon.xlsx" if os.path.exists("../results/lexicon/sentiment_lexicon.xlsx") else None
    )
    
    try:
        # Prioritaskan data dari dataset
        data = pd.read_excel("../data/Press_Release.xlsx")
        if len(data) > 0:
            # Analisis sample teks
            sample_text = data['Isi'].iloc[0]
            sample_title = data['Judul'].iloc[0] if 'Judul' in data.columns else "Tidak ada judul"
            sample_date = data['Tanggal'].iloc[0] if 'Tanggal' in data.columns else "Tidak ada tanggal"
            
            print(f"Menganalisis sampel dari dataset ({len(data)} dokumen)")
            print(f"Judul: {sample_title}")
            print(f"Tanggal: {sample_date}")
            
            result = analyzer.analyze_sentiment(sample_text)
            
            print(f"\nHasil analisis:")
            print(f"Label sentimen: {result['label']}")
            print(f"Skor sentimen: {result['score']:.3f}")
            
            # Tampilkan hasil dari beberapa metode
            if 'method' in result and result['method'] == 'combined':
                print("\nDetail hasil kombinasi metode:")
                print(f"BERT score: {result['bert_result']['score']:.3f}")
                print(f"Lexicon score: {result['lexicon_result']['score']:.3f}")
                
                # Kata-kata sentimen dari lexicon
                if 'lexicon_result' in result and 'counts' in result['lexicon_result']:
                    counts = result['lexicon_result']['counts']
                    print(f"Kata positif: {counts['positif']}")
                    print(f"Kata negatif: {counts['negatif']}")
                    print(f"Kata netral: {counts['netral']}")
        else:
            # Fallback ke contoh hardcoded jika dataset kosong
            print("Dataset kosong. Menggunakan contoh teks...")
            
            example_text = """
            Bank Indonesia memutuskan untuk mempertahankan suku bunga acuan BI 7-Day Reverse Repo Rate (BI7DRR) sebesar 5,75%.
            Keputusan ini sejalan dengan kebijakan untuk memastikan inflasi tetap terkendali dalam sasaran 3,0±1%.
            """
            
            result = analyzer.analyze_sentiment(example_text)
            print(f"Label sentimen: {result['label']}")
            print(f"Skor sentimen: {result['score']:.3f}")
    except Exception as e:
        # Fallback ke contoh hardcoded jika dataset tidak tersedia
        print(f"Error saat memuat dataset: {e}")
        print("Menggunakan contoh teks...")
        
        example_text = """
        Bank Indonesia memutuskan untuk mempertahankan suku bunga acuan BI 7-Day Reverse Repo Rate (BI7DRR) sebesar 5,75%.
        Keputusan ini sejalan dengan kebijakan untuk memastikan inflasi tetap terkendali dalam sasaran 3,0±1%.
        """
        
        result = analyzer.analyze_sentiment(example_text)
        print(f"Label sentimen: {result['label']}")
        print(f"Skor sentimen: {result['score']:.3f}")

    # Analisis paragraf
    paragraph_result = analyzer.analyze_paragraphs(text)
    
    print("\nAnalisis per paragraf:")
    for i, para in enumerate(paragraph_result['paragraphs']):
        print(f"Paragraf {i+1}: {para['sentiment']['label']} ({para['sentiment']['score']:.2f})")
        print(f"  {para['text']}\n")