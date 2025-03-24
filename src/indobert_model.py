#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
indobert_model.py - Implementasi model IndoBERT untuk proyek Bank Sentral
"""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModel
import logging
import numpy as np
import os
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IndoBERTModel:
    """
    Kelas dasar untuk model IndoBERT yang digunakan dalam proyek Bank Sentral
    """
    
    def __init__(self, model_name="indobenchmark/indobert-base-p1"):
        """
        Inisialisasi model IndoBERT
        
        Parameters:
        -----------
        model_name : str, default "indobenchmark/indobert-base-p1"
            Nama model IndoBERT dari Hugging Face
        """
        self.model_name = model_name
        
        # Cek apakah menggunakan GPU
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else 
                                 "mps" if torch.backends.mps.is_available() else 
                                 "cpu")
        
        logger.info(f"Menggunakan device: {self.device}")
        
        # Inisialisasi model dan tokenizer
        self.load_model()
    
    def load_model(self):
        """
        Memuat model dan tokenizer IndoBERT
        """
        try:
            # Muat tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Muat model
            self.model = AutoModel.from_pretrained(self.model_name)
            
            # Pindahkan model ke device yang sesuai
            self.model.to(self.device)
            
            logger.info(f"Model IndoBERT berhasil dimuat dari {self.model_name}")
            
            return True
        
        except Exception as e:
            logger.error(f"Error saat memuat model: {e}")
            self.model = None
            self.tokenizer = None
            return False
    
    def generate_embeddings(self, text, max_length=256):
        """
        Menghasilkan embedding untuk teks menggunakan IndoBERT
        
        Parameters:
        -----------
        text : str
            Teks yang akan diekstrak embeddingnya
        max_length : int, default 256
            Panjang maksimum token (optimal untuk dataset kecil)
            
        Returns:
        --------
        numpy.ndarray
            Vektor embedding
        """
        if not isinstance(text, str) or not text.strip() or self.model is None:
            return np.zeros(768)  # Return vektor 0 jika teks kosong
        
        try:
            # Tokenisasi teks
            inputs = self.tokenizer(
                text, 
                truncation=True, 
                padding='max_length', 
                max_length=max_length, 
                return_tensors='pt'
            )
            
            # Pindahkan input ke device yang sama dengan model
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Lakukan inferensi
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Ambil vektor representasi dari hidden state terakhir layer terakhir
            # Gunakan [CLS] token (token pertama) sebagai representasi dokumen
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
            return embeddings[0]
        
        except Exception as e:
            logger.error(f"Error saat menghasilkan embedding: {e}")
            return np.zeros(768)  # Return vektor 0 jika terjadi error
    
    def batch_generate_embeddings(self, texts, batch_size=4, max_length=256):
        """
        Menghasilkan embedding untuk batch teks
        
        Parameters:
        -----------
        texts : list
            Daftar teks yang akan diekstrak embeddingnya
        batch_size : int, default 4
            Ukuran batch (optimal untuk MacBook M3)
        max_length : int, default 256
            Panjang maksimum token (optimal untuk dataset kecil)
            
        Returns:
        --------
        numpy.ndarray
            Matriks embedding
        """
        if self.model is None:
            return np.zeros((len(texts), 768))
        
        all_embeddings = []
        
        # Progress bar
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
            batch_texts = texts[i:i+batch_size]
            
            # Skip teks kosong
            valid_indices = []
            valid_texts = []
            
            for j, text in enumerate(batch_texts):
                if isinstance(text, str) and text.strip():
                    valid_indices.append(j)
                    valid_texts.append(text)
            
            if not valid_texts:
                # Tambahkan embedding kosong untuk batch ini
                batch_embeddings = np.zeros((len(batch_texts), 768))
                all_embeddings.append(batch_embeddings)
                continue
            
            try:
                # Tokenisasi batch
                inputs = self.tokenizer(
                    valid_texts,
                    truncation=True,
                    padding='max_length',
                    max_length=max_length,
                    return_tensors='pt'
                )
                
                # Pindahkan input ke device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Lakukan inferensi
                with torch.no_grad():
                    outputs = self.model(**inputs)
                
                # Ambil embeddings
                valid_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                
                # Buat array untuk seluruh batch
                batch_embeddings = np.zeros((len(batch_texts), 768))
                
                # Isi dengan embeddings yang valid
                for j, valid_idx in enumerate(valid_indices):
                    batch_embeddings[valid_idx] = valid_embeddings[j]
                
                all_embeddings.append(batch_embeddings)
            
            except Exception as e:
                logger.error(f"Error saat batch processing: {e}")
                batch_embeddings = np.zeros((len(batch_texts), 768))
                all_embeddings.append(batch_embeddings)
        
        # Gabungkan semua batch
        return np.vstack(all_embeddings)