#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
preprocessing.py - Script untuk preprocessing data menggunakan IndoBERT
"""

import pandas as pd
import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from tqdm import tqdm
import pickle
import os
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IndoBERTPreprocessor:
    def __init__(self, excel_path, sheet_name=0, model_name="indobenchmark/indobert-base-p1"):
        """
        Inisialisasi preprocessor dengan IndoBERT untuk data siaran pers
        
        Parameters:
        -----------
        excel_path : str
            Path ke file Excel yang berisi data siaran pers
        sheet_name : int atau str, default 0
            Nama atau indeks sheet dalam file Excel
        model_name : str, default "indobenchmark/indobert-base-p1"
            Nama model IndoBERT dari Hugging Face
        """
        self.excel_path = excel_path
        self.sheet_name = sheet_name
        self.data = None
        self.model_name = model_name
        
        # Inisialisasi tokenizer dan model IndoBERT
        logger.info(f"Memuat model IndoBERT: {model_name}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            # Cek apakah menggunakan GPU
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Menggunakan device: {self.device}")
            
            # Muat model IndoBERT untuk encoding teks
            self.model = AutoModel.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()  # Set model ke mode evaluasi
            logger.info("Model IndoBERT berhasil dimuat")
        except Exception as e:
            logger.error(f"Error saat memuat model IndoBERT: {e}")
            self.tokenizer = None
            self.model = None
        
        # Stopwords untuk Bahasa Indonesia
        try:
            self.stopwords_id = set(stopwords.words('indonesian'))
            logger.info("Stopwords Bahasa Indonesia berhasil dimuat")
        except:
            logger.warning("Stopwords Bahasa Indonesia tidak tersedia, mengunduh...")
            nltk.download('stopwords')
            self.stopwords_id = set(stopwords.words('indonesian'))
        
        # Tambahan stopwords khusus domain perbankan
        self.domain_stopwords = {
            'bank', 'sentral', 'bi', 'indonesia', 'gubernur', 'deputi', 'direktur',
            'rapat', 'dewan', 'the', 'and', 'for', 'that', 'dengan', 'dalam', 'pada',
            'dari', 'yang', 'dan', 'ini', 'itu', 'atau', 'juga', 'untuk', 'oleh', 'di',
            'ke', 'tidak', 'akan', 'telah', 'sebagai', 'atas', 'serta', 'sedangkan',
            'sementara', 'yaitu', 'yakni', 'bahwa', 'menteri', 'kebijakan'
        }
        
        # Gabungkan semua stopwords
        self.all_stopwords = self.stopwords_id.union(self.domain_stopwords)
    
    def load_data(self):
        """
        Memuat data dari file Excel dan melakukan preprocessing awal
        
        Returns:
        --------
        bool
            True jika berhasil, False jika gagal
        """
        try:
            self.data = pd.read_excel(self.excel_path, sheet_name=self.sheet_name)
            logger.info(f"Data berhasil dimuat dengan {self.data.shape[0]} baris dan {self.data.shape[1]} kolom")
            
            # Pastikan nama kolom yang dibutuhkan tersedia
            required_columns = ['Tanggal', 'Judul', 'Isi']
            missing_columns = [col for col in required_columns if col not in self.data.columns]
            
            if missing_columns:
                # Jika kolom yang dibutuhkan tidak ada, coba tebak kolom dari data
                logger.warning(f"Kolom yang dibutuhkan tidak ditemukan: {missing_columns}")
                logger.info("Mencoba menebak kolom dari data yang tersedia...")
                
                if len(self.data.columns) >= 3:
                    # Coba menetapkan kolom berdasarkan posisi
                    date_col = self.data.columns[0]
                    title_col = self.data.columns[1]
                    content_col = self.data.columns[2]
                    
                    # Memeriksa apakah kolom tanggal berisi tanggal
                    if pd.api.types.is_datetime64_any_dtype(self.data[date_col]) or "tanggal" in date_col.lower():
                        # Ganti nama kolom
                        self.data = self.data.rename(columns={
                            date_col: 'Tanggal',
                            title_col: 'Judul',
                            content_col: 'Isi'
                        })
                        logger.info("Kolom berhasil ditetapkan berdasarkan posisi")
                    else:
                        raise ValueError("Format kolom tanggal tidak terdeteksi dengan benar")
                else:
                    raise ValueError("Jumlah kolom terlalu sedikit untuk menebak struktur data")
            
            # Konversi kolom tanggal ke datetime jika belum
            if not pd.api.types.is_datetime64_any_dtype(self.data['Tanggal']):
                self.data['Tanggal'] = pd.to_datetime(self.data['Tanggal'], errors='coerce')
            
            # Mengecek dan menangani nilai yang hilang
            missing_values = self.data.isnull().sum()
            if missing_values.sum() > 0:
                logger.info(f"Nilai yang hilang dalam data: \n{missing_values}")
                
                # Menghapus baris dengan nilai yang hilang di kolom penting
                self.data = self.data.dropna(subset=['Isi']).reset_index(drop=True)
                logger.info(f"Data setelah menangani nilai yang hilang: {self.data.shape[0]} baris")
            
            # Membuat kolom tahun dan bulan untuk analisis temporal
            self.data['Tahun'] = self.data['Tanggal'].dt.year
            self.data['Bulan'] = self.data['Tanggal'].dt.month
            self.data['Kuartal'] = self.data['Tanggal'].dt.quarter
            
            # Menambahkan kolom untuk hasil preprocessing
            self.data['Clean_Text'] = None
            self.data['BERT_Embedding'] = None
            self.data['Tokens'] = None
            self.data['Sentences'] = None
            
            return True
        
        except Exception as e:
            logger.error(f"Error saat memuat data: {e}")
            return False
    
    def text_preprocessing(self, text):
        """
        Melakukan preprocessing teks dasar
        
        Parameters:
        -----------
        text : str
            Teks yang akan di-preprocessing
            
        Returns:
        --------
        str
            Teks yang sudah di-preprocessing
        """
        if not isinstance(text, str):
            return ""
        
        # Mengkonversi ke lowercase
        text = text.lower()
        
        # Menghapus angka (tetapi simpan persentase dan nilai penting)
        text = re.sub(r'(?<!\d)\d+(?!\d%)', ' ', text)  # Menghapus angka yang bukan bagian dari persentase
        
        # Menghapus email dan URL
        text = re.sub(r'\S*@\S*\s?', '', text)
        text = re.sub(r'http\S+', '', text)
        
        # Menghapus karakter khusus dan whitespace berlebih, tetapi simpan tanda baca penting
        text = re.sub(r'[^\w\s.,;:?!%-]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize_text(self, text):
        """
        Melakukan tokenisasi teks dan menghapus stopwords
        
        Parameters:
        -----------
        text : str
            Teks yang akan ditokenisasi
            
        Returns:
        --------
        list
            Daftar token setelah menghapus stopwords
        """
        # Tokenisasi
        tokens = word_tokenize(text)
        
        # Menghapus stopwords
        tokens = [word for word in tokens if word not in self.all_stopwords and len(word) > 1]
        
        return tokens
    
    def extract_paragraphs(self, text):
        """
        Mengekstrak paragraf dari teks
        
        Parameters:
        -----------
        text : str
            Teks yang akan diekstrak paragrafnya
            
        Returns:
        --------
        list
            Daftar paragraf dalam teks
        """
        if not isinstance(text, str):
            return []
        
        # Membagi teks menjadi paragraf berdasarkan baris kosong
        paragraphs = re.split(r'\n\s*\n', text)
        
        # Membersihkan paragraf
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        return paragraphs
    
    def generate_bert_embeddings(self, text, max_length=512):
        """
        Menghasilkan embedding menggunakan IndoBERT
        
        Parameters:
        -----------
        text : str
            Teks yang akan diekstrak embeddingnya
        max_length : int, default 512
            Panjang maksimum token untuk IndoBERT
            
        Returns:
        --------
        numpy.ndarray
            Vektor embedding
        """
        if not isinstance(text, str) or not text.strip() or self.tokenizer is None or self.model is None:
            return np.zeros(768)  # Return vektor 0 jika teks kosong atau model tidak tersedia
        
        try:
            # Tokenisasi teks menggunakan tokenizer IndoBERT
            encoded_input = self.tokenizer(text, truncation=True, padding='max_length', 
                                         max_length=max_length, return_tensors='pt')
            
            # Pindahkan input ke device yang sama dengan model
            encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
            
            # Lakukan inferensi
            with torch.no_grad():
                model_output = self.model(**encoded_input)
            
            # Ambil vektor representasi dari hidden state terakhir layer terakhir
            # Gunakan [CLS] token (token pertama) sebagai representasi dokumen
            embeddings = model_output.last_hidden_state[:, 0, :].cpu().numpy()
            
            return embeddings[0]  # Return embedding dari batch pertama
        
        except Exception as e:
            logger.error(f"Error saat menghasilkan embedding: {e}")
            return np.zeros(768)  # Return vektor 0 jika terjadi error
    
    def preprocess_all_documents(self, use_bert=True, batch_size=8):
        """
        Melakukan preprocessing untuk semua dokumen dalam dataset
        
        Parameters:
        -----------
        use_bert : bool, default True
            Apakah akan menghasilkan embedding BERT
        batch_size : int, default 8
            Ukuran batch untuk inferensi BERT
            
        Returns:
        --------
        pd.DataFrame
            DataFrame yang telah di-preprocessing
        """
        logger.info("Memulai preprocessing dokumen...")
        
        # Simpan embedding dalam list terpisah
        bert_embeddings = []
        
        for i in tqdm(range(len(self.data))):
            # Preprocessing teks
            clean_text = self.text_preprocessing(self.data.loc[i, 'Isi'])
            self.data.loc[i, 'Clean_Text'] = clean_text
            
            # Tokenisasi
            tokens = self.tokenize_text(clean_text)
            self.data.loc[i, 'Tokens'] = ' '.join(tokens)
            
            # Ekstrak paragraf
            if isinstance(self.data.loc[i, 'Isi'], str):
                paragraphs = self.extract_paragraphs(self.data.loc[i, 'Isi'])
                self.data.loc[i, 'Paragraphs'] = len(paragraphs)
            else:
                self.data.loc[i, 'Paragraphs'] = 0
            
            # Ekstrak kalimat
            if isinstance(self.data.loc[i, 'Isi'], str):
                sentences = sent_tokenize(self.data.loc[i, 'Isi'])
                self.data.loc[i, 'Sentences'] = len(sentences)
            else:
                self.data.loc[i, 'Sentences'] = 0
            
            # Generate BERT embeddings
            if use_bert and self.model is not None:
                embedding = self.generate_bert_embeddings(clean_text)
                bert_embeddings.append(embedding)
            else:
                bert_embeddings.append(None)
        
        # Simpan embedding dalam kolom terpisah
        self.data['BERT_Embedding'] = bert_embeddings
        
        logger.info("Preprocessing dokumen selesai")
        return self.data
    
    def save_preprocessed_data(self, output_path, save_embeddings=False):
        """
        Menyimpan data yang telah di-preprocessing
        
        Parameters:
        -----------
        output_path : str
            Path file untuk menyimpan data
        save_embeddings : bool, default False
            Apakah akan menyimpan embedding (disimpan terpisah jika True)
            
        Returns:
        --------
        bool
            True jika berhasil, False jika gagal
        """
        try:
            # Jika tidak menyimpan embedding, hapus kolom embedding
            if not save_embeddings:
                data_to_save = self.data.drop(columns=['BERT_Embedding'], errors='ignore')
                
                # Simpan data ke Excel
                data_to_save.to_excel(output_path, index=False)
                logger.info(f"Data tanpa embedding berhasil disimpan ke {output_path}")
            else:
                # Simpan data utama tanpa embedding
                data_to_save = self.data.drop(columns=['BERT_Embedding'], errors='ignore')
                data_to_save.to_excel(output_path, index=False)
                
                # Simpan embedding secara terpisah menggunakan pickle
                embeddings_path = output_path.replace('.xlsx', '_embeddings.pkl')
                embeddings = {i: emb for i, emb in enumerate(self.data['BERT_Embedding']) if emb is not None}
                
                with open(embeddings_path, 'wb') as f:
                    pickle.dump(embeddings, f)
                
                logger.info(f"Data tanpa embedding berhasil disimpan ke {output_path}")
                logger.info(f"Embedding berhasil disimpan ke {embeddings_path}")
            
            return True
        
        except Exception as e:
            logger.error(f"Error saat menyimpan data: {e}")
            return False
    
    def save_tokens_to_file(self, output_path):
        """
        Menyimpan semua token ke file teks
        
        Parameters:
        -----------
        output_path : str
            Path file untuk menyimpan token
            
        Returns:
        --------
        bool
            True jika berhasil, False jika gagal
        """
        try:
            tokens = []
            for token_text in self.data['Tokens']:
                if isinstance(token_text, str):
                    tokens.extend(token_text.split())
            
            with open(output_path, 'w', encoding='utf-8') as f:
                for token in tokens:
                    f.write(f"{token}\n")
            
            logger.info(f"Token berhasil disimpan ke {output_path}")
            return True
        
        except Exception as e:
            logger.error(f"Error saat menyimpan token: {e}")
            return False
    
    def extract_keyword_contexts(self, keywords, context_size=5):
        """
        Mengekstrak konteks kata kunci untuk anotasi manual
        
        Parameters:
        -----------
        keywords : list
            Daftar kata kunci yang akan diekstrak konteksnya
        context_size : int, default 5
            Jumlah kata sebelum dan sesudah kata kunci
            
        Returns:
        --------
        dict
            Dictionary berisi konteks kata kunci
        """
        contexts = {}
        
        for keyword in keywords:
            contexts[keyword] = []
            
            for i, row in self.data.iterrows():
                if not isinstance(row['Clean_Text'], str):
                    continue
                
                # Tokenisasi teks
                tokens = row['Clean_Text'].split()
                
                # Mencari indeks kata kunci
                for j, token in enumerate(tokens):
                    if token.lower() == keyword.lower():
                        # Mendapatkan konteks
                        start = max(0, j - context_size)
                        end = min(len(tokens), j + context_size + 1)
                        
                        context = " ".join(tokens[start:end])
                        
                        # Menambahkan ke daftar konteks
                        contexts[keyword].append({
                            'doc_id': i,
                            'date': row['Tanggal'],
                            'context': context
                        })
        
        return contexts