#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
preprocess.py - Script untuk preprocessing data komunikasi Bank Sentral
"""

import pandas as pd
import numpy as np
import re
import os
import logging
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from datetime import datetime
from tqdm import tqdm
import argparse
import sys

# Import from local modules
from utils import setup_device, configure_logging, load_data, save_results
from indobert_model import IndoBERTModel

# Pastikan path dapat diakses
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Download NLTK resources (jika belum)
nltk.download('punkt', quiet=True)

# Setup logging
logger = configure_logging(
    log_file=os.path.join('logs', f'preprocess_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
)

# Coba import Sastrawi untuk Bahasa Indonesia
try:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
    stemmer_factory = StemmerFactory()
    stemmer = stemmer_factory.create_stemmer()
    stopword_factory = StopWordRemoverFactory()
    stopword_remover = stopword_factory.create_stop_word_remover()
    sastrawi_stopwords = stopword_factory.get_stop_words()
    use_sastrawi = True
    logger.info("Sastrawi tersedia dan akan digunakan")
except ImportError:
    use_sastrawi = False
    sastrawi_stopwords = []
    logger.warning("Sastrawi tidak tersedia, menggunakan metode standar")

class TextPreprocessor:
    """
    Kelas untuk melakukan preprocessing text data komunikasi Bank Sentral
    """
    
    def __init__(self, use_indobert=True, indobert_model=None):
        """
        Inisialisasi preprocessor
        
        Parameters:
        -----------
        use_indobert : bool, default True
            Apakah akan menggunakan IndoBERT untuk tokenisasi dan embedding
        indobert_model : IndoBERTModel, optional
            Model IndoBERT yang sudah diinisialisasi
        """
        self.use_sastrawi = use_sastrawi
        self.use_indobert = use_indobert
        
        # Setup device
        self.device = setup_device()
        
        # Inisialisasi IndoBERT model jika diperlukan
        if use_indobert:
            if indobert_model is not None:
                self.indobert = indobert_model
            else:
                logger.info("Inisialisasi model IndoBERT baru...")
                self.indobert = IndoBERTModel()
                
            # Cek jika model berhasil dimuat
            if self.indobert.model is None:
                logger.warning("Model IndoBERT tidak berhasil dimuat. Fallback ke metode standar.")
                self.use_indobert = False
    
    def clean_text(self, text, preserve_case=True, preserve_punct=True):
        """
        Membersihkan teks dari artefak yang tidak diinginkan
        
        Parameters:
        -----------
        text : str
            Teks yang akan dibersihkan
        preserve_case : bool, default True
            Jika True, tidak mengubah case teks
        preserve_punct : bool, default True
            Jika True, mempertahankan tanda baca
            
        Returns:
        --------
        str
            Teks yang sudah dibersihkan
        """
        if not isinstance(text, str) or not text.strip():
            return ''
        
        # Buat salinan teks
        cleaned = text
        
        # Ubah ke lowercase jika diminta
        if not preserve_case:
            cleaned = cleaned.lower()
        
        # Hapus HTML tags
        cleaned = re.sub(r'<.*?>', '', cleaned)
        
        # Hapus URL
        cleaned = re.sub(r'http\S+|www\S+|https\S+', '', cleaned)
        
        # Hapus karakter non-alfanumerik dan tanda baca sesuai parameter
        if preserve_punct:
            # Bersihkan teks tetapi pertahankan tanda baca yang umum
            cleaned = re.sub(r'[^\w\s.,!?:;()\-]', '', cleaned)
        else:
            # Hapus semua tanda baca dan karakter non-alfanumerik
            cleaned = re.sub(r'[^\w\s]', '', cleaned)
        
        # Hapus angka (opsional, dikomentari untuk mempertahankan angka)
        # cleaned = re.sub(r'\d+', '', cleaned)
        
        # Hapus extra whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        return cleaned
    
    def clean_text_for_tokens(self, text):
        """
        Membersihkan teks khusus untuk tokenisasi dan analisis leksikal
        
        Parameters:
        -----------
        text : str
            Teks yang akan dibersihkan
            
        Returns:
        --------
        str
            Teks yang sudah dibersihkan untuk tokenisasi
        """
        if not isinstance(text, str) or not text.strip():
            return ''
        
        # Ubah ke lowercase
        cleaned = text.lower()
        
        # Hapus HTML tags
        cleaned = re.sub(r'<.*?>', '', cleaned)
        
        # Hapus URL
        cleaned = re.sub(r'http\S+|www\S+|https\S+', '', cleaned)
        
        # Hapus tanda baca
        cleaned = re.sub(r'[^\w\s]', '', cleaned)
        
        # Hapus angka
        cleaned = re.sub(r'\d+', '', cleaned)
        
        # Hapus extra whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        # Hapus stopwords jika Sastrawi tersedia
        if self.use_sastrawi:
            cleaned = stopword_remover.remove(cleaned)
        
        return cleaned
    
    def tokenize_text(self, text):
        """
        Tokenisasi teks menggunakan IndoBERT atau NLTK
        
        Parameters:
        -----------
        text : str
            Teks yang akan ditokenisasi
            
        Returns:
        --------
        list
            Daftar token
        """
        if not isinstance(text, str) or not text.strip():
            return []
        
        if self.use_indobert and hasattr(self, 'indobert') and self.indobert.tokenizer is not None:
            # Tokenisasi dengan IndoBERT
            try:
                tokens = self.indobert.tokenizer.tokenize(text)
                # Hapus token khusus dari IndoBERT
                tokens = [t for t in tokens if not t.startswith('##') and t not in ['[CLS]', '[SEP]']]
                return tokens
            except Exception as e:
                logger.error(f"Error saat tokenisasi dengan IndoBERT: {e}")
                logger.info("Fallback ke tokenisasi standar NLTK.")
        
        # Tokenisasi standar dengan NLTK
        return word_tokenize(text)
    
    def extract_sentences(self, text):
        """
        Ekstraksi kalimat dari teks
        
        Parameters:
        -----------
        text : str
            Teks yang akan diekstrak kalimatnya
            
        Returns:
        --------
        list
            Daftar kalimat
        """
        if not isinstance(text, str) or not text.strip():
            return []
        
        # Tokenisasi kalimat menggunakan NLTK
        sentences = sent_tokenize(text)
        
        # Filter kalimat kosong
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    def extract_paragraphs(self, text):
        """
        Ekstraksi paragraf dari teks
        
        Parameters:
        -----------
        text : str
            Teks yang akan diekstrak paragrafnya
            
        Returns:
        --------
        list
            Daftar paragraf
        """
        if not isinstance(text, str) or not text.strip():
            return []
        
        # Split berdasarkan baris kosong atau kombinasi newline
        paragraphs = re.split(r'\n\s*\n|\r\n\s*\r\n|\r\s*\r', text)
        
        # Filter paragraf kosong
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        # Jika tidak ditemukan paragraf dengan metode di atas, coba metode lain
        if len(paragraphs) <= 1 and len(text) > 100:
            # Coba split berdasarkan newline
            paragraphs = re.split(r'\n', text)
            paragraphs = [p.strip() for p in paragraphs if p.strip()]
            
            # Jika masih sedikit, coba split berdasarkan indentasi atau pola awal paragraf
            if len(paragraphs) <= 1:
                paragraphs = re.split(r'\n\s+', text)
                paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        return paragraphs
    
    def generate_embeddings(self, text):
        """
        Menghasilkan embeddings untuk teks menggunakan IndoBERT
        
        Parameters:
        -----------
        text : str
            Teks yang akan dihasilkan embeddingnya
            
        Returns:
        --------
        numpy.ndarray
            Vektor embedding
        """
        if not self.use_indobert or not hasattr(self, 'indobert') or self.indobert.model is None:
            logger.warning("IndoBERT tidak tersedia untuk generate embeddings")
            return None
        
        return self.indobert.generate_embeddings(text)
    
    def batch_generate_embeddings(self, texts, batch_size=4):
        """
        Menghasilkan embeddings untuk batch teks
        
        Parameters:
        -----------
        texts : list
            Daftar teks yang akan dihasilkan embeddingnya
        batch_size : int, default 4
            Ukuran batch
            
        Returns:
        --------
        numpy.ndarray
            Matriks embedding
        """
        if not self.use_indobert or not hasattr(self, 'indobert') or self.indobert.model is None:
            logger.warning("IndoBERT tidak tersedia untuk batch generate embeddings")
            return None
        
        return self.indobert.batch_generate_embeddings(texts, batch_size=batch_size)
    
    def preprocess_dataframe(self, df, content_col='konten', date_col='tanggal', 
                            generate_embeddings=False, batch_size=4):
        """
        Preprocessing DataFrame komunikasi Bank Sentral
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame yang berisi data komunikasi
        content_col : str, default 'konten'
            Nama kolom yang berisi teks
        date_col : str, default 'tanggal'
            Nama kolom yang berisi tanggal
        generate_embeddings : bool, default False
            Apakah akan menghasilkan embeddings
        batch_size : int, default 4
            Ukuran batch untuk generate embeddings
            
        Returns:
        --------
        pd.DataFrame
            DataFrame yang sudah dipreprocess
        """
        logger.info("Memulai preprocessing DataFrame...")
        
        # Buat salinan DataFrame
        processed_df = df.copy()
        
        # Pastikan kolom tanggal dalam format datetime
        if date_col in processed_df.columns:
            processed_df[date_col] = pd.to_datetime(processed_df[date_col])
            
            # Ekstrak tahun, bulan, dan kuartal
            processed_df['Tahun'] = processed_df[date_col].dt.year
            processed_df['Bulan'] = processed_df[date_col].dt.month
            processed_df['Kuartal'] = processed_df[date_col].dt.quarter
            
            logger.info(f"Ekstraksi fitur tanggal selesai: Tahun, Bulan, Kuartal")
        
        # Preprocessing teks
        logger.info(f"Preprocessing teks dari kolom '{content_col}'...")
        
        if content_col in processed_df.columns:
            # Clean text (2 versi)
            processed_df['Clean_Text'] = processed_df[content_col].progress_apply(
                lambda x: self.clean_text(x, preserve_case=True, preserve_punct=True)
            )
            
            processed_df['Clean_Text_For_Tokens'] = processed_df[content_col].progress_apply(
                lambda x: self.clean_text_for_tokens(x)
            )
            
            # Tokenisasi
            logger.info("Melakukan tokenisasi...")
            processed_df['Tokens'] = processed_df['Clean_Text_For_Tokens'].progress_apply(
                lambda x: self.tokenize_text(x)
            )
            processed_df['Tokens_Count'] = processed_df['Tokens'].apply(len)
            
            # Ekstrak kalimat
            logger.info("Ekstraksi kalimat...")
            processed_df['Sentences'] = processed_df[content_col].progress_apply(
                lambda x: self.extract_sentences(x)
            )
            processed_df['Sentences_Count'] = processed_df['Sentences'].apply(len)
            
            # Ekstrak paragraf
            logger.info("Ekstraksi paragraf...")
            processed_df['Paragraphs'] = processed_df[content_col].progress_apply(
                lambda x: self.extract_paragraphs(x)
            )
            processed_df['Paragraphs_Count'] = processed_df['Paragraphs'].apply(len)
            
            # Generate embeddings jika diminta
            if generate_embeddings and self.use_indobert:
                logger.info("Menghasilkan embeddings menggunakan IndoBERT...")
                
                # Ambil teks yang dibersihkan
                texts = processed_df['Clean_Text'].tolist()
                
                # Generate embeddings dalam batch
                embeddings = self.batch_generate_embeddings(texts, batch_size=batch_size)
                
                if embeddings is not None:
                    # Simpan embeddings dalam format list (untuk disimpan di Excel/CSV)
                    processed_df['Embeddings'] = [emb.tolist() for emb in embeddings]
                    logger.info("Embeddings berhasil dihasilkan")
                else:
                    logger.warning("Gagal menghasilkan embeddings")
        else:
            logger.error(f"Kolom '{content_col}' tidak ditemukan dalam DataFrame")
        
        logger.info("Preprocessing DataFrame selesai")
        return processed_df

def main():
    """
    Fungsi utama untuk menjalankan preprocessing
    """
    # Setup argument parser
    parser = argparse.ArgumentParser(description='Preprocessing data komunikasi Bank Sentral')
    parser.add_argument('--input', type=str, required=True, help='Path ke file input (Excel/CSV)')
    parser.add_argument('--output', type=str, required=True, help='Path ke file output (Excel/CSV)')
    parser.add_argument('--content-col', type=str, default='konten', help='Nama kolom konten')
    parser.add_argument('--date-col', type=str, default='tanggal', help='Nama kolom tanggal')
    parser.add_argument('--use-indobert', action='store_true', help='Gunakan IndoBERT untuk tokenisasi dan embeddings')
    parser.add_argument('--generate-embeddings', action='store_true', help='Hasilkan embeddings menggunakan IndoBERT')
    parser.add_argument('--batch-size', type=int, default=4, help='Ukuran batch untuk generate embeddings')
    
    args = parser.parse_args()
    
    # Load data
    df = load_data(args.input)
    
    if df is None or df.empty:
        logger.error(f"Gagal memuat data dari {args.input}")
        return
    
    # Inisialisasi preprocessor
    preprocessor = TextPreprocessor(use_indobert=args.use_indobert)
    
    # Tambahkan tqdm ke pandas apply
    tqdm.pandas(desc="Processing")
    
    # Preprocessing
    processed_df = preprocessor.preprocess_dataframe(
        df,
        content_col=args.content_col,
        date_col=args.date_col,
        generate_embeddings=args.generate_embeddings,
        batch_size=args.batch_size
    )
    
    # Simpan hasil
    save_results(processed_df, args.output)
    
    logger.info(f"Preprocessing selesai. Hasil disimpan ke {args.output}")

if __name__ == "__main__":
    main()