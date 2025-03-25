#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
create_annotation.py - Script untuk membuat file data anotasi

Script ini berguna untuk membuat file anotasi dari data hasil preprocessing
untuk keperluan anotasi manual. Script ini menggunakan data yang sudah dipreprocess
dan menghasilkan file anotasi yang dapat digunakan oleh tim peneliti untuk
melakukan anotasi manual.
"""

import pandas as pd
import numpy as np
import os
import re
import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import argparse
import logging
import sys
from tqdm import tqdm

# Pastikan dapat mengakses modul lokal
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

# Import dari modul lokal
from utils import configure_logging, load_data, save_results

# Setup logging
logger = configure_logging(
    log_file=os.path.join('logs', f'create_annotation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
)

# Download NLTK resources
try:
    nltk.download('punkt', quiet=True)
except:
    pass

# Fungsi untuk memuat data stopwords
def load_stopwords():
    """
    Memuat dan menggabungkan stopwords Bahasa Indonesia standar
    dengan stopwords khusus untuk konteks ekonomi dan bank sentral
    """
    try:
        from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
        from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
        
        # Cek apakah Sastrawi berhasil dimuat
        factory = StopWordRemoverFactory()
        stemmer_factory = StemmerFactory()
        stopword_remover = factory.create_stop_word_remover()
        stemmer = stemmer_factory.create_stemmer()
        
        # Test Sastrawi dengan stopword_remover
        test_text = "saya adalah contoh kalimat"
        test_result = stopword_remover.remove(test_text)
        logger.info(f"Sastrawi test: '{test_text}' -> '{test_result}'")
        
        # Test stemming
        test_stem = stemmer.stem("menjalankan")
        logger.info(f"Sastrawi stemming test: 'menjalankan' -> '{test_stem}'")
        
        # Dapatkan stopwords dasar
        stopwords = factory.get_stop_words()
        logger.info(f"Jumlah stopwords dasar Sastrawi: {len(stopwords)}")
        
        # Tambahkan stopwords khusus ekonomi/perbankan yang terlalu umum
        # dan tidak memberikan nilai analitis
        economic_stopwords = [
            'bank', 'sentral', 'indonesia', 'bi', 'persen', 'rupiah', 'triwulan', 
            'bulan', 'tahun', 'kuartal', 'rapat', 'gubernur', 'dewan',
            'anggota', 'memutuskan', 'tetap', 'tanggal', 'hari', 'triwulan', 'kuartal',
            'januari', 'februari', 'maret', 'april', 'mei', 'juni', 'juli',
            'agustus', 'september', 'oktober', 'november', 'desember',
            'sebesar', 'menjadi', 'sejalan', 'mencapai', 'tersebut', 'terhadap',
            'namun', 'serta', 'melalui', 'terkait', 'sedangkan', 'seiring', 'sementara'
        ]
        
        # Jangan jadikan kata-kata kunci berikut sebagai stopwords
        # karena penting untuk analisis komunikasi bank sentral
        exclude_from_stopwords = [
            'inflasi', 'stabilitas', 'pertumbuhan', 'nilai', 'tukar', 'suku', 'bunga',
            'likuiditas', 'defisit', 'surplus', 'cadangan', 'kredit', 'makroprudensial',
            'fiskal', 'volatilitas', 'devisa', 'global', 'domestik', 'ekspor', 'impor',
            'kebijakan', 'moneter', 'ekonomi', 'keuangan', 'pasar', 'penurunan',
            'peningkatan', 'tekanan', 'krisis', 'berlanjut', 'terjaga', 'menengah',
            'jangka', 'panjang', 'pendek', 'merespons', 'covid', 'pandemi', 'risiko',
            'neraca', 'pembayaran', 'uang', 'investasi', 'tumbuh', 'melambat'
        ]
        
        # Gabungkan stopwords standar dengan stopwords ekonomi
        combined_stopwords = list(set(stopwords + economic_stopwords))
        
        # Hapus kata-kata kunci penting dari daftar stopwords
        final_stopwords = [word for word in combined_stopwords if word not in exclude_from_stopwords]
        
        logger.info(f"Total stopwords setelah penyesuaian: {len(final_stopwords)}")
        return final_stopwords
    
    except ImportError as e:
        logger.warning(f"Sastrawi tidak tersedia: {e}")
        logger.warning("Menggunakan daftar stopwords minimal")
        # Daftar minimal stopwords untuk konteks ekonomi
        return ['yang', 'dan', 'di', 'dengan', 'untuk', 'pada', 'ini', 'itu', 
                'atau', 'adalah', 'tidak', 'dari', 'dalam', 'akan', 'telah', 
                'sebagai', 'oleh', 'secara', 'juga', 'merupakan', 'tentang',
                'dapat', 'tahun', 'bulan', 'persen', 'tanggal']
    except Exception as e:
        logger.error(f"Error saat memuat stopwords: {e}")
        return ['yang', 'dan', 'di', 'dengan', 'untuk', 'pada', 'ini', 'itu']

# Fungsi untuk ekstraksi keyword menggunakan TF-IDF
def extract_keywords_tfidf(texts, n_keywords=10, stopwords=None):
    """
    Ekstraksi keywords menggunakan TF-IDF
    
    Parameters:
    -----------
    texts : list
        Daftar teks
    n_keywords : int, default 10
        Jumlah keyword yang akan diekstrak
    stopwords : list, optional
        Daftar stopwords
        
    Returns:
    --------
    list
        Daftar keywords per dokumen
    """
    # Pastikan texts adalah list
    if isinstance(texts, str):
        texts = [texts]
    
    # Inisialisasi TF-IDF Vectorizer
    tfidf_vectorizer = TfidfVectorizer(
        max_features=500,
        stop_words=stopwords,
        ngram_range=(1, 2)  # Unigram dan Bigram
    )
    
    # Fit dan transform texts
    tfidf_matrix = tfidf_vectorizer.fit_transform(texts)
    feature_names = tfidf_vectorizer.get_feature_names_out()
    
    # Ambil top keywords untuk setiap dokumen
    keywords_per_doc = []
    for doc_idx in range(len(texts)):
        tfidf_scores = tfidf_matrix[doc_idx].toarray().flatten()
        top_indices = tfidf_scores.argsort()[-n_keywords:][::-1]
        top_keywords = [(feature_names[idx], tfidf_scores[idx]) for idx in top_indices]
        keywords_per_doc.append(top_keywords)
    
    return keywords_per_doc

# Fungsi untuk topic modeling menggunakan LDA
def perform_topic_modeling(texts, num_topics=5, num_words=10, stopwords=None):
    """
    Melakukan topic modeling menggunakan LDA
    
    Parameters:
    -----------
    texts : list
        Daftar teks
    num_topics : int, default 5
        Jumlah topik
    num_words : int, default 10
        Jumlah kata per topik
    stopwords : list, optional
        Daftar stopwords
        
    Returns:
    --------
    tuple
        (topics, doc_topics)
    """
    # Pastikan texts adalah list
    if isinstance(texts, str):
        texts = [texts]
    
    # Vectorize teks
    vectorizer = TfidfVectorizer(
        max_features=1000,
        stop_words=stopwords,
        ngram_range=(1, 1)
    )
    
    X = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    
    # Train LDA model
    lda = LatentDirichletAllocation(
        n_components=num_topics,
        max_iter=10,
        learning_method='online',
        random_state=42
    )
    
    lda.fit(X)
    
    # Ekstrak topik
    topics = []
    for topic_idx, topic in enumerate(lda.components_):
        top_words_idx = topic.argsort()[:-num_words-1:-1]
        top_words = [feature_names[i] for i in top_words_idx]
        topics.append((topic_idx, top_words))
    
    # Assign topik ke setiap dokumen
    doc_topic_dist = lda.transform(X)
    doc_topics = []
    for doc_dist in doc_topic_dist:
        topic_idx = doc_dist.argmax()
        doc_topics.append((topic_idx, topics[topic_idx][1]))
    
    return topics, doc_topics

# Fungsi untuk analisis sentimen sederhana
def simple_sentiment_analysis(text):
    """
    Analisis sentimen sederhana menggunakan leksikon
    
    Parameters:
    -----------
    text : str
        Teks yang akan dianalisis
        
    Returns:
    --------
    str
        Label sentimen ('Positif', 'Netral', 'Negatif')
    """
    # Leksikon sentimen untuk konteks ekonomi dan bank sentral
    positive_words = [
        'baik', 'positif', 'meningkat', 'tumbuh', 'stabil', 'kuat', 'surplus',
        'menguat', 'optimis', 'mendukung', 'efektif', 'berhasil', 'surplus',
        'sehat', 'maju', 'membaik', 'naik', 'berkembang', 'terjaga', 'lancar',
        'berlanjut', 'terkendali', 'kokoh', 'solid', 'ekspansif', 'peningkatan',
        'memperkuat', 'mempertahankan', 'perbaikan', 'pemulihan', 'berdaya',
        'kondusif', 'surplus', 'percepatan', 'keberlanjutan', 'pertumbuhan',
        'keseimbangan', 'memadai', 'konsolidasi', 'terjangkau', 'kompetitif'
    ]
    
    negative_words = [
        'buruk', 'negatif', 'menurun', 'turun', 'defisit', 'melemah', 'krisis',
        'resesi', 'inflasi', 'merosot', 'defisit', 'gejolak', 'volatil', 'tekanan',
        'risiko', 'lambat', 'kontraksi', 'terhambat', 'melemah', 'terganggu',
        'melambat', 'tertekan', 'stagnan', 'penurunan', 'ketidakpastian', 'guncangan',
        'keterbatasan', 'kerentanan', 'tantangan', 'hambatan', 'ketegangan',
        'ketidakseimbangan', 'perlambatan', 'memburuk', 'terkontraksi', 'terbatas',
        'kekhawatiran', 'stagnasi', 'ketimpangan', 'pengangguran', 'terdampak'
    ]
    
    # Preprocessing teks
    text = text.lower()
    words = word_tokenize(text)
    
    # Hitung kata positif dan negatif
    pos_count = sum(1 for word in words if word in positive_words)
    neg_count = sum(1 for word in words if word in negative_words)
    
    # Tentukan sentimen
    if pos_count > neg_count:
        return 'Positif'
    elif neg_count > pos_count:
        return 'Negatif'
    else:
        return 'Netral'

# Fungsi untuk mempersiapkan data anotasi
def prepare_annotation_data(df, output_file='data_for_annotation.xlsx', use_sentiment_analyzer=False):
    """
    Menyiapkan data untuk anotasi manual
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame hasil preprocessing
    output_file : str, default 'data_for_annotation.xlsx'
        Path file output untuk data anotasi
    use_sentiment_analyzer : bool, default False
        Apakah akan menggunakan IndoBERTSentimentAnalyzer untuk analisis sentimen
        
    Returns:
    --------
    pd.DataFrame
        DataFrame untuk anotasi
    """
    logger.info("Menyiapkan data untuk anotasi manual...")
    
    # Buat dataframe untuk anotasi
    annotation_data = []
    
    # Muat stopwords yang sudah disesuaikan untuk konteks ekonomi
    stopwords = load_stopwords()
    
    # Coba impor sentiment analyzer jika diminta
    sentiment_analyzer = None
    if use_sentiment_analyzer:
        try:
            from sentiment_analyzer import IndoBERTSentimentAnalyzer
            sentiment_analyzer = IndoBERTSentimentAnalyzer()
            logger.info("IndoBERTSentimentAnalyzer berhasil diinisialisasi")
        except ImportError as e:
            logger.warning(f"Tidak dapat mengimpor IndoBERTSentimentAnalyzer: {e}")
            logger.info("Menggunakan analisis sentimen sederhana")
        except Exception as e:
            logger.error(f"Error saat inisialisasi IndoBERTSentimentAnalyzer: {e}")
            logger.info("Menggunakan analisis sentimen sederhana")
    
    # Pastikan kolom tanggal dalam format datetime
    if 'tanggal' in df.columns:
        try:
            df['tanggal'] = pd.to_datetime(df['tanggal'])
            logger.info("Kolom tanggal berhasil dikonversi ke format datetime")
        except Exception as e:
            logger.warning(f"Tidak dapat mengkonversi kolom tanggal: {e}")
    
    # Ekstrak paragraf dari semua dokumen dan persiapkan untuk anotasi
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Memproses Dokumen"):
        doc_id = idx + 1  # ID dokumen
        
        # Ambil paragraf
        if 'Paragraphs' in row:
            # Pastikan paragraphs adalah list
            if isinstance(row['Paragraphs'], str):
                try:
                    paragraphs = eval(row['Paragraphs'])
                except:
                    paragraphs = []
            else:
                paragraphs = row['Paragraphs']
        else:
            paragraphs = []
        
        # Jika tidak ada paragraf, gunakan teks lengkap sebagai satu paragraf
        if not paragraphs and 'konten' in row:
            paragraphs = [row['konten']]
        elif not paragraphs and 'Clean_Text' in row:
            paragraphs = [row['Clean_Text']]
        
        # Extract document date
        doc_date = None
        if 'tanggal' in row and not pd.isna(row['tanggal']):
            doc_date = row['tanggal']
            logger.debug(f"Document {doc_id} date: {doc_date}")
        else:
            logger.warning(f"No valid date found for document {doc_id}")
        
        # Ekstrak keywords dari dokumen menggunakan TF-IDF
        doc_text = row['Clean_Text'] if 'Clean_Text' in row else (row['konten'] if 'konten' in row else "")
        doc_keywords = extract_keywords_tfidf([doc_text], n_keywords=10, stopwords=stopwords)[0]
        doc_keywords_str = ', '.join([kw[0] for kw in doc_keywords])
        
        # Analisis sentimen awal
        doc_sentiment = "Netral"
        if sentiment_analyzer is not None:
            try:
                result = sentiment_analyzer.analyze_sentiment(doc_text)
                doc_sentiment = result['label']
            except Exception as e:
                logger.error(f"Error saat analisis sentimen dengan IndoBERT: {e}")
                # Fallback ke analisis sentimen sederhana
                doc_sentiment = simple_sentiment_analysis(doc_text)
        else:
            # Gunakan analisis sentimen sederhana
            doc_sentiment = simple_sentiment_analysis(doc_text)
        
        # Ambil topik utama menggunakan LDA
        if idx % 100 == 0:  # Lakukan topic modeling setiap 100 dokumen untuk efisiensi
            corpus_size = min(100, len(df))
            start_idx = max(0, idx - corpus_size + 1)
            all_texts = []
            for i in range(start_idx, idx + 1):
                if i < len(df):
                    text = df.iloc[i]['Clean_Text'] if 'Clean_Text' in df.iloc[i] else ""
                    if text:
                        all_texts.append(text)
            
            topics, doc_topics = perform_topic_modeling(all_texts, num_topics=5, stopwords=stopwords)
            topic_idx = len(all_texts) - 1  # Indeks untuk dokumen saat ini
            if topic_idx >= 0 and topic_idx < len(doc_topics):
                doc_topic = doc_topics[topic_idx]
                topic_words = ', '.join(doc_topic[1])
            else:
                topic_words = ""
        else:
            topic_words = ""
        
        # Tambahkan setiap paragraf ke dataframe anotasi
        for para_idx, para_text in enumerate(paragraphs):
            if not isinstance(para_text, str) or len(para_text.strip()) < 10:  # Skip paragraf yang terlalu pendek
                continue
                
            # Generate ID unik untuk paragraf
            para_id = f"{doc_id}_{para_idx+1}"
            
            # Analisis sentimen untuk paragraf
            para_sentiment = doc_sentiment
            if sentiment_analyzer is not None:
                try:
                    result = sentiment_analyzer.analyze_sentiment(para_text)
                    para_sentiment = result['label']
                except Exception as e:
                    logger.error(f"Error saat analisis sentimen paragraf dengan IndoBERT: {e}")
                    # Fallback ke analisis sentimen sederhana
                    para_sentiment = simple_sentiment_analysis(para_text)
            else:
                # Gunakan analisis sentimen sederhana
                para_sentiment = simple_sentiment_analysis(para_text)
            
            # Tambahkan ke data anotasi
            annotation_data.append({
                'ID_Dokumen': doc_id,
                'ID_Paragraf': para_id,
                'Teks_Paragraf': para_text,
                'Sentimen': para_sentiment,  # Pre-filled sentiment
                'Topik_Utama': topic_words,  # Pre-filled topic
                'Keyword': doc_keywords_str,  # Pre-filled keywords
                'Tanggal': doc_date,  # Gunakan tanggal yang sudah diekstrak
                'Judul': row['Judul'] if 'Judul' in row else "",
                'URL': row['url'] if 'url' in row else (row['URL'] if 'URL' in row else ""),
                'Anotasi_Final': False  # Flag untuk menandai anotasi yang sudah final
            })
    
    # Buat DataFrame dari data anotasi
    annotation_df = pd.DataFrame(annotation_data)
    
    # Debug info
    logger.info(f"Jumlah paragraf yang dianotasi: {len(annotation_df)}")
    logger.info(f"Kolom-kolom dalam df anotasi: {annotation_df.columns.tolist()}")
    logger.info(f"Sample tanggal: {annotation_df['Tanggal'].head()}")
    
    # Simpan ke Excel
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        annotation_df.to_excel(output_file, index=False)
        logger.info(f"Data anotasi disimpan ke {output_file}")
    
    return annotation_df

def main():
    """
    Fungsi utama untuk menjalankan pembuatan file anotasi
    """
    # Setup argument parser
    parser = argparse.ArgumentParser(description='Membuat file anotasi dari data hasil preprocessing')
    parser.add_argument('--input', type=str, required=True, help='Path ke file input (hasil preprocessing)')
    parser.add_argument('--output', type=str, required=True, help='Path ke file output untuk anotasi')
    parser.add_argument('--use-sentiment-analyzer', action='store_true', help='Gunakan IndoBERTSentimentAnalyzer untuk analisis sentimen')
    
    args = parser.parse_args()
    
    # Load data
    df = load_data(args.input)
    
    if df is None or df.empty:
        logger.error(f"Gagal memuat data dari {args.input}")
        return
    
    # Tampilkan informasi tentang kolom df
    logger.info(f"Kolom-kolom dalam DataFrame: {df.columns.tolist()}")
    if 'tanggal' in df.columns:
        logger.info(f"Sample tanggal dari data input: {df['tanggal'].head()}")
    
    # Tambahkan tqdm ke pandas apply
    tqdm.pandas(desc="Processing")
    
    # Persiapkan data anotasi
    annotation_df = prepare_annotation_data(
        df,
        output_file=args.output,
        use_sentiment_analyzer=args.use_sentiment_analyzer
    )
    
    logger.info(f"Pembuatan file anotasi selesai. Hasil disimpan ke {args.output}")
    
    # Export dalam format khusus keyword jika diminta
    keyword_output = args.output.replace('.xlsx', '_keywords.xlsx')
    keyword_df = annotation_df[['ID_Dokumen', 'ID_Paragraf', 'Keyword', 'Tanggal', 'Judul']]
    keyword_df.to_excel(keyword_output, index=False)
    logger.info(f"Data khusus keyword disimpan ke {keyword_output}")

if __name__ == "__main__":
    main()