#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py - Script utama untuk menjalankan pipeline analisis Bank Sentral
"""

import argparse
import os
import logging
from src.preprocess import IndoBERTPreprocessor
from src.indobert_model import IndoBERTModel as IndoBERTSentimentAnalyzer
from src.fine_tuning import IndoBERTFineTuner

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Pipeline Analisis Bank Sentral dengan IndoBERT")
    parser.add_argument("--preprocess", action="store_true", help="Jalankan preprocessing")
    parser.add_argument("--fine-tune", action="store_true", help="Jalankan fine-tuning")
    parser.add_argument("--analyze", action="store_true", help="Jalankan analisis sentimen")
    parser.add_argument("--batch-size", type=int, default=2, help="Ukuran batch untuk fine-tuning")
    parser.add_argument("--epochs", type=int, default=5, help="Jumlah epoch untuk fine-tuning")
    
    args = parser.parse_args()
    
    # Buat direktori yang diperlukan
    os.makedirs("results", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    if args.preprocess:
        # Jalankan preprocessing
        preprocessor = IndoBERTPreprocessor("data/Press_Release.xlsx")
        if preprocessor.load_data():
            data = preprocessor.preprocess_all_documents(use_bert=True)
            preprocessor.save_preprocessed_data("results/preprocessed_data.xlsx")
    
    if args.fine_tune:
        # Jalankan fine-tuning
        fine_tuner = IndoBERTFineTuner(output_dir="models/fine-tuned-indobert")
        # Asumsi data anotasi tersedia
        # Ganti dengan path data anotasi Anda
        annotated_data = pd.read_excel("results/annotated_data.xlsx")
        
        train_dataset, val_dataset = fine_tuner.prepare_data(
            annotated_data,
            text_col="Text",  # Sesuaikan dengan nama kolom Anda
            label_col="Sentiment"  # Sesuaikan dengan nama kolom Anda
        )
        
        fine_tuner.fine_tune(
            train_dataset,
            val_dataset,
            batch_size=args.batch_size,
            epochs=args.epochs
        )
    
    if args.analyze:
        # Jalankan analisis sentimen
        analyzer = IndoBERTSentimentAnalyzer(
            model_path="models/fine-tuned-indobert/best_model",
            lexicon_path="results/sentiment_lexicon.xlsx"
        )
        
        # Memuat data
        data = pd.read_excel("data/Press_Release.xlsx")
        
        # Analisis sentimen
        results = analyzer.batch_analyze(data["Isi"], batch_size=4)
        
        # Simpan hasil
        data["Sentiment"] = [r["label"] for r in results]
        data["Sentiment_Score"] = [r["score"] for r in results]
        data.to_excel("results/sentiment_results.xlsx", index=False)
        
        # Analisis temporal
        sentiment_by_time = analyzer.analyze_temporal_sentiment(
            data, 
            text_col="Isi", 
            date_col="Tanggal"
        )
        
        # Plot hasil
        analyzer.plot_sentiment_trends(
            sentiment_by_time, 
            output_path="results/sentiment_trends.png"
        )

if __name__ == "__main__":
    main()
