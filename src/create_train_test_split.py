import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def create_train_test_split(input_file, output_dir, test_size=0.2, random_state=42):
    """
    Membagi dataset hasil anotasi menjadi train dan test set
    
    Parameters:
    -----------
    input_file : str
        Path ke file Excel dengan data anotasi gabungan (merged_annotations.xlsx)
    output_dir : str
        Direktori untuk menyimpan hasil pembagian
    test_size : float
        Persentase data untuk test set
    random_state : int
        Seed untuk reproducibility
    """
    print(f"Membuat pembagian train-test dari file: {input_file}")
    
    # Buat direktori jika belum ada
    os.makedirs(output_dir, exist_ok=True)
    
    # Baca data
    try:
        data = pd.read_excel(input_file)
        print(f"Data berhasil dibaca: {len(data)} paragraf dari {data['ID_Dokumen'].nunique()} dokumen")
    except Exception as e:
        print(f"Error membaca file: {e}")
        return
    
    # Pastikan kolom yang diperlukan ada
    required_columns = ['ID_Dokumen', 'ID_Paragraf', 'Teks_Paragraf', 'Sentimen', 'Topik_Utama', 'Keyword']
    missing_columns = [col for col in required_columns if col not in data.columns]
    
    if missing_columns:
        print(f"Kolom yang diperlukan tidak ditemukan: {missing_columns}")
        print(f"Kolom yang tersedia: {data.columns.tolist()}")
        return
    
    # Jika ada kolom Anotasi_Final, gunakan itu sebagai sumber sentimen
    if 'Anotasi_Final' in data.columns:
        data['Sentimen_Final'] = data['Anotasi_Final']
    else:
        data['Sentimen_Final'] = data['Sentimen']
    
    # Get unique document IDs
    unique_docs = data['ID_Dokumen'].unique()
    n_docs = len(unique_docs)
    
    # Stratifikasi berdasarkan proporsi sentimen di setiap dokumen jika memungkinkan
    # Hitung proporsi sentimen per dokumen
    sentiments_by_doc = {}
    for doc in unique_docs:
        doc_data = data[data['ID_Dokumen'] == doc]
        sentiment_counts = doc_data['Sentimen_Final'].value_counts(normalize=True)
        
        # Get the dominant sentiment
        if not sentiment_counts.empty:
            dominant = sentiment_counts.idxmax()
            sentiments_by_doc[doc] = dominant
        else:
            sentiments_by_doc[doc] = 'Unknown'
    
    # Buat array untuk stratifikasi
    doc_sentiments = [sentiments_by_doc[doc] for doc in unique_docs]
    
    # Split at document level (stratified by dominant sentiment if possible)
    try:
        train_docs, test_docs = train_test_split(
            unique_docs,
            test_size=test_size,
            random_state=random_state,
            stratify=doc_sentiments
        )
    except ValueError:
        # Jika stratifikasi gagal (misalnya karena terlalu sedikit sampel), gunakan tanpa stratifikasi
        print("Stratifikasi berdasarkan sentimen gagal, menggunakan split acak")
        train_docs, test_docs = train_test_split(
            unique_docs,
            test_size=test_size,
            random_state=random_state,
            stratify=None
        )
    
    # Create train and test datasets
    train_data = data[data['ID_Dokumen'].isin(train_docs)].copy()
    test_data = data[data['ID_Dokumen'].isin(test_docs)].copy()
    
    print(f"Train set: {len(train_docs)} dokumen ({len(train_data)} paragraf)")
    print(f"Test set: {len(test_docs)} dokumen ({len(test_data)} paragraf)")
    
    # Tambahkan kolom untuk identifikasi split
    data['Split'] = 'train'
    data.loc[data['ID_Dokumen'].isin(test_docs), 'Split'] = 'test'
    
    # Simpan hasil
    with pd.ExcelWriter(os.path.join(output_dir, 'train_test_split.xlsx')) as writer:
        data.to_excel(writer, sheet_name='Full_Dataset', index=False)
        train_data.to_excel(writer, sheet_name='Train_Set', index=False)
        test_data.to_excel(writer, sheet_name='Test_Set', index=False)
        
        # Tambahkan statistik
        stats_train = train_data['Sentimen_Final'].value_counts().reset_index()
        stats_train.columns = ['Sentimen', 'Jumlah_Train']
        
        stats_test = test_data['Sentimen_Final'].value_counts().reset_index()
        stats_test.columns = ['Sentimen', 'Jumlah_Test']
        
        stats = pd.merge(stats_train, stats_test, on='Sentimen', how='outer').fillna(0)
        stats['Total'] = stats['Jumlah_Train'] + stats['Jumlah_Test']
        stats['Proporsi_Train'] = stats['Jumlah_Train'] / stats['Total']
        stats['Proporsi_Test'] = stats['Jumlah_Test'] / stats['Total']
        
        stats.to_excel(writer, sheet_name='Statistik', index=False)
    
    # Simpan file CSV terpisah untuk train dan test
    train_data.to_csv(os.path.join(output_dir, 'train_data.csv'), index=False)
    test_data.to_csv(os.path.join(output_dir, 'test_data.csv'), index=False)
    
    print(f"Pembagian train-test selesai. File disimpan di: {output_dir}")
    
    return train_data, test_data

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Membagi dataset anotasi menjadi train dan test set')
    parser.add_argument('--input', type=str, default='results/annotations/merged_annotations.xlsx', 
                       help='Path ke file input (merged_annotations.xlsx)')
    parser.add_argument('--output_dir', type=str, default='results/train_test', 
                       help='Direktori untuk output')
    parser.add_argument('--test_size', type=float, default=0.2, 
                       help='Proporsi data untuk test set (default: 0.2)')
    parser.add_argument('--seed', type=int, default=42, 
                       help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    create_train_test_split(
        input_file=args.input,
        output_dir=args.output_dir,
        test_size=args.test_size,
        random_state=args.seed
    )