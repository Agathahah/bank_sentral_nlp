import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import re

def create_annotation_splits(input_file, output_dir, overlap_ratio=0.15, random_state=42):
    """
    Membagi dataset untuk anotasi dengan metode partial overlap
    
    Parameters:
    -----------
    input_file : str
        Path ke file Excel dengan data paragraf
    output_dir : str
        Direktori untuk menyimpan hasil pembagian
    overlap_ratio : float
        Persentase paragraf yang akan dianotasi oleh semua anotator
    random_state : int
        Seed untuk reproducibility
    """
    print(f"Membuat pembagian anotasi dari file: {input_file}")
    
    # Buat direktori jika belum ada
    os.makedirs(output_dir, exist_ok=True)
    
    # Baca data
    try:
        data = pd.read_excel(input_file)
        print(f"Data berhasil dibaca: {len(data)} baris")
    except Exception as e:
        print(f"Error membaca file: {e}")
        return
    
    # Memeriksa kolom yang tersedia
    print(f"Kolom yang tersedia: {data.columns.tolist()}")
    
    # Memastikan kolom yang diperlukan ada atau membuat kolom baru
    # Map kolom yang ada ke nama kolom yang diharapkan
    column_mapping = {
        'tanggal': 'Tanggal',
        'Judul': 'Judul',
        'konten': 'Teks_Konten',
        'url': 'URL',
        'Hits': 'Hits'
    }
    
    # Rename kolom sesuai mapping
    data = data.rename(columns=column_mapping)
    
    # Membuat ID dokumen berdasarkan indeks
    data['ID_Dokumen'] = [f"DOC_{i+1:03d}" for i in range(len(data))]
    
    # Membagi konten menjadi paragraf
    paragraphs_data = []
    
    for idx, row in data.iterrows():
        doc_id = row['ID_Dokumen']
        content = row['Teks_Konten']
        
        if pd.isna(content) or not isinstance(content, str):
            print(f"Melewati konten kosong atau bukan string di dokumen {doc_id}")
            continue
        
        # Pisahkan konten menjadi paragraf
        # Paragraf dipisahkan oleh baris kosong atau double newline
        paragraphs = re.split(r'\n\s*\n|\r\n\s*\r\n', content)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        # Simpan paragraf
        for p_idx, paragraph in enumerate(paragraphs):
            paragraphs_data.append({
                'ID_Dokumen': doc_id,
                'ID_Paragraf': f"{doc_id}_P{p_idx+1:03d}",
                'Teks_Paragraf': paragraph,
                'Tanggal': row.get('Tanggal', None),
                'Judul': row.get('Judul', None),
                'URL': row.get('URL', None),
                'Paragraf_Ke': p_idx + 1,
                'Total_Paragraf': len(paragraphs)
            })
    
    # Buat dataframe paragraf
    paragraphs_df = pd.DataFrame(paragraphs_data)
    print(f"Total paragraf ditemukan: {len(paragraphs_df)}")
    
    # Jumlah paragraf untuk overlap set
    n_overlap = int(len(paragraphs_df) * overlap_ratio)
    print(f"Jumlah paragraf untuk overlap set: {n_overlap}")
    
    # Stratified selection untuk overlap set (1-2 paragraf per dokumen)
    overlap_indices = []
    for doc_id in paragraphs_df['ID_Dokumen'].unique():
        doc_data = paragraphs_df[paragraphs_df['ID_Dokumen'] == doc_id]
        n_select = min(2, len(doc_data))
        if n_select > 0:
            selected = doc_data.sample(n=n_select, random_state=random_state).index.tolist()
            overlap_indices.extend(selected)
    
    # Adjust to get exact desired overlap size
    np.random.seed(random_state)
    if len(overlap_indices) < n_overlap:
        non_selected = paragraphs_df.index.difference(overlap_indices)
        additional = np.random.choice(non_selected, n_overlap - len(overlap_indices), replace=False)
        overlap_indices.extend(additional)
    elif len(overlap_indices) > n_overlap:
        overlap_indices = np.random.choice(overlap_indices, n_overlap, replace=False)
    
    # Create overlap set
    overlap_set = paragraphs_df.loc[overlap_indices].copy()
    
    # Create non-overlap set and split for 3 annotators
    non_overlap = list(paragraphs_df.index.difference(overlap_indices))
    np.random.shuffle(non_overlap)
    
    n_per_annotator = len(non_overlap) // 3
    annotator_1_indices = non_overlap[:n_per_annotator]
    annotator_2_indices = non_overlap[n_per_annotator:2*n_per_annotator]
    annotator_3_indices = non_overlap[2*n_per_annotator:]
    
    # Combine with overlap set
    annotator_1_all = sorted(overlap_indices.tolist() + annotator_1_indices)
    annotator_2_all = sorted(overlap_indices.tolist() + annotator_2_indices)
    annotator_3_all = sorted(overlap_indices.tolist() + annotator_3_indices)
    
    # Create assignment dataset
    paragraphs_df['Set_Overlap'] = paragraphs_df.index.isin(overlap_indices)
    paragraphs_df['Set_Annotator_1'] = paragraphs_df.index.isin(annotator_1_all)
    paragraphs_df['Set_Annotator_2'] = paragraphs_df.index.isin(annotator_2_all)
    paragraphs_df['Set_Annotator_3'] = paragraphs_df.index.isin(annotator_3_all)
    
    # Persiapkan template untuk setiap annotator
    template_columns = ['ID_Dokumen', 'ID_Paragraf', 'Teks_Paragraf', 'Sentimen', 'Topik_Utama', 'Keyword']
    
    # Tambahkan kolom tambahan
    optional_columns = ['Tanggal', 'Judul', 'URL', 'Paragraf_Ke', 'Total_Paragraf']
    for col in optional_columns:
        if col in paragraphs_df.columns:
            template_columns.append(col)
    
    # Tambahkan kolom kosong untuk anotasi
    annotator_template = paragraphs_df[template_columns].copy()
    for col in ['Sentimen', 'Topik_Utama', 'Keyword']:
        if col in annotator_template.columns:
            annotator_template[col] = ''
        else:
            annotator_template[col] = ''
    
    # Tambahkan kolom Catatan
    annotator_template['Catatan'] = ''
    
    # Buat sheet instruksi
    instructions = pd.DataFrame({
        'Bagian': ['Definisi Sentimen', 'Definisi Sentimen', 'Definisi Sentimen', 
                   'Format Topik_Utama', 'Format Keyword', 'Catatan Penting'],
        'Kategori': ['Positif', 'Netral', 'Negatif', 'Panduan', 'Panduan', 'Panduan'],
        'Penjelasan': [
            'Paragraf mengandung tone optimis, perkembangan positif, atau kebijakan yang diharapkan berdampak baik',
            'Paragraf bersifat informatif tanpa kecenderungan, berisi data faktual tanpa penilaian',
            'Paragraf mengandung tone pesimis, penurunan indikator, risiko, tantangan, atau masalah',
            'Frasa singkat (3-7 kata) yang mewakili inti pembahasan paragraf',
            '3-7 kata kunci dipisahkan koma, fokus pada istilah teknis, indikator, atau konsep penting',
            'Tandai kasus ambigu dengan komentar di kolom Catatan. Konsistensi anotasi sangat penting.'
        ]
    })
    
    # Simpan hasil
    with pd.ExcelWriter(os.path.join(output_dir, 'annotation_assignment.xlsx')) as writer:
        # Sheet utama
        paragraphs_df.to_excel(writer, sheet_name='Dataset_Full', index=False)
        instructions.to_excel(writer, sheet_name='Instruksi', index=False)
        
        # Set overlap
        overlap_set.to_excel(writer, sheet_name='Overlap_Set', index=False)
        
        # Dataset original (sebelum pemecahan paragraf)
        data.to_excel(writer, sheet_name='Original_Data', index=False)
        
        # Statistik pembagian
        stats = pd.DataFrame({
            'Kategori': ['Total Dokumen', 'Total Paragraf', 'Overlap Set', 'Annotator 1', 'Annotator 2', 'Annotator 3'],
            'Jumlah': [
                paragraphs_df['ID_Dokumen'].nunique(),
                len(paragraphs_df), 
                len(overlap_indices), 
                len(annotator_1_all), 
                len(annotator_2_all), 
                len(annotator_3_all)
            ]
        })
        stats.to_excel(writer, sheet_name='Statistik', index=False)
    
    # Buat file untuk masing-masing anotator
    for i, indices in enumerate([annotator_1_all, annotator_2_all, annotator_3_all], 1):
        annotator_data = annotator_template.loc[indices].copy()
        
        # Flag paragraf yang termasuk overlap set
        annotator_data['Is_Overlap'] = annotator_data.index.isin(overlap_indices)
        
        output_file = os.path.join(output_dir, f'Annotator_{i}.xlsx')
        with pd.ExcelWriter(output_file) as writer:
            instructions.to_excel(writer, sheet_name='Instruksi', index=False)
            annotator_data.to_excel(writer, sheet_name=f'Anotasi', index=False)
            
            # Jika ada, tambahkan sheet overlap set terpisah
            overlap_annotator = annotator_data[annotator_data['Is_Overlap']].copy()
            if not overlap_annotator.empty:
                overlap_annotator.to_excel(writer, sheet_name='Overlap_Set', index=False)
        
        print(f"File untuk Annotator {i} dibuat: {output_file} dengan {len(annotator_data)} paragraf")
    
    print("Pembagian dataset untuk anotasi selesai!")
    return paragraphs_df

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Membagi dataset untuk anotasi dengan partial overlap')
    parser.add_argument('--input', type=str, default='data/Press_Release.xlsx', 
                       help='Path ke file input (Excel)')
    parser.add_argument('--output_dir', type=str, default='results/annotations', 
                       help='Direktori untuk output')
    parser.add_argument('--overlap', type=float, default=0.15, 
                       help='Persentase paragraf untuk overlap (default: 0.15)')
    parser.add_argument('--seed', type=int, default=42, 
                       help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    create_annotation_splits(
        input_file=args.input,
        output_dir=args.output_dir,
        overlap_ratio=args.overlap,
        random_state=args.seed
    )