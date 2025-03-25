import streamlit as st
import pandas as pd
import numpy as np
import os
import re
import datetime
from io import BytesIO
import base64
import sys
import matplotlib.pyplot as plt
import seaborn as sns

# Pastikan path ke src dapat diakses
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modul lokal jika diperlukan
from src.utils import configure_logging, load_data, save_results

# Setup logging
import logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Konstanta
RESULTS_DIR = "results"
ANNOTATIONS_DIR = os.path.join(RESULTS_DIR, "annotations")
MASTER_FILE = os.path.join(RESULTS_DIR, "data_for_annotation.xlsx")

# Pastikan folder annotations ada
os.makedirs(ANNOTATIONS_DIR, exist_ok=True)

# Fungsi untuk memuat data anotasi master
def load_annotation_data(file_path=MASTER_FILE):
    """
    Memuat data anotasi dari file Excel
    """
    try:
        df = pd.read_excel(file_path)
        logger.info(f"Loaded annotation data: {len(df)} rows")
        return df
    except Exception as e:
        logger.error(f"Error loading annotation data: {e}")
        st.error(f"Error loading data: {e}")
        return None

# Fungsi alokasi dokumen untuk anotator
def allocate_documents(df, annotator_name):
    """
    Mengalokasikan dokumen untuk anotator
    
    Metode: setiap anotator mendapatkan 1/3 dari total dokumen,
    didistribusikan secara berurutan berdasarkan ID Dokumen
    """
    if df is None or len(df) == 0:
        return pd.DataFrame()
    
    # Strategi 1: Berdasarkan ID_Dokumen modulo 3
    unique_docs = df['ID_Dokumen'].unique()
    
    if annotator_name == "Annotator 1":
        assigned_docs = [doc_id for i, doc_id in enumerate(unique_docs) if i % 3 == 0]
    elif annotator_name == "Annotator 2":
        assigned_docs = [doc_id for i, doc_id in enumerate(unique_docs) if i % 3 == 1]
    else:  # Annotator 3
        assigned_docs = [doc_id for i, doc_id in enumerate(unique_docs) if i % 3 == 2]
    
    # Filter DataFrame untuk hanya menyertakan dokumen yang dialokasikan
    annotator_df = df[df['ID_Dokumen'].isin(assigned_docs)].copy()
    
    logger.info(f"Allocated {len(assigned_docs)} documents ({len(annotator_df)} paragraphs) to {annotator_name}")
    return annotator_df

# Fungsi untuk menyimpan hasil anotasi
def save_annotation_data(df, annotator_name):
    """
    Menyimpan data anotasi ke file Excel
    """
    if df is None or len(df) == 0:
        logger.warning("No data to save")
        return None
    
    # Buat timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Nama file untuk versi terbaru
    latest_file = os.path.join(ANNOTATIONS_DIR, f"latest_{annotator_name.replace(' ', '_')}.xlsx")
    
    # Nama file dengan timestamp untuk backup
    backup_file = os.path.join(ANNOTATIONS_DIR, f"{annotator_name.replace(' ', '_')}_{timestamp}.xlsx")
    
    try:
        # Simpan versi terbaru
        df.to_excel(latest_file, index=False)
        logger.info(f"Saved annotation data to {latest_file}")
        
        # Simpan backup
        df.to_excel(backup_file, index=False)
        logger.info(f"Saved backup to {backup_file}")
        
        return latest_file
    except Exception as e:
        logger.error(f"Error saving annotation data: {e}")
        st.error(f"Error saving data: {e}")
        return None

# Fungsi untuk menggabungkan hasil anotasi dari semua anotator
def merge_annotations():
    """
    Menggabungkan semua hasil anotasi
    """
    annotator_files = [
        os.path.join(ANNOTATIONS_DIR, "latest_Annotator_1.xlsx"),
        os.path.join(ANNOTATIONS_DIR, "latest_Annotator_2.xlsx"),
        os.path.join(ANNOTATIONS_DIR, "latest_Annotator_3.xlsx")
    ]
    
    all_dfs = []
    for file in annotator_files:
        if os.path.exists(file):
            try:
                df = pd.read_excel(file)
                all_dfs.append(df)
                logger.info(f"Loaded {len(df)} rows from {file}")
            except Exception as e:
                logger.error(f"Error loading {file}: {e}")
    
    if not all_dfs:
        logger.warning("No annotation files found to merge")
        return None
    
    try:
        # Gabungkan semua DataFrame
        merged_df = pd.concat(all_dfs, ignore_index=True)
        
        # Urutkan berdasarkan ID_Dokumen dan ID_Paragraf
        merged_df.sort_values(['ID_Dokumen', 'ID_Paragraf'], inplace=True)
        
        # Simpan hasil gabungan
        output_file = os.path.join(ANNOTATIONS_DIR, "merged_annotations.xlsx")
        merged_df.to_excel(output_file, index=False)
        logger.info(f"Saved merged annotations to {output_file}")
        
        return output_file
    except Exception as e:
        logger.error(f"Error merging annotations: {e}")
        return None

# Fungsi untuk generate link download
def get_binary_file_downloader_html(file_path, file_label='File'):
    """
    Membuat link HTML untuk download file
    """
    with open(file_path, 'rb') as f:
        data = f.read()
    
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{os.path.basename(file_path)}">{file_label}</a>'
    return href

# Fungsi untuk visualisasi statistik anotasi
def plot_annotation_stats(df):
    """
    Memplot statistik anotasi
    """
    if df is None or len(df) == 0:
        return None
    
    # Progress anotasi
    total = len(df)
    completed = df['Anotasi_Final'].sum()
    
    # Distribusi sentimen
    sentiment_counts = df[df['Anotasi_Final']]['Sentimen'].value_counts()
    
    # Buat plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Progress bar
    ax1.bar(['Completed', 'Pending'], [completed, total-completed], color=['green', 'lightgray'])
    ax1.set_title(f'Annotation Progress: {completed}/{total} ({completed/total*100:.1f}%)')
    
    # Plot 2: Sentiment distribution
    if len(sentiment_counts) > 0:
        colors = {'Positif': 'green', 'Netral': 'blue', 'Negatif': 'red'}
        sentiment_counts.plot(kind='bar', ax=ax2, color=[colors.get(x, 'gray') for x in sentiment_counts.index])
        ax2.set_title('Sentiment Distribution')
    else:
        ax2.text(0.5, 0.5, 'No annotations yet', ha='center', va='center')
        ax2.set_title('Sentiment Distribution')
    
    plt.tight_layout()
    return fig

# Fungsi utama aplikasi Streamlit
def main():
    # Setup page config
    st.set_page_config(
        page_title="Bank Sentral Annotation Tool",
        page_icon="📊",
        layout="wide"
    )
    
    # Judul dan deskripsi
    st.title("Tool Anotasi Komunikasi Bank Sentral")
    st.markdown("""
    Tool ini digunakan untuk anotasi data komunikasi Bank Sentral.
    Setiap anotator dialokasikan beberapa dokumen untuk dianotasi.
    """)
    
    # Sidebar untuk pemilihan anotator
    st.sidebar.title("Anotator")
    annotator_name = st.sidebar.selectbox(
        "Pilih Anotator:",
        ["Annotator 1", "Annotator 2", "Annotator 3", "Admin"]
    )
    
    # Load data anotasi master
    master_df = load_annotation_data(MASTER_FILE)
    
    if master_df is None:
        st.error(f"Data anotasi master tidak ditemukan di {MASTER_FILE}")
        st.info("Pastikan file data_for_annotation.xlsx sudah dibuat dengan menjalankan create_annotation.py")
        return
    
    # Mode Admin
    if annotator_name == "Admin":
        admin_interface(master_df)
        return
    
    # Alokasi dokumen untuk anotator
    annotator_file = os.path.join(ANNOTATIONS_DIR, f"latest_{annotator_name.replace(' ', '_')}.xlsx")
    
    # Check if annotator already has a saved file
    if os.path.exists(annotator_file):
        annotator_df = pd.read_excel(annotator_file)
        st.sidebar.success(f"Memuat data anotasi yang tersimpan")
    else:
        # Allocate new documents
        annotator_df = allocate_documents(master_df, annotator_name)
        # Save initial allocation
        save_annotation_data(annotator_df, annotator_name)
    
    # Navigation in sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Pilih Halaman:",
        ["Anotasi", "Statistik", "Download"]
    )
    
    # Display appropriate page
    if page == "Anotasi":
        annotation_page(annotator_df, annotator_name)
    elif page == "Statistik":
        statistics_page(annotator_df, annotator_name)
    else:  # Download
        download_page(annotator_df, annotator_name)

# Halaman anotasi
def annotation_page(df, annotator_name):
    st.header("Anotasi Dokumen")
    
    if df is None or len(df) == 0:
        st.warning("Tidak ada dokumen yang dialokasikan untuk anotator ini")
        return
    
    # Statistik progres
    total_docs = df['ID_Dokumen'].nunique()
    total_paras = len(df)
    completed_paras = df['Anotasi_Final'].sum()
    
    # Tampilkan statistik
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Dokumen", total_docs)
    col2.metric("Total Paragraf", total_paras)
    col3.metric("Paragraf Selesai", f"{completed_paras} ({completed_paras/total_paras*100:.1f}%)")
    
    # Progress bar
    st.progress(completed_paras/total_paras)
    
    # Select document
    doc_ids = sorted(df['ID_Dokumen'].unique())
    
    # Add completion status to document selection
    doc_options = []
    for doc_id in doc_ids:
        doc_df = df[df['ID_Dokumen'] == doc_id]
        completed = doc_df['Anotasi_Final'].sum()
        total = len(doc_df)
        doc_options.append(f"Dokumen {doc_id} ({completed}/{total})")
    
    selected_doc_idx = st.selectbox("Pilih Dokumen:", range(len(doc_options)), format_func=lambda x: doc_options[x])
    selected_doc = doc_ids[selected_doc_idx]
    
    # Filter for selected document
    doc_df = df[df['ID_Dokumen'] == selected_doc]
    
    # Get document info
    doc_info = doc_df.iloc[0]
    
    # Display document info
    st.subheader(f"Dokumen {selected_doc}")
    st.write(f"**Judul:** {doc_info['Judul']}")
    st.write(f"**Tanggal:** {doc_info['Tanggal']}")
    
    # Select paragraph
    para_ids = doc_df['ID_Paragraf'].tolist()
    para_options = []
    for i, para_id in enumerate(para_ids):
        status = "✅" if doc_df.iloc[i]['Anotasi_Final'] else "⬜"
        para_options.append(f"{para_id} {status}")
    
    selected_para_idx = st.selectbox("Pilih Paragraf:", range(len(para_options)), format_func=lambda x: para_options[x])
    
    # Display selected paragraph
    row = doc_df.iloc[selected_para_idx]
    
    # Quick navigation
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬅️ Paragraf Sebelumnya") and selected_para_idx > 0:
            st.session_state.selected_para_idx = selected_para_idx - 1
            st.experimental_rerun()
    with col2:
        if st.button("Paragraf Berikutnya ➡️") and selected_para_idx < len(para_ids) - 1:
            st.session_state.selected_para_idx = selected_para_idx + 1
            st.experimental_rerun()
    
    # Display paragraph text
    st.markdown("### Teks Paragraf")
    st.markdown(f"```\n{row['Teks_Paragraf']}\n```")
    
    # Annotation form
    with st.form(key=f"annotation_form_{row['ID_Paragraf']}"):
        st.markdown("### Form Anotasi")
        
        # Sentiment selection
        sentiment = st.radio(
            "Sentimen", 
            ["Positif", "Netral", "Negatif"],
            index=["Positif", "Netral", "Negatif"].index(row['Sentimen']) if row['Sentimen'] in ["Positif", "Netral", "Negatif"] else 1
        )
        
        # Topic input
        topic = st.text_input("Topik Utama", value=row['Topik_Utama'] if pd.notna(row['Topik_Utama']) else "")
        
        # Keywords input
        keywords = st.text_area("Keywords", value=row['Keyword'] if pd.notna(row['Keyword']) else "")
        
        # Annotation status
        is_final = st.checkbox("Anotasi Final (Centang jika sudah selesai)", value=bool(row['Anotasi_Final']))
        
        # Submit button
        submitted = st.form_submit_button("Simpan Anotasi")
        
        if submitted:
            # Get index in the original dataframe
            idx = df[(df['ID_Dokumen'] == row['ID_Dokumen']) & (df['ID_Paragraf'] == row['ID_Paragraf'])].index[0]
            
            # Update the dataframe
            df.at[idx, 'Sentimen'] = sentiment
            df.at[idx, 'Topik_Utama'] = topic
            df.at[idx, 'Keyword'] = keywords
            df.at[idx, 'Anotasi_Final'] = is_final
            
            # Save to file
            save_annotation_data(df, annotator_name)
            
            st.success("Anotasi berhasil disimpan!")
            
            # Auto-navigate to next paragraph if this was marked as final
            if is_final and selected_para_idx < len(para_ids) - 1:
                st.experimental_rerun()

# Halaman statistik
def statistics_page(df, annotator_name):
    st.header("Statistik Anotasi")
    
    if df is None or len(df) == 0:
        st.warning("Tidak ada data anotasi untuk ditampilkan")
        return
    
    # Overall statistics
    total_docs = df['ID_Dokumen'].nunique()
    total_paras = len(df)
    completed_paras = df['Anotasi_Final'].sum()
    completion_rate = (completed_paras / total_paras * 100) if total_paras > 0 else 0
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Dokumen", total_docs)
    col2.metric("Total Paragraf", total_paras)
    col3.metric("Progress", f"{completion_rate:.1f}%", f"{completed_paras}/{total_paras}")
    
    # Create and display visualization
    if completed_paras > 0:
        fig = plot_annotation_stats(df)
        st.pyplot(fig)
    
    # Document-level statistics
    st.subheader("Progress per Dokumen")
    
    # Calculate stats per document
    doc_stats = []
    for doc_id in df['ID_Dokumen'].unique():
        doc_df = df[df['ID_Dokumen'] == doc_id]
        total = len(doc_df)
        completed = doc_df['Anotasi_Final'].sum()
        doc_stats.append({
            'ID_Dokumen': doc_id,
            'Total': total,
            'Completed': completed,
            'Progress': f"{completed/total*100:.1f}%" if total > 0 else "0%"
        })
    
    # Create dataframe
    doc_stats_df = pd.DataFrame(doc_stats)
    st.dataframe(doc_stats_df)

# Halaman download
def download_page(df, annotator_name):
    st.header("Download Data Anotasi")
    
    if df is None or len(df) == 0:
        st.warning("Tidak ada data anotasi untuk didownload")
        return
    
    # Save current data
    file_path = save_annotation_data(df, annotator_name)
    
    if file_path:
        # Create download link
        st.markdown(
            get_binary_file_downloader_html(file_path, f"Download Data Anotasi ({annotator_name})"),
            unsafe_allow_html=True
        )
        
        # Display file info
        st.info(f"File: {os.path.basename(file_path)}")
        st.info(f"Jumlah baris: {len(df)}")
        st.info(f"Paragraf teranotasi: {df['Anotasi_Final'].sum()}/{len(df)}")

# Interface Admin
def admin_interface(master_df):
    st.header("Admin Interface")
    
    # Password protection
    password = st.text_input("Admin Password", type="password")
    if password != "adminbanksentralnlp":  # Ganti dengan password yang lebih aman
        st.warning("Masukkan password admin untuk melanjutkan")
        return
    
    # Admin options
    st.subheader("Admin Options")
    
    tab1, tab2, tab3 = st.tabs(["Merge Annotations", "Allocation Status", "Download Files"])
    
    with tab1:
        st.write("Gabungkan hasil anotasi dari semua anotator")
        if st.button("Merge All Annotations"):
            merged_file = merge_annotations()
            if merged_file:
                st.success(f"Berhasil menggabungkan anotasi di {merged_file}")
                # Create download link
                st.markdown(
                    get_binary_file_downloader_html(merged_file, "Download Merged Annotations"),
                    unsafe_allow_html=True
                )
            else:
                st.error("Tidak ada file anotasi untuk digabungkan")
    
    with tab2:
        st.write("Status alokasi dokumen untuk setiap anotator")
        
        # Check each annotator's file
        annotator_files = [
            os.path.join(ANNOTATIONS_DIR, "latest_Annotator_1.xlsx"),
            os.path.join(ANNOTATIONS_DIR, "latest_Annotator_2.xlsx"),
            os.path.join(ANNOTATIONS_DIR, "latest_Annotator_3.xlsx")
        ]
        
        allocation_stats = []
        for i, file in enumerate(annotator_files):
            annotator = f"Annotator {i+1}"
            if os.path.exists(file):
                df = pd.read_excel(file)
                total_docs = df['ID_Dokumen'].nunique()
                total_paras = len(df)
                completed_paras = df['Anotasi_Final'].sum()
                progress = f"{completed_paras/total_paras*100:.1f}%" if total_paras > 0 else "0%"
                status = "In Progress"
            else:
                total_docs = 0
                total_paras = 0
                completed_paras = 0
                progress = "0%"
                status = "Not Started"
            
            allocation_stats.append({
                'Annotator': annotator,
                'Documents': total_docs,
                'Paragraphs': total_paras,
                'Completed': completed_paras,
                'Progress': progress,
                'Status': status
            })
        
        # Display stats
        st.table(pd.DataFrame(allocation_stats))
    
    with tab3:
        st.write("Download file anotasi")
        
        files = []
        for root, dirs, filenames in os.walk(ANNOTATIONS_DIR):
            for filename in filenames:
                if filename.endswith('.xlsx'):
                    file_path = os.path.join(root, filename)
                    files.append({
                        'Filename': filename,
                        'Path': file_path,
                        'Size': f"{os.path.getsize(file_path)/1024:.1f} KB",
                        'Modified': datetime.datetime.fromtimestamp(os.path.getmtime(file_path)).strftime("%Y-%m-%d %H:%M:%S")
                    })
        
        if files:
            files_df = pd.DataFrame(files)
            st.dataframe(files_df[['Filename', 'Size', 'Modified']])
            
            selected_file = st.selectbox("Select file to download:", files_df['Filename'])
            selected_path = files_df[files_df['Filename'] == selected_file]['Path'].values[0]
            
            st.markdown(
                get_binary_file_downloader_html(selected_path, f"Download {selected_file}"),
                unsafe_allow_html=True
            )
        else:
            st.info("No annotation files found")

# Entry point for the Streamlit app
if __name__ == "__main__":
    main()