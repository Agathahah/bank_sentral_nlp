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

# Pastikan path ke src dapat diakses
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Konstanta
RESULTS_DIR = "results"
ANNOTATIONS_DIR = os.path.join(RESULTS_DIR, "annotations")
MASTER_FILE = os.path.join(RESULTS_DIR, "data_for_annotation.xlsx")

# Pastikan folder annotations ada
os.makedirs(ANNOTATIONS_DIR, exist_ok=True)

# Fungsi untuk membagi dokumen berdasarkan ID (1/3 untuk setiap anotator)
def get_annotator_documents(annotator_name):
    """Mendapatkan ID dokumen untuk anotator tertentu"""
    df = pd.read_excel(MASTER_FILE)
    unique_docs = sorted(df['ID_Dokumen'].unique())
    total_docs = len(unique_docs)
    
    # Pembagian yang jelas: anotator 1 mendapat 1/3 pertama, dst.
    if annotator_name == "Annotator 1":
        start_idx = 0
        end_idx = total_docs // 3
    elif annotator_name == "Annotator 2":
        start_idx = total_docs // 3
        end_idx = 2 * (total_docs // 3)
    else:  # Annotator 3
        start_idx = 2 * (total_docs // 3)
        end_idx = total_docs
    
    # Pilih dokumen untuk anotator ini
    assigned_docs = unique_docs[start_idx:end_idx]
    
    # Filter data
    annotator_data = df[df['ID_Dokumen'].isin(assigned_docs)].copy()
    return annotator_data

# Fungsi untuk membuat link download
def get_download_link(df, filename="anotasi.xlsx"):
    """Membuat link download untuk file Excel"""
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='openpyxl')
    df.to_excel(writer, index=False)
    writer.close()
    
    excel_data = output.getvalue()
    b64 = base64.b64encode(excel_data).decode()
    href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}">Download {filename}</a>'
    return href

# Tampilkan halaman anotasi untuk anotator
def show_annotation_page(annotator_name):
    """Tampilkan halaman anotasi untuk anotator tertentu"""
    st.title(f"Tool Anotasi Bank Sentral - {annotator_name}")
    
    # Muat data untuk anotator ini
    annotator_file = os.path.join(ANNOTATIONS_DIR, f"{annotator_name.replace(' ', '_')}.xlsx")
    
    # Cek jika file anotator sudah ada
    if os.path.exists(annotator_file):
        df = pd.read_excel(annotator_file)
        st.sidebar.success(f"Memuat data anotasi yang sudah ada")
    else:
        # Buat pembagian baru
        df = get_annotator_documents(annotator_name)
        # Simpan pembagian awal
        os.makedirs(ANNOTATIONS_DIR, exist_ok=True)
        df.to_excel(annotator_file, index=False)
        st.sidebar.info(f"Membuat pembagian tugas baru")
    
    # Tampilkan statistik
    total_docs = df['ID_Dokumen'].nunique()
    total_paras = len(df)
    completed = df['Anotasi_Final'].sum()
    
    # Progress bar
    st.sidebar.subheader("Progress Anotasi")
    st.sidebar.progress(completed/total_paras if total_paras > 0 else 0)
    st.sidebar.write(f"Selesai: {completed}/{total_paras} paragraf ({round(completed/total_paras*100 if total_paras > 0 else 0, 1)}%)")
    
    # Tampilkan dokumen untuk dianotasi
    st.subheader("Dokumen untuk Dianotasi")
    
    # Pilih dokumen
    doc_ids = sorted(df['ID_Dokumen'].unique())
    
    # Buat opsi dokumen dengan info progress
    doc_options = []
    for doc_id in doc_ids:
        doc_df = df[df['ID_Dokumen'] == doc_id]
        doc_completed = doc_df['Anotasi_Final'].sum()
        doc_total = len(doc_df)
        doc_status = "✅" if doc_completed == doc_total else "⏳" if doc_completed > 0 else "⬜"
        doc_options.append(f"Dokumen {doc_id} {doc_status} ({doc_completed}/{doc_total})")
    
    selected_doc_idx = st.selectbox("Pilih Dokumen:", range(len(doc_options)), format_func=lambda x: doc_options[x])
    selected_doc = doc_ids[selected_doc_idx]
    
    # Filter untuk dokumen terpilih
    doc_df = df[df['ID_Dokumen'] == selected_doc]
    
    # Tampilkan info dokumen
    doc_info = doc_df.iloc[0]
    st.write(f"**Tanggal:** {doc_info['Tanggal']}")
    st.write(f"**Judul:** {doc_info['Judul']}")
    
    # Pilih paragraf
    para_ids = doc_df['ID_Paragraf'].tolist()
    para_options = []
    for i, para_id in enumerate(para_ids):
        status = "✅" if doc_df.iloc[i]['Anotasi_Final'] else "⬜"
        para_options.append(f"{para_id} {status}")
    
    selected_para_idx = st.selectbox("Pilih Paragraf:", range(len(para_options)), format_func=lambda x: para_options[x])
    row = doc_df.iloc[selected_para_idx]
    
    # Tampilkan teks paragraf
    st.markdown("### Teks Paragraf")
    st.markdown(f"```\n{row['Teks_Paragraf']}\n```")
    
    # Form anotasi
    with st.form(key=f"form_{row['ID_Paragraf']}"):
        st.markdown("### Anotasi")
        
        # Sentimen
        sentiment = st.radio(
            "Sentimen", 
            ["Positif", "Netral", "Negatif"],
            index=["Positif", "Netral", "Negatif"].index(row['Sentimen']) if row['Sentimen'] in ["Positif", "Netral", "Negatif"] else 1
        )
        
        # Topik
        topic = st.text_input("Topik Utama", value=row['Topik_Utama'] if pd.notna(row['Topik_Utama']) else "")
        
        # Keywords
        keywords = st.text_area("Keywords", value=row['Keyword'] if pd.notna(row['Keyword']) else "")
        
        # Status anotasi
        is_final = st.checkbox("Anotasi Final", value=bool(row['Anotasi_Final']))
        
        submitted = st.form_submit_button("Simpan Anotasi")
        
        if submitted:
            # Update DataFrame
            idx = df[(df['ID_Dokumen'] == selected_doc) & (df['ID_Paragraf'] == row['ID_Paragraf'])].index[0]
            df.at[idx, 'Sentimen'] = sentiment
            df.at[idx, 'Topik_Utama'] = topic
            df.at[idx, 'Keyword'] = keywords
            df.at[idx, 'Anotasi_Final'] = is_final
            
            # Simpan ke file
            df.to_excel(annotator_file, index=False)
            
            st.success("Anotasi berhasil disimpan!")
            
            # Download link selalu tersedia
            st.markdown(get_download_link(df, f"{annotator_name.replace(' ', '_')}.xlsx"), unsafe_allow_html=True)
    
    # Navigasi antar paragraf
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬅️ Paragraf Sebelumnya") and selected_para_idx > 0:
            # Gunakan session state jika tersedia
            if hasattr(st, 'session_state'):
                st.session_state.para_idx = selected_para_idx - 1
            st.experimental_rerun()
    
    with col2:
        if st.button("Paragraf Berikutnya ➡️") and selected_para_idx < len(para_ids) - 1:
            # Gunakan session state jika tersedia
            if hasattr(st, 'session_state'):
                st.session_state.para_idx = selected_para_idx + 1
            st.experimental_rerun()
    
    # Link download selalu tersedia di sidebar
    st.sidebar.markdown("### Download Data Anotasi")
    st.sidebar.markdown(get_download_link(df, f"{annotator_name.replace(' ', '_')}.xlsx"), unsafe_allow_html=True)
    st.sidebar.info("Selalu download hasil anotasi Anda sebelum menutup aplikasi!")

# Tampilkan halaman admin
def show_admin_page():
    """Tampilkan halaman untuk admin"""
    st.title("Admin Panel - Bank Sentral Annotation Tool")
    
    # Password sederhana untuk admin
    password = st.text_input("Admin Password", type="password")
    if password != "admin123":  # Ganti dengan password yang lebih aman
        st.warning("Masukkan password admin untuk melanjutkan")
        return
    
    st.success("Login berhasil")
    
    # Opsi admin
    st.subheader("Pembagian Tugas Anotasi")
    
    # Tampilkan pembagian dokumen untuk setiap anotator
    df = pd.read_excel(MASTER_FILE)
    unique_docs = sorted(df['ID_Dokumen'].unique())
    total_docs = len(unique_docs)
    
    # Hitung pembagian
    docs_per_annotator = total_docs // 3
    
    annotator1_docs = unique_docs[:docs_per_annotator]
    annotator2_docs = unique_docs[docs_per_annotator:2*docs_per_annotator]
    annotator3_docs = unique_docs[2*docs_per_annotator:]
    
    # Tampilkan info pembagian
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Annotator 1**")
        st.write(f"Dokumen: {annotator1_docs}")
        st.write(f"Total: {len(annotator1_docs)} dokumen")
        
        # Cek progress jika file ada
        annotator1_file = os.path.join(ANNOTATIONS_DIR, "Annotator_1.xlsx")
        if os.path.exists(annotator1_file):
            df1 = pd.read_excel(annotator1_file)
            completed1 = df1['Anotasi_Final'].sum()
            total1 = len(df1)
            st.progress(completed1/total1 if total1 > 0 else 0)
            st.write(f"Progress: {completed1}/{total1} paragraf")
    
    with col2:
        st.markdown("**Annotator 2**")
        st.write(f"Dokumen: {annotator2_docs}")
        st.write(f"Total: {len(annotator2_docs)} dokumen")
        
        # Cek progress jika file ada
        annotator2_file = os.path.join(ANNOTATIONS_DIR, "Annotator_2.xlsx")
        if os.path.exists(annotator2_file):
            df2 = pd.read_excel(annotator2_file)
            completed2 = df2['Anotasi_Final'].sum()
            total2 = len(df2)
            st.progress(completed2/total2 if total2 > 0 else 0)
            st.write(f"Progress: {completed2}/{total2} paragraf")
    
    with col3:
        st.markdown("**Annotator 3**")
        st.write(f"Dokumen: {annotator3_docs}")
        st.write(f"Total: {len(annotator3_docs)} dokumen")
        
        # Cek progress jika file ada
        annotator3_file = os.path.join(ANNOTATIONS_DIR, "Annotator_3.xlsx")
        if os.path.exists(annotator3_file):
            df3 = pd.read_excel(annotator3_file)
            completed3 = df3['Anotasi_Final'].sum()
            total3 = len(df3)
            st.progress(completed3/total3 if total3 > 0 else 0)
            st.write(f"Progress: {completed3}/{total3} paragraf")
    
    # Opsi untuk menggabungkan hasil anotasi
    st.subheader("Gabungkan Hasil Anotasi")
    
    if st.button("Gabungkan Semua Hasil Anotasi"):
        all_dfs = []
        
        # Cek dan gabungkan file dari setiap anotator
        for i in range(1, 4):
            file_path = os.path.join(ANNOTATIONS_DIR, f"Annotator_{i}.xlsx")
            if os.path.exists(file_path):
                df_annotator = pd.read_excel(file_path)
                all_dfs.append(df_annotator)
        
        if all_dfs:
            # Gabungkan semua DataFrame
            merged_df = pd.concat(all_dfs)
            
            # Simpan hasil gabungan
            merged_file = os.path.join(ANNOTATIONS_DIR, "merged_annotations.xlsx")
            merged_df.to_excel(merged_file, index=False)
            
            st.success(f"Berhasil menggabungkan {len(all_dfs)} file anotasi!")
            
            # Link download
            st.markdown(get_download_link(merged_df, "merged_annotations.xlsx"), unsafe_allow_html=True)
        else:
            st.warning("Tidak ada file anotasi yang ditemukan")

# Fungsi utama aplikasi
def main():
    st.set_page_config(
        page_title="Bank Sentral Annotation Tool",
        page_icon="📊",
        layout="wide"
    )
    
    # Sidebar untuk memilih user
    st.sidebar.title("Bank Sentral Annotation Tool")
    user_type = st.sidebar.selectbox(
        "Pilih User:",
        ["Annotator 1", "Annotator 2", "Annotator 3", "Admin"]
    )
    
    # Tampilkan halaman sesuai tipe user
    if user_type == "Admin":
        show_admin_page()
    else:
        show_annotation_page(user_type)

if __name__ == "__main__":
    main()