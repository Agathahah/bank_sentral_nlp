import pandas as pd

# Muat hasil anotasi
annotated_data = pd.read_excel("results/annotated_data.xlsx")

# Cek kelengkapan
null_count = annotated_data['Sentiment'].isnull().sum()
print(f"Jumlah data tanpa anotasi: {null_count} dari {len(annotated_data)}")

# Cek distribusi sentimen
sentiment_counts = annotated_data['Sentiment'].value_counts()
print("\nDistribusi sentimen:")
print(sentiment_counts)

# Cek nilai unik (pastikan tidak ada typo)
unique_sentiments = annotated_data['Sentiment'].unique()
print("\nNilai unik sentimen:")
print(unique_sentiments)

# Validasi nilai
valid_sentiments = ['Positif', 'Netral', 'Negatif']
invalid_sentiments = [s for s in unique_sentiments if s not in valid_sentiments and not pd.isna(s)]
if invalid_sentiments:
    print(f"\nPeringatan: Ditemukan nilai sentimen tidak valid: {invalid_sentiments}")
    print("Hanya 'Positif', 'Netral', dan 'Negatif' yang diperbolehkan")
else:
    print("\nSemua nilai sentimen valid")