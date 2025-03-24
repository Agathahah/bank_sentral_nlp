import pandas as pd

# Baca file Excel
try:
    df = pd.read_excel('data/Press_Release.xlsx')
    print("Berhasil membaca file Excel")
    
    # Simpan sebagai CSV
    df.to_csv('data/Press_Release.csv', index=False, encoding='utf-8')
    print("Berhasil menyimpan ke CSV")
except Exception as e:
    print(f"Error: {e}")
