import streamlit as st
import numpy as np
import joblib

# 1. Load Model
model = joblib.load('model_inventory.pkl')

# 2. Judul dan Deskripsi
st.title("Aplikasi Prediksi Inventory Sales")
st.write("""
Aplikasi ini memprediksi **Sales Quantity** (Jumlah Terjual) berdasarkan parameter produk.
Berguna untuk perencanaan stok gudang.
""")

st.write("---")

# 3. Input User (Sidebar agar lebih rapi)
st.sidebar.header("Input Parameter Produk")

# Input sesuai fitur yang kita latih: SalesPrice, ExciseTax, Volume
sales_price = st.sidebar.number_input('Harga Jual (Sales Price)', min_value=0.0, value=10.0, step=0.1)
excise_tax = st.sidebar.number_input('Pajak Cukai (Excise Tax)', min_value=0.0, value=0.5, step=0.01)
volume = st.sidebar.number_input('Volume Produk (ml/liter)', min_value=0.0, value=750.0, step=10.0)

# Tampilkan input user di halaman utama untuk konfirmasi
st.subheader("Data Produk:")
st.write(f"**Harga:** ${sales_price}")
st.write(f"**Pajak:** ${excise_tax}")
st.write(f"**Volume:** {volume}")

# 4. Tombol Prediksi
if st.button('Prediksi Penjualan'):
    # Bentuk array data input
    input_data = np.array([[sales_price, excise_tax, volume]])
    
    # Prediksi
    prediction = model.predict(input_data)
    
    # Tampilkan Hasil
    st.success(f"Estimasi Jumlah Terjual: **{int(prediction[0])} unit**")