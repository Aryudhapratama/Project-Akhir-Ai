import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# Fungsi untuk memuat data dan melatih model
# @st.cache_data digunakan agar proses ini tidak diulang setiap kali user berinteraksi
@st.cache_data
def load_and_train_model():
    # 1. Memuat Data
    try:
        data = pd.read_csv('heart.csv')
    except FileNotFoundError:
        st.error("Error: File 'heart.csv' tidak ditemukan. Pastikan file berada di folder yang sama.")
        return None, None, None

    # 2. Persiapan Data
    X = data.drop('target', axis=1)
    y = data['target']
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 3. Normalisasi Fitur
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # 4. Melatih Model k-NN
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_scaled, y_train)
    
    return knn, scaler, data

# --- Tampilan Aplikasi Streamlit ---

# Memuat model, scaler, dan data
model, scaler, data = load_and_train_model()

# Judul Aplikasi
st.title('Aplikasi Prediksi Risiko Penyakit Jantung')
st.write('Aplikasi ini menggunakan algoritma k-NN untuk memprediksi risiko penyakit jantung berdasarkan data medis.')

# --- Input dari User di Sidebar ---
st.sidebar.header('Input Data Pasien:')

def user_input_features():
    # Membuat slider dan selectbox untuk setiap fitur
    age = st.sidebar.slider('Usia', 20, 80, 50)
    sex = st.sidebar.selectbox('Jenis Kelamin', ('Pria', 'Wanita'))
    cp = st.sidebar.selectbox('Tipe Nyeri Dada (cp)', (0, 1, 2, 3))
    trestbps = st.sidebar.slider('Tekanan Darah (trestbps)', 90, 200, 120)
    chol = st.sidebar.slider('Kolesterol (chol)', 120, 570, 240)
    fbs = st.sidebar.selectbox('Gula Darah Puasa > 120 mg/dl (fbs)', ('Tidak', 'Ya'))
    restecg = st.sidebar.selectbox('Hasil Elektrokardiografi (restecg)', (0, 1, 2))
    thalach = st.sidebar.slider('Detak Jantung Maksimum (thalach)', 70, 210, 150)
    exang = st.sidebar.selectbox('Nyeri Dada saat Olahraga (exang)', ('Tidak', 'Ya'))
    oldpeak = st.sidebar.slider('Oldpeak', 0.0, 6.2, 1.0)
    slope = st.sidebar.selectbox('Slope', (0, 1, 2))
    ca = st.sidebar.selectbox('Jumlah Pembuluh Darah Utama (ca)', (0, 1, 2, 3, 4))
    thal = st.sidebar.selectbox('Thal', (0, 1, 2, 3))

    # Konversi input menjadi format yang sesuai untuk model
    sex_val = 1 if sex == 'Pria' else 0
    fbs_val = 1 if fbs == 'Ya' else 0
    exang_val = 1 if exang == 'Ya' else 0

    # Membuat dictionary dari input
    input_data = {
        'age': age, 'sex': sex_val, 'cp': cp, 'trestbps': trestbps, 'chol': chol, 
        'fbs': fbs_val, 'restecg': restecg, 'thalach': thalach, 'exang': exang_val, 
        'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
    }
    
    # Mengubah dictionary menjadi dataframe
    features = pd.DataFrame(input_data, index=[0])
    return features

# Mengambil input dari user
input_df = user_input_features()

# Menampilkan data input dari user
st.subheader('Data Pasien yang Diinput:')
st.write(input_df)

# Tombol untuk melakukan prediksi
if st.button('Lakukan Prediksi'):
    if model is not None and scaler is not None:
        # Normalisasi input user menggunakan scaler yang sudah dilatih
        input_scaled = scaler.transform(input_df)
        
        # Lakukan prediksi
        prediction = model.predict(input_scaled)
        prediction_proba = model.predict_proba(input_scaled)

        st.subheader('Hasil Prediksi:')
        if prediction[0] == 1:
            st.error('**Risiko Tinggi Terkena Penyakit Jantung**')
        else:
            st.success('**Risiko Rendah Terkena Penyakit Jantung**')
        
        st.write(f"Tingkat kepercayaan model (probabilitas):")
        st.write(f"- Risiko Rendah: {prediction_proba[0][0]*100:.2f}%")
        st.write(f"- Risiko Tinggi: {prediction_proba[0][1]*100:.2f}%")

# Menampilkan informasi tambahan
st.sidebar.markdown("---")
st.sidebar.info("Pastikan file `heart.csv` berada di direktori yang sama dengan aplikasi ini.")

if data is not None:
    st.markdown("---")
    st.subheader('Tampilan Data Awal (5 Baris Pertama)')
    st.write(data.head())