import streamlit as st
import numpy as np
import joblib
from pathlib import Path
from tensorflow.keras.models import load_model

# 1. Konfigurasi Path Model Dataset
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "letter_model.keras"
RF_PATH = BASE_DIR / "letter_model_rf.pkl"
SVM_PATH = BASE_DIR / "letter_model_svm.pkl"
SCALER_PATH = BASE_DIR / "scaler.pkl"
LE_PATH = BASE_DIR / "label_encoder.pkl"

st.set_page_config(page_title="Letter Recognition Classifier", layout="wide")
st.title("🔤 Letter Recognition Dataset Classifier")

st.markdown("Aplikasi ini digunakan untuk menguji dan memvalidasi performa model yang telah dilatih menggunakan "
            "**Dataset Letter Recognition (16 Fitur Numerik)**. Anda dapat memasukkan nilai karakteristik "
            "geometris huruf untuk melihat perbandingan prediksi dari tiga algoritma sekaligus.")

# 2. Load Semua Model dan Tools Latihan
try:
    mlp_model = load_model(MODEL_PATH)
    rf_model = joblib.load(RF_PATH)
    svm_model = joblib.load(SVM_PATH)
    scaler = joblib.load(SCALER_PATH)
    le = joblib.load(LE_PATH)
except Exception as e:
    st.error(f"Gagal memuat model dataset. Pastikan semua file .pkl dan .keras ada di folder yang sama: {e}")
    st.stop()

# 3. Form Input 16 Fitur Sesuai Dataset
st.subheader("📊 Input 16 Fitur Karakteristik Huruf")

# Panduan arti dari setiap fitur berdasarkan dataset UCI
with st.expander("📖 Lihat Panduan Arti Fitur (Bahan Presentasi ke Dosen)"):
    st.markdown("""
    Setiap fitur menerima input nilai **0 sampai 15** yang merepresentasikan:
    * **Fitur 1 (X-Box):** Posisi horizontal rata-rata dari kotak pembungkus huruf.
    * **Fitur 2 (Y-Box):** Posisi vertikal rata-rata dari kotak pembungkus huruf.
    * **Fitur 3 (Width):** Lebar dari kotak pembungkus huruf.
    * **Fitur 4 (Height):** Tinggi dari kotak pembungkus huruf.
    * **Fitur 5 (Total Pixels):** Jumlah total piksel hitam (on-pixels) dalam kotak.
    * **Fitur 6 (X-Mean):** Nilai rata-rata posisi X dari piksel hitam di dalam kotak.
    * **Fitur 7 (Y-Mean):** Nilai rata-rata posisi Y dari piksel hitam di dalam kotak.
    * **Fitur 8 (X-Variance):** Varians posisi X (seberapa menyebar piksel secara horizontal).
    * **Fitur 9 (Y-Variance):** Varians posisi Y (seberapa menyebar piksel secara vertikal).
    * **Fitur 10 (XY-Correlation):** Korelasi spasial rata-rata antara sumbu X dan Y dari piksel.
    * **Fitur 11 (X-Ege Mean):** Rata-rata posisi X dari tepi/border huruf kiri ke kanan.
    * **Fitur 12 (X-Edge Variance):** Korelasi posisi tepi X terhadap posisi Y.
    * **Fitur 13 (Y-Edge Mean):** Rata-rata posisi Y dari tepi/border huruf bawah ke atas.
    * **Fitur 14 (Y-Edge Variance):** Korelasi posisi tepi Y terhadap posisi X.
    * **Fitur 15 (X-Variance-Y):** Varians spasial gabungan dengan bobot sumbu X.
    * **Fitur 16 (Y-Variance-X):** Varians spasial gabungan dengan bobot sumbu Y.
    """)

# Nama fitur asli untuk label input field agar lebih ilmiah
feature_labels = [
    "1. X-Box Position", "2. Y-Box Position", "3. Box Width", "4. Box Height",
    "5. Total On-Pixels", "6. X-Mean Pixel", "7. Y-Mean Pixel", "8. X-Variance",
    "9. Y-Variance", "10. XY-Correlation", "11. X-Edge Mean", "12. X-Edge Variance",
    "13. Y-Edge Mean", "14. Y-Edge Variance", "15. X-Var-Y Weight", "16. Y-Var-X Weight"
]

with st.form("dataset_form"):
    cols = st.columns(4)
    feature_values = []
    
    # Looping otomatis membuat 16 input field dengan label ilmiah
    for idx in range(16):
        with cols[idx % 4]:
            value = st.number_input(
                feature_labels[idx], # Menggunakan label ilmiah, bukan cuma "Fitur X"
                min_value=0, 
                max_value=15, 
                value=0, 
                step=1, 
                key=f"feat_{idx}"
            )
            feature_values.append(value)
            
    submit = st.form_submit_button("Lakukan Prediksi Huruf")
    
# 4. Eksekusi Prediksi Multi-Model
if submit:
    # Preprocessing input agar sesuai format scaler saat training
    input_array = np.array(feature_values).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    
    # A. Prediksi Model MLP (Deep Learning)
    mlp_pred = mlp_model.predict(input_scaled, verbose=0)
    mlp_letter = le.inverse_transform([np.argmax(mlp_pred)])[0]
    mlp_conf = float(np.max(mlp_pred)) * 100
    
    # B. Prediksi Model Random Forest
    rf_pred = rf_model.predict(input_scaled)
    rf_letter = le.inverse_transform(rf_pred)[0]
    
    # C. Prediksi Model SVM
    svm_pred = svm_model.predict(input_scaled)
    svm_letter = le.inverse_transform(svm_pred)[0]
    
    # 5. Tampilkan Hasil Komparasi ke Dashboard
    st.write("---")
    st.subheader("🔮 Hasil Perbandingan Prediksi Huruf:")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("MLP Neural Network", mlp_letter, f"Confidence {mlp_conf:.2f}%")
    with col2:
        st.metric("Random Forest", rf_letter)
    with col3:
        st.metric("SVM Classifier", svm_letter)
        
    st.info(f"**Data Input Terproses:** {feature_values}")