> ⚠️ **Catatan Penting:** Karena ukuran model Random Forest (`letter_model_rf.pkl`) melebihi batas 100MB GitHub (sebesar 111MB), file model tersebut dapat diunduh secara terpisah melalui [Link Google Drive Ini](TARUH_LINK_GD_KAMU_DI_SINI). Pastikan Anda meletakkannya di folder yang sama sebelum menjalankan aplikasi Streamlit.
# Letter Recognition: End-to-End Machine Learning Pipeline

Proyek ini mengimplementasikan sistem pengenalan 26 huruf kapital bahasa Inggris (A-Z) berdasarkan 16 fitur statistik geometri piksel menggunakan dataset orisinal dari UCI Machine Learning Repository. Proyek ini membandingkan algoritma Machine Learning Klasik dengan Deep Learning serta di-deploy menggunakan aplikasi web interaktif.

## Hasil Evaluasi & Performa Model

Berdasarkan eksperimen dan laporan metrik (*Classification Report*), berikut adalah performa ketiga model pada data uji (*test set*):

| Model | Akurasi (%) | Precision (Avg) | Recall (Avg) | F1-Score (Avg) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest Classifier** | **96.08%** | **0.96** | **0.96** | **0.96** | **Model Terbaik** |
| **Support Vector Machine (SVM)** | 95.00% | 0.95 | 0.95 | 0.95 | Cukup Stabil |
| **Multilayer Perceptron (MLP)** | 94.90% | 0.95 | 0.95 | 0.95 | Stabil |

## Fitur Proyek

1. **Exploratory Data Analysis (EDA):** Analisis distribusi kelas target dan korelasi matriks 16 fitur geometri menggunakan `seaborn`.
2. **Modular Preprocessing:** Menggunakan `LabelEncoder` untuk target kategorikal dan `StandardScaler` untuk normalisasi fitur numerik.
3. **Model Architecture:** Pelatihan komparatif antara Random Forest, SVM, dan Deep Learning MLP (TensorFlow/Keras).
4. **Computer Vision Integration:** Aplikasi web dilengkapi modul `OpenCV` untuk mengekstrak 16 fitur geometri dari gambar *screenshot* secara otomatis sebelum di-prediksi oleh model.
5. **Interactive Deployment:** Antarmuka aplikasi web berbasis `Streamlit`.

## Panduan Instalasi & Penggunaan

### 1. Prasyarat (Prerequisites)
Pastikan Anda telah menginstal Python (versi 3.8 - 3.11 direkomendasikan). Install semua pustaka yang diperlukan dengan menjalankan perintah berikut di Terminal/CMD:

```bash
pip install numpy pandas scikit-learn tensorflow streamlit joblib matplotlib seaborn opencv-python pillow
