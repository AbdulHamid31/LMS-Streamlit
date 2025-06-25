import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np

# Konfigurasi halaman
st.set_page_config(page_title="LMS Mahasiswa", layout="wide")

# Load dataset mahasiswa
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("dataset_mahasiswa_812.csv")
        
        # Konversi status akademik ke numerik
        ipk_mapping = {
            'IPK < 2.5': 0,
            'IPK 2.5–3.0': 1,
            'IPK > 3.0': 2
        }
        df['status_akademik_numerik'] = df['status_akademik_terakhir'].map(ipk_mapping)
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

df = load_data()

# 🔐 Login dari CSV
st.sidebar.header("🔐 Login Mahasiswa")
nama_list = df["Nama"].unique().tolist()
nama = st.sidebar.selectbox("Pilih Nama Mahasiswa", nama_list)
nim_input = st.sidebar.text_input("Masukkan NIM Mahasiswa")

# Validasi Nama & NIM
valid_mahasiswa = df[df["Nama"] == nama]
if not valid_mahasiswa.empty:
    nim_terdaftar = str(valid_mahasiswa.iloc[0]["ID Mahasiswa"])
    login_berhasil = (nim_input == nim_terdaftar)
else:
    login_berhasil = False

# ✅ Jika login berhasil
if login_berhasil:
    st.title(f"🎓 LMS Mahasiswa - {nama}")
    menu = st.sidebar.radio("Navigasi", ["Beranda", "Materi", "Tugas", "Prediksi Dropout"])

    if menu == "Beranda":
        st.subheader(f"👋 Selamat Datang, {nama}!")
        col1, col2, col3 = st.columns(3)
        col1.metric("Status Login", "Aktif")
        
        # Ambil data mahasiswa yang login
        mhs_data = valid_mahasiswa.iloc[0]
        
        # Tampilkan IPK sesuai mapping
        ipk_status = mhs_data["status_akademik_terakhir"]
        col2.metric("Status Akademik", ipk_status)
        
        # Hitung progress berdasarkan materi selesai
        progress = mhs_data["materi_selesai"] / 100
        col3.metric("Materi Selesai", f"{mhs_data['materi_selesai']}%")
        st.progress(progress)

    elif menu == "Materi":
        st.subheader("📘 Materi Pembelajaran")
        with st.expander("Modul 1 - Dasar Pemrograman"):
            st.markdown("📄 Pengantar Python dan Algoritma")
        with st.expander("Modul 2 - Analisis Data"):
            st.markdown("🧠 Statistika Dasar dan Visualisasi Data")
        with st.expander("Modul 3 - Machine Learning"):
            st.markdown("📊 Model Prediktif dan Evaluasi")

    elif menu == "Tugas":
        st.subheader("📝 Daftar Tugas")
        tugas_data = pd.DataFrame({
            "Judul": ["Tugas 1 - Analisis Eksplorasi", "Tugas 2 - Model Prediksi", "Tugas 3 - Presentasi"],
            "Status": ["✅ Selesai", "❌ Belum", "❌ Belum"],
            "Deadline": ["2025-06-15", "2025-06-25", "2025-07-01"]
        })
        st.table(tugas_data)

        st.markdown("### 📎 Upload Tugas")
        uploaded = st.file_uploader("Upload file tugas (.pdf/.docx/.ipynb)", type=["pdf", "docx", "ipynb"])
        if uploaded:
            st.success(f"File '{uploaded.name}' berhasil diunggah!")

    elif menu == "Prediksi Dropout":
        st.subheader("📊 Prediksi Dropout Mahasiswa")
        
        # Data mahasiswa yang login
        mhs_data = valid_mahasiswa.iloc[0]
        
        # 1. Tampilkan profil akademik
        st.markdown("### Profil Akademik")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Login", mhs_data["total_login"])
        col2.metric("Materi Selesai", f"{mhs_data['materi_selesai']}%")
        col3.metric("Skor Kuis Rata-rata", f"{mhs_data['skor_kuis_rata2']:.2f}")
        col4.metric("Durasi Akses (jam)", f"{mhs_data['durasi_total_akses']:.1f}")
        
        # 2. Prediksi Dropout
        st.markdown("## Probabilitas Dropout")
        
        try:
            # Persiapkan data untuk modeling
            features = [
                'total_login', 'materi_selesai', 'skor_kuis_rata2', 
                'partisipasi_forum', 'durasi_total_akses', 'interaksi_mingguan',
                'jumlah_tugas_dikumpulkan', 'frekuensi_kuis', 'aktivitas_mobile',
                'status_akademik_numerik'
            ]
            
            X = df[features]
            y = df['dropout']
            
            # Bagi data dan train model
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=42,
                class_weight='balanced'
            )
            model.fit(X_train, y_train)
            
            # Prediksi untuk mahasiswa ini
            input_data = mhs_data[features].values.reshape(1, -1)
            proba = model.predict_proba(input_data)[0][1] * 100  # Probabilitas dropout
            
            # Tampilkan hasil prediksi
            st.markdown(f"<h1 style='font-size: 48px;'>{proba:.2f}%</h1>", unsafe_allow_html=True)
            
            if proba < 15:
                st.success("✅ Risiko rendah - Kemungkinan kecil untuk dropout")
            elif proba < 40:
                st.warning("⚠ Risiko sedang - Perlu perhatian")
            else:
                st.error("❌ Risiko tinggi - Perlu intervensi")
            
            # 3. SHAP Explanation
            st.subheader("Penjelasan Prediksi (SHAP)")
            
            # Hitung SHAP values
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(input_data)
            
            # Waterfall plot untuk penjelasan individual
            st.markdown("### Kontribusi Fitur untuk Prediksi Ini")
            fig, ax = plt.subplots()
            shap.plots._waterfall.waterfall_legacy(
                explainer.expected_value[1], 
                shap_values[1][0], 
                feature_names=features,
                max_display=10
            )
            st.pyplot(fig)
            plt.clf()
            
            # Summary plot untuk semua data
            st.markdown("### Pola Umum Prediksi Dropout (Berdasarkan Seluruh Data)")
            fig, ax = plt.subplots()
            shap.summary_plot(
                shap_values, 
                X_test, 
                plot_type="bar",
                feature_names=features,
                show=False
            )
            st.pyplot(fig)
            plt.clf()
            
            # Interpretasi fitur penting
            st.markdown("### Interpretasi Fitur Penting")
            st.write("""
            - **Total Login**: Semakin sering login, risiko dropout semakin rendah
            - **Materi Selesai**: Persentase materi yang diselesaikan berpengaruh negatif terhadap dropout
            - **Skor Kuis**: Nilai rata-rata kuis yang tinggi mengurangi risiko dropout
            - **Status Akademik**: Mahasiswa dengan IPK lebih tinggi cenderung tidak dropout
            - **Durasi Akses**: Waktu akses sistem yang lebih lama berkorelasi dengan risiko dropout lebih rendah
            """)
            
        except Exception as e:
            st.error(f"Terjadi error dalam prediksi: {str(e)}")
            st.info("Pastikan data yang digunakan sudah sesuai format")

else:
    st.title("🎓 LMS Mahasiswa")
    st.warning("Nama atau NIM tidak cocok. Silakan login kembali.")
