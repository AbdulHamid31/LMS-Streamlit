import streamlit as st
import pandas as pd
import pickle
import xgboost as xgb
import shap
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(page_title="LMS Mahasiswa", layout="wide")

# Disable PyplotGlobalUse warning


# Load student dataset
@st.cache_data
def load_data():
    df = pd.read_csv("dataset_mahasiswa_812.csv")
    return df

# Load XGBoost model
@st.cache_resource
def load_model():
    try:
        # Try loading as XGBoost model
        model = xgb.Booster()
        model.load_model('model_xgb.pkl')
        return model
    except:
        try:
            # Try loading as pickle file
            with open('model_xgb.pkl', 'rb') as file:
                model = pickle.load(file)
                return model
        except Exception as e:
            st.error(f"Failed to load model: {str(e)}")
            return None

df = load_data()
model = load_model()

# 🔐 Student Login
st.sidebar.header("🔐 Login Mahasiswa")
nama_list = df["Nama"].unique().tolist()
nama = st.sidebar.selectbox("Pilih Nama Mahasiswa", nama_list)
nim_input = st.sidebar.text_input("Masukkan NIM Mahasiswa")

# Validate credentials
valid_mahasiswa = df[df["Nama"] == nama]
if not valid_mahasiswa.empty:
    nim_terdaftar = str(valid_mahasiswa.iloc[0]["ID Mahasiswa"])
    login_berhasil = (nim_input == nim_terdaftar)
else:
    login_berhasil = False

# ✅ If login successful
if login_berhasil:
    st.title(f"🎓 LMS Mahasiswa - {nama}")
    menu = st.sidebar.radio("Navigasi", ["Beranda", "Materi", "Tugas", "Prediksi Dropout"])

    if menu == "Beranda":
        st.subheader(f"👋 Selamat Datang, {nama}!")
        col1, col2, col3 = st.columns(3)
        col1.metric("Status Login", "Aktif")
        col2.metric("IPK Terakhir", "3.25")
        col3.metric("Kemajuan Kelas", "70%")
        st.progress(0.7)

    elif menu == "Materi":
        st.subheader("📘 Materi Pembelajaran")
        with st.expander("Modul 1"):
            st.markdown("📄 Pengantar Data")
        with st.expander("Modul 2"):
            st.markdown("🧠 Machine Learning Dasar")
        with st.expander("Modul 3"):
            st.markdown("📊 Evaluasi Model")

    elif menu == "Tugas":
        st.subheader("📝 Daftar Tugas")
        tugas_data = pd.DataFrame({
            "Judul": ["Tugas 1", "Tugas 2", "Tugas 3"],
            "Status": ["✅ Selesai", "❌ Belum", "❌ Belum"],
            "Deadline": ["2025-06-15", "2025-06-25", "2025-07-01"]
        })
        st.table(tugas_data)

        st.markdown("### 📎 Upload Tugas")
        uploaded = st.file_uploader("Upload file tugas (.pdf/.docx)", type=["pdf", "docx"])
        if uploaded:
            st.success(f"File '{uploaded.name}' berhasil diunggah!")

    elif menu == "Prediksi Dropout":
        st.subheader("📊 Prediksi Dropout Mahasiswa")
        
        if model is None:
            st.error("Model tidak tersedia. Silakan hubungi administrator.")
            st.stop()
        
        try:
            # Get student data
            mahasiswa_data = df[df["Nama"] == nama].iloc[0]
            
            # Prepare input features (adjust based on your model's features)
            input_data = pd.DataFrame({
                'total_login': [mahasiswa_data['total_login']],
                'materi_selesai': [mahasiswa_data['materi_selesai']],
                'skor_kuis_rata2': [mahasiswa_data['skor_kuis_rata2']],
                'partisipasi_forum': [mahasiswa_data['partisipasi_forum']],
                'durasi_total_akses': [mahasiswa_data['durasi_total_akses']],
                'status_akademik_terakhir': [mahasiswa_data['status_akademik_terakhir']],
                'interaksi_mingguan': [mahasiswa_data['interaksi_mingguan']],
                'jumlah_tugas_dikumpulkan': [mahasiswa_data['jumlah_tugas_dikumpulkan']],
                'frekuensi_kuis': [mahasiswa_data['frekuensi_kuis']],
                'aktivitas_mobile': [mahasiswa_data['aktivitas_mobile']]
            })
            
            # Make prediction
            if hasattr(model, 'predict_proba'):
                # For scikit-learn style model
                proba = model.predict_proba(input_data)[0][1] * 100
            elif hasattr(model, 'predict'):
                # For XGBoost booster
                dmatrix = xgb.DMatrix(input_data)
                proba = model.predict(dmatrix)[0] * 100
            else:
                st.error("Model tidak valid")
                st.stop()
            
            # Display results
            st.markdown(f"## Probabilitas Dropout: {proba:.2f}%")
            
            if proba < 30:
                st.success("✅ Risiko rendah - Mahasiswa aktif dan berprestasi")
            elif proba < 60:
                st.warning("⚠️ Risiko sedang - Perlu perhatian lebih")
            else:
                st.error("❌ Risiko tinggi - Segera lakukan intervensi")
            
            # Feature importance
            st.subheader("Faktor yang Mempengaruhi Prediksi")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Login", mahasiswa_data['total_login'])
                st.metric("Materi Diselesaikan", mahasiswa_data['materi_selesai'])
                st.metric("Skor Kuis Rata-rata", f"{mahasiswa_data['skor_kuis_rata2']:.2f}")
            
            with col2:
                st.metric("Durasi Akses", f"{mahasiswa_data['durasi_total_akses']:.1f} jam")
                st.metric("Partisipasi Forum", mahasiswa_data['partisipasi_forum'])
                st.metric("Tugas Dikumpulkan", mahasiswa_data['jumlah_tugas_dikumpulkan'])
            
            # SHAP Explanation (if available)
            try:
                st.subheader("Penjelasan Prediksi (SHAP)")
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(input_data)
                
                fig, ax = plt.subplots()
                shap.summary_plot(shap_values, input_data, show=False)
                st.pyplot(fig)
                plt.close()
            except Exception as e:
                st.warning(f"Tidak dapat menampilkan penjelasan SHAP: {str(e)}")
                
        except Exception as e:
            st.error(f"Terjadi kesalahan saat prediksi: {str(e)}")

else:
    st.title("🎓 LMS Mahasiswa")
    st.warning("Nama atau NIM tidak cocok. Silakan login kembali.")
