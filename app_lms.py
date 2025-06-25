import streamlit as st
import pandas as pd

st.set_page_config(page_title="LMS Mahasiswa", layout="wide")

# Load dataset mahasiswa
@st.cache_data
def load_data():
    df = pd.read_csv("dataset_mahasiswa_812.csv")
    return df

df = load_data()

# 🔐 Login dari CSV
st.sidebar.header("🔐 Login Mahasiswa")
nama_list = df["Nama"].unique().tolist()
nama = st.sidebar.selectbox("Pilih Nama Mahasiswa", nama_list)
nim_input = st.sidebar.text_input("Masukkan NIM Mahasiswa")

# Validasi Nama & NIM
valid_mahasiswa = df[df["Nama"] == nama]
if not valid_mahasiswa.empty:
    nim_terdaftar = str(valid_mahasiswa.iloc[0]["ID Mahasiswa"])  # dianggap NIM
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
        st.subheader("📊 Prediksi Dropout Mahasiswa (Simulasi)")

        import shap
        import pickle
        import matplotlib.pyplot as plt
    
        # Load model
        model = pickle.load(open("model_xgb.pkl", "rb"))
    
        # Ambil data fitur mahasiswa yang login
        data_mahasiswa = df[df["Nama"] == nama]
        fitur_model = data_mahasiswa.drop(columns=["Nama", "ID Mahasiswa"])
        X = fitur_model
    
        # Probabilitas prediksi dari model
        probas = model.predict_proba(X)[0][1] * 100
        st.markdown("## Probabilitas Dropout")
        st.markdown(f"<h1 style='font-size: 48px;'>{probas:.2f}%</h1>", unsafe_allow_html=True)
    
        if probas < 30:
            st.success("✅ Mahasiswa ini sangat kecil kemungkinannya untuk dropout.")
        else:
            st.error("⚠️ Mahasiswa ini berisiko tinggi untuk dropout.")
    

else:
    st.title("🎓 LMS Mahasiswa")
    st.warning("Nama atau NIM tidak cocok. Silakan login kembali.")
