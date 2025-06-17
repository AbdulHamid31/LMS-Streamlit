
import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="LMS Mahasiswa", layout="wide")

# Load dataset mahasiswa dan model prediksi
@st.cache_data
def load_data():
    df = pd.read_csv("dataset_mahasiswa_812_with_nim.csv")
    df['status_akademik_terakhir'] = df['status_akademik_terakhir'].map({
        'IPK < 2.5': 0, 'IPK 2.5 - 3.0': 1, 'IPK > 3.0': 2
    })
    return df

@st.cache_resource
def load_model():
    return pickle.load(open("model_xgb.pkl", "rb"))

df = load_data()
model = load_model()

# Inisialisasi session state
if "login" not in st.session_state:
    st.session_state.login = False
if "nama" not in st.session_state:
    st.session_state.nama = ""
if "nim" not in st.session_state:
    st.session_state.nim = ""

# Halaman login
if not st.session_state.login:
    st.sidebar.header("🔐 Login Mahasiswa")
    nama = st.sidebar.selectbox("Pilih Nama Mahasiswa", df["Nama"].unique().tolist())
    nim_input = st.sidebar.text_input("Masukkan NIM Mahasiswa")

    with st.sidebar.expander("📋 Lihat Daftar NIM & Nama"):
        st.dataframe(df[["NIM", "Nama"]])

    valid_mahasiswa = df[df["Nama"] == nama]
    if not valid_mahasiswa.empty:
        nim_terdaftar = str(valid_mahasiswa.iloc[0]["NIM"])
        if nim_input == nim_terdaftar:
            st.session_state.login = True
            st.session_state.nama = nama
            st.session_state.nim = nim_input
            st.success("✅ Login berhasil!")
            st.experimental_rerun()
        elif nim_input:
            st.error("❌ NIM tidak cocok.")
else:
    nama = st.session_state.nama
    nim = st.session_state.nim
    st.title(f"🎓 LMS Mahasiswa - {nama}")

    # Tombol logout
    if st.sidebar.button("🔓 Logout"):
        st.session_state.login = False
        st.session_state.nama = ""
        st.session_state.nim = ""
        st.experimental_rerun()

    # Navigasi
    menu = st.sidebar.radio("Navigasi", ["Beranda", "Materi", "Tugas"])

    # 🔍 Prediksi dropout otomatis muncul di atas
    st.subheader("📊 Hasil Prediksi Dropout")
    mahasiswa = df[df["Nama"] == nama]
    if not mahasiswa.empty:
        X = mahasiswa.drop(columns=["ID Mahasiswa", "Nama", "NIM", "dropout"])
        proba = model.predict_proba(X)[0][1]
        pred = model.predict(X)[0]

        st.metric("Probabilitas Dropout", f"{proba:.2%}")
        if pred == 1:
            st.error("❌ Mahasiswa ini diprediksi berisiko dropout.")
        else:
            st.success("✅ Mahasiswa ini diprediksi tidak dropout.")
    else:
        st.warning("Data mahasiswa tidak ditemukan.")

    if menu == "Beranda":
        st.subheader("👋 Selamat Datang di Dashboard LMS!")
        col1, col2, col3 = st.columns(3)
        col1.metric("Status Login", "Aktif")
        col2.metric("NIM", nim)
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
