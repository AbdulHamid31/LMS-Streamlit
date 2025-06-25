import streamlit as st
import pandas as pd
import pickle # Ditambahkan dari app_lms_final.py
import shap # Ditambahkan dari app_lms_final.py
import matplotlib.pyplot as plt # Ditambahkan dari app_lms_final.py

# Konfigurasi halaman
st.set_page_config(page_title="LMS Mahasiswa", layout="wide")

# Fungsi untuk load dataset dan model
@st.cache_data
def load_data():
    # Diubah untuk memuat dataset_mahasiswa_812_with_nim.csv dan memetakan status_akademik_terakhir
    df = pd.read_csv("dataset_mahasiswa_812_with_nim.csv")
    df['status_akademik_terakhir'] = df['status_akademik_terakhir'].map({
        'IPK < 2.5': 0, 'IPK 2.5 - 3.0': 1, 'IPK > 3.0': 2
    })
    return df

@st.cache_resource
def load_model():
    return pickle.load(open("model_xgb.pkl", "rb"))

# Load data dan model
df = load_data()
model = load_model()

# Inisialisasi session state (dari app_lms_final.py untuk mempertahankan status login)
if "login" not in st.session_state:
    st.session_state.login = False
if "nama" not in st.session_state:
    st.session_state.nama = ""
if "nim" not in st.session_state:
    st.session_state.nim = ""

# Tampilan login
if not st.session_state.login:
    st.sidebar.header("🔐 Login Mahasiswa")
    nama = st.sidebar.selectbox("Pilih Nama Mahasiswa", df["Nama"].unique().tolist())
    nim_input = st.sidebar.text_input("Masukkan NIM Mahasiswa")

    with st.sidebar.expander("📋 Lihat Daftar NIM & Nama"): # Diubah agar lebih jelas
        st.dataframe(df[["NIM", "Nama"]]) # Menampilkan kolom NIM

    valid_mahasiswa = df[df["Nama"] == nama]
    if not valid_mahasiswa.empty:
        # Diubah untuk menggunakan kolom 'NIM' dari dataset
        nim_terdaftar = str(valid_mahasiswa.iloc[0]["NIM"])
        if nim_input == nim_terdaftar:
            st.session_state.login = True
            st.session_state.nama = nama
            st.session_state.nim = nim_input
            st.success("✅ Login berhasil!")
            st.experimental_rerun() # Untuk memuat ulang halaman setelah login
        elif nim_input:
            st.error("❌ NIM tidak cocok.")
else: # Ini adalah blok utama aplikasi setelah login berhasil (dari app_lms_final.py)
    nama = st.session_state.nama
    nim = st.session_state.nim
    st.title(f"🎓 LMS Mahasiswa - {nama}")

    if st.sidebar.button("🔓 Logout"): # Tombol logout
        st.session_state.login = False
        st.session_state.nama = ""
        st.session_state.nim = ""
        st.experimental_rerun() # Untuk memuat ulang halaman setelah logout

    menu = st.sidebar.radio("Navigasi", ["Beranda", "Materi", "Tugas", "Prediksi Dropout"])

    if menu == "Beranda":
        st.subheader("👋 Selamat Datang di Dashboard LMS!") # Diubah sedikit untuk konsistensi
        col1, col2, col3 = st.columns(3)
        col1.metric("Status Login", "Aktif")
        col2.metric("NIM", nim) # Menampilkan NIM yang sudah login
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
        # Seluruh blok ini disalin dari app_lms_final.py
        st.subheader("📊 Hasil Prediksi Dropout")
        mahasiswa = df[df["Nama"] == nama]

        if not mahasiswa.empty:
            # Memilih fitur untuk prediksi, menghapus kolom yang tidak relevan.
            # Asumsi dataset_mahasiswa_812_with_nim.csv digunakan.
            X = mahasiswa.drop(columns=["ID Mahasiswa", "Nama", "NIM", "dropout"])
            proba = model.predict_proba(X)[0][1]
            pred = model.predict(X)[0]

            st.metric("Probabilitas Dropout", f"{proba:.2%}")
            if pred == 1:
                st.error("❌ Mahasiswa ini diprediksi berisiko dropout.")
            else:
                st.success("✅ Mahasiswa ini diprediksi tidak dropout.")

            st.markdown("---")
            st.subheader("📈 Visualisasi SHAP - Faktor yang Mempengaruhi Prediksi")

            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)

            plt.figure(figsize=(10, 4))
            shap.plots.bar(shap.Explanation(
                values=shap_values,
                base_values=explainer.expected_value,
                data=X,
                feature_names=X.columns.tolist()
            ))
            st.pyplot(plt.gcf())
            plt.clf()

            st.markdown("---")
            st.subheader("📊 Distribusi Dropout Keseluruhan")
            dropout_counts = df['dropout'].value_counts()
            labels = ['Tidak Dropout', 'Dropout']
            colors = ['#28a745', '#dc3545']

            fig, ax = plt.subplots()
            ax.pie(dropout_counts, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
            ax.axis('equal')
            st.pyplot(fig)

        else:
            st.warning("Data mahasiswa tidak ditemukan.")

# Bagian else dari app_lms (4).py untuk kasus login gagal
# Namun, karena session state dari app_lms_final.py sudah menangani ini,
# blok ini kemungkinan tidak akan tercapai dalam alur yang baru.
# Saya akan tinggalkan sebagai komentar jika Anda memerlukannya.
# else:
#    st.title("🎓 LMS Mahasiswa")
#    st.warning("Nama atau NIM tidak cocok. Silakan login kembali.")
