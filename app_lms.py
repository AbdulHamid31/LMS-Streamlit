import streamlit as st
import pandas as pd

st.set_page_config(page_title="LMS Mahasiswa", layout="wide")

# 🔐 Simulasi Login Mahasiswa
st.sidebar.header("🔐 Login Mahasiswa")
nama = st.sidebar.text_input("Masukkan Nama Mahasiswa")
nim = st.sidebar.text_input("Masukkan NIM Mahasiswa")

# Tampilkan seluruh isi aplikasi jika sudah login
if nama and nim:
    st.title(f"🎓 LMS Mahasiswa - {nama}")

    # Menu Navigasi LMS
    menu = st.sidebar.radio("Navigasi", ["Beranda", "Materi", "Tugas", "Prediksi Dropout"])

    # Halaman Beranda
    if menu == "Beranda":
        st.subheader(f"👋 Selamat Datang, {nama}!")
        col1, col2, col3 = st.columns(3)
        col1.metric("Status Login", "Aktif")
        col2.metric("IPK Terakhir", "3.25")
        col3.metric("Kemajuan Kelas", "70%")
        st.progress(0.7)

    # Halaman Materi
    elif menu == "Materi":
        st.subheader("📘 Materi Pembelajaran")
        with st.expander("Modul 1 - Pengantar Data"):
            st.markdown("📄 Materi ini membahas tentang pengantar data, bentuk, dan struktur.")
        with st.expander("Modul 2 - Machine Learning"):
            st.markdown("🧠 Pembahasan supervised dan unsupervised learning.")
        with st.expander("Modul 3 - Evaluasi Model"):
            st.markdown("📊 Precision, Recall, F1-score, dan confusion matrix.")

    # Halaman Tugas
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

    # Halaman Prediksi Dropout (simulasi)
    elif menu == "Prediksi Dropout":
        st.subheader("📊 Prediksi Dropout Mahasiswa (Simulasi)")
        st.metric("Probabilitas Dropout", "8.42%")
        st.success("✅ Mahasiswa ini tidak berisiko dropout.")
        st.markdown("Fitur-fitur yang memengaruhi prediksi:")
        st.markdown("- Total Login: 43")
        st.markdown("- Materi Selesai: 91")
        st.markdown("- IPK: < 2.5")
        st.markdown("- Durasi Akses: 58.6 jam")

else:
    st.title("🎓 LMS Mahasiswa")
    st.warning("Silakan masukkan nama dan NIM mahasiswa untuk mengakses konten LMS.")
