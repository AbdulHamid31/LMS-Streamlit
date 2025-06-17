
import streamlit as st
import pandas as pd

st.set_page_config(page_title="LMS Mahasiswa", layout="wide")
# 🔐 Simulasi Login Mahasiswa (TAMBAHKAN INI)
st.sidebar.header("🔐 Login Mahasiswa")
nama = st.sidebar.text_input("Masukkan Nama Mahasiswa")
nim = st.sidebar.text_input("Masukkan NIM Mahasiswa")
st.title("🎓 Learning Management System Mahasiswa")

# Sidebar menu navigasi
st.sidebar.header("🔍 Navigasi")
# 🔐 Login Mahasiswa
st.sidebar.header("🔐 Login Mahasiswa")
nama = st.sidebar.text_input("Masukkan Nama Mahasiswa")
nim = st.sidebar.text_input("Masukkan NIM Mahasiswa")

menu = st.sidebar.radio("Pilih Halaman", ["Beranda", "Materi", "Tugas", "Prediksi Dropout"])

# Halaman Beranda
if menu == "Beranda":
    st.subheader("👋 Selamat Datang di Dashboard LMS!")
    st.write("Ini adalah halaman utama LMS mahasiswa untuk melihat status akademik dan aktivitas.")
    col1, col2, col3 = st.columns(3)
    col1.metric("Status Login", "Aktif")
    col2.metric("IPK Terakhir", "3.25")
    col3.metric("Kemajuan Kelas", "70%")
    st.progress(0.7)

# Halaman Materi
elif menu == "Materi":
    st.subheader("📘 Daftar Materi Pembelajaran")
    with st.expander("Modul 1 - Pengantar Data"):
        st.markdown("📄 Materi ini membahas tentang data, sumber, dan bentuknya.")
    with st.expander("Modul 2 - Machine Learning Dasar"):
        st.markdown("🧠 Pembahasan supervised learning, unsupervised, dan algoritma umum.")
    with st.expander("Modul 3 - Evaluasi Model"):
        st.markdown("📊 Precision, Recall, F1-score, dan penggunaan confusion matrix.")

# Halaman Tugas
elif menu == "Tugas":
    st.subheader("📝 Daftar Tugas")

    # ➤ Daftar tugas mahasiswa
    tugas_data = pd.DataFrame({
        "Judul": ["Tugas 1", "Tugas 2", "Tugas 3"],
        "Status": ["✅ Selesai", "❌ Belum", "❌ Belum"],
        "Deadline": ["2025-06-15", "2025-06-25", "2025-07-01"]
    })
    st.table(tugas_data)

    # ➤ Upload tugas (HANYA muncul di halaman Tugas)
    st.markdown("### 📎 Upload Tugas")
    uploaded = st.file_uploader("Upload file tugas (.pdf/.docx)", type=["pdf", "docx"])
    if uploaded:
        st.success(f"File '{uploaded.name}' berhasil diunggah!")



# Halaman Prediksi Dropout
elif menu == "Prediksi Dropout":
    st.subheader("📊 Prediksi Dropout Mahasiswa")
    st.write("Prediksi ini berdasarkan aktivitas mahasiswa dalam LMS.")
    st.success("✅ Mahasiswa ini tidak berisiko dropout berdasarkan data saat ini.")
    st.metric("Probabilitas Dropout", "8.42%")
    st.markdown("Fitur yang mempengaruhi prediksi:")
    st.markdown("- Total Login: 43")
    st.markdown("- Materi Selesai: 91")
    st.markdown("- IPK: < 2.5")
    st.markdown("- Durasi Akses: 58.6 jam")
