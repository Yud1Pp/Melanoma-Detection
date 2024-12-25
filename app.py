import streamlit as st
from streamlit_option_menu import option_menu
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import joblib
import io

scaler = joblib.load('model/scaler.joblib')
label_encoder = joblib.load('model/label_encoder.joblib')
model = joblib.load('model/melanoma_model.joblib')

# Fungsi untuk segmentasi
def segment_image(original_image):
    # Mengubah gambar menjadi grayscale
    gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    # Melakukan thresholding menggunakan metode Otsu
    _, segmented = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return segmented

# Edge
def detect_edges(segmented_image):
    # Menggunakan segmented image (hanya lesi)
    edges = cv2.Canny(segmented_image, 50, 150)
    return edges

# Fungsi untuk menghitung histogram
def extract_histogram_mean(original_image):
    # Mengubah gambar menjadi grayscale
    gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    
    # Menghitung histogram intensitas piksel dengan 256 bins
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    
    # Normalisasi histogram
    hist = cv2.normalize(hist, hist).flatten()
    
    # Menghitung rata-rata intensitas dari histogram
    mean_intensity = np.sum(hist * np.arange(256))  # Mengalikan intensitas dengan frekuensinya
    return hist, mean_intensity

def calculate_compactness(segmented_image):
    contours, _ = cv2.findContours(segmented_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        contour = max(contours, key=cv2.contourArea)  # Kontur terbesar (lesi)
        perimeter = cv2.arcLength(contour, True)
        area = cv2.contourArea(contour)
        compactness = (perimeter ** 2) / (4 * np.pi * area)
        return compactness
    return 0

def calculate_asymmetry(segmented_image):
    # Ambil bounding box atau area dari objek lesi (segmented)
    contours, _ = cv2.findContours(segmented_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        # Ambil kontur lesi terbesar
        lesion = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(lesion)
        # Potong bagian lesi dari gambar
        cropped_lesion = segmented_image[y:y+h, x:x+w]
        # Membagi lesi menjadi dua bagian, pastikan width genap
        width = cropped_lesion.shape[1]
        half_width = width // 2
        
        # Handle odd width by adjusting the split
        left = cropped_lesion[:, :half_width]
        right = cropped_lesion[:, width - half_width:] # Adjust the right split

        
        right_flipped = cv2.flip(right, 1)
        
        # Resize to match if sizes are still different after adjustment
        if left.shape != right_flipped.shape:
            right_flipped = cv2.resize(right_flipped, (left.shape[1], left.shape[0]))
        
        # Hitung perbedaan antara kiri dan kanan
        difference = cv2.absdiff(left, right_flipped)
        asymmetry_score = np.sum(difference) / np.sum(cropped_lesion)
        return asymmetry_score
    return 0

def extract_color_features(original_image):
    # Mengubah gambar ke ruang warna HSV
    hsv = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)
    # Menghitung rata-rata Hue, Saturation, dan Value
    mean_hue = np.mean(hsv[:, :, 0])
    mean_saturation = np.mean(hsv[:, :, 1])
    mean_value = np.mean(hsv[:, :, 2])
    return mean_hue, mean_saturation, mean_value

def calculate_edge_irregularity(segmented_image):
    edges = cv2.Canny(segmented_image, 50, 150)
    edge_count = np.count_nonzero(edges)
    total_pixels = segmented_image.shape[0] * segmented_image.shape[1]
    irregularity = edge_count / total_pixels
    return irregularity

def extract_features_from_image(original_image):
    # Segmentasi gambar
    segmented = segment_image(original_image)
    
    # Ekstraksi fitur
    asymmetry = calculate_asymmetry(segmented)
    compactness = calculate_compactness(segmented)
    hist, mean_intensity = extract_histogram_mean(original_image)
    mean_hue, mean_saturation, mean_value = extract_color_features(original_image)
    irregularity = calculate_edge_irregularity(segmented)

    # Gabungkan hasil fitur
    features = {
        "asymmetry": asymmetry,
        "compactness": compactness,
        "histogram": mean_intensity,
        "mean_hue": mean_hue,
        "mean_saturation": mean_saturation,
        "mean_value": mean_value,
        "irregularity": irregularity
    }
    return features

# Fungsi untuk menampilkan histogram dalam streamlit
def display_histogram(hist):
    # Menggunakan matplotlib untuk menampilkan histogram
    plt.figure(figsize=(8, 6))
    plt.plot(hist, color='black')
    plt.title("Histogram Gambar")
    plt.xlabel("Intensitas Piksel")
    plt.ylabel("Frekuensi")
    plt.grid(True)
    
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close()
    
    histogram = Image.open(buf)
    return histogram

# Konfigurasi awal halaman
st.set_page_config(
    page_title="Melanoma Detection",
    page_icon="img/icon.png",
    layout="wide"
)

# Navigasi menu di sidebar
with st.sidebar:
    selected_menu = option_menu(
        menu_title= "Menu",
        options=["Home", "Prediction", "About App"],
        icons=['house', 'search','info-circle'],
        styles={
            "container": {"padding": "0!important", "background-color": "transparent"},
            "icon": {"color": "#fff", "font-size": "16px"},
            "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px", "--hover-color": "#2e2e2e"},
            "nav-link-selected": {"background-color": "#c26b30", "color": "#fff"},
        }
    )

if selected_menu == "Home":
    # Judul halaman
    st.title("Aplikasi Pendeteksi Kanker Kulit Melanoma")
    
    st.subheader("Tentang Penyakit")
    st.write("""
        Kanker Kulit Melanoma adalah jenis kanker kulit yang berkembang dari melanosit, yaitu sel-sel yang menghasilkan pigmen melanin pada kulit. Melanoma adalah bentuk kanker kulit yang paling berbahaya karena cenderung tumbuh dengan cepat dan dapat menyebar (metastasis) ke organ tubuh lainnya jika tidak segera ditangani.
    """)
    st.write("""
        Aplikasi ini dirancang untuk membantu mendeteksi kanker kulit melanoma berdasarkan gambar kulit yang diunggah. 
        Dengan memanfaatkan teknologi kecerdasan buatan dan pengolahan citra, aplikasi ini dapat mengidentifikasi dua kategori utama:
        kanker kulit jinak dan ganas.
    """)
    
    # Kategori yang dideteksi
    st.subheader("Kategori yang Dideteksi")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.image("img/benign.jpg", caption="Melanoma Jinak", use_container_width=True)
    with col2:
        st.write("""
             Melanoma jinak adalah jenis melanoma yang memiliki sifat pertumbuhan yang lambat dan tidak menyebar ke bagian tubuh lainnya. Ciri-cirinya lebih teratur, simetris, dan biasanya tidak menimbulkan gejala serius. Meskipun demikian, melanoma jinak tetap perlu diawasi karena dapat berkembang menjadi lebih berbahaya jika tidak diperhatikan dengan baik.
        """)
    
    with col3:
        st.image("img/malignant.jpg", caption="Melanoma Ganas", use_container_width=True)
    
    with col4:
        st.write("""
             Melanoma ganas adalah jenis melanoma yang sangat berbahaya dan memiliki potensi untuk menyebar (metastasis) ke organ tubuh lainnya, seperti kelenjar getah bening, paru-paru, atau organ vital lainnya. Melanoma ganas biasanya tumbuh lebih cepat, tidak teratur, dan memiliki sifat asimetris.
        """)

    # Cara melakukan prediksi
    st.subheader("")
    st.subheader("Cara Melakukan Prediksi")
    st.write("""
        1. Klik icon sidebar (>) di pojok kiri atas.
        2. Klik "Prediction" pada menu sidebar.
        3. Unggah gambar kulit yang ingin diperiksa menggunakan tombol Upload Gambar.
        4. Tunggu hasil prediksi yang akan ditampilkan setelah gambar diproses.
    """)

elif selected_menu == "Prediction":
    # Judul halaman
    st.title("Prediksi Kanker Kulit")
    
    # Tombol upload gambar
    st.write("Unggah gambar untuk melakukan prediksi:")
    uploaded_file = st.file_uploader("Upload gambar", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # st.image(uploaded_file, caption="Gambar yang diunggah", use_column_width=True)
        img = Image.open(uploaded_file)
        img_resized = img.resize((224, 224))
        img_array = np.array(img_resized)
        
        segmented_image = segment_image(img_array)
        edges_image = detect_edges(segmented_image)
        hist, mean_intensity = extract_histogram_mean(img_array)
        uji = extract_features_from_image(img_array)

        test_df = pd.DataFrame([uji])
        test_scaled = scaler.transform(test_df)
        prediction = model.predict(test_scaled)
        reshaped_prediction = prediction.reshape(-1)

        decoded_prediction = label_encoder.inverse_transform(reshaped_prediction)
        hasil = 'Jinak' if decoded_prediction == 'Benign' else 'Ganas'

        semua = [img_array, segmented_image, edges_image]
        capt = ["Gambar Original", "Hasil Segmentasi", "Hasil Deteksi Tepi"]
        
        row = st.columns(3)
            
        for i, col in enumerate(row):
            with col:
                st.image(semua[i], caption=capt[i], use_container_width=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])  # Rasio kolom untuk memusatkan gambar
        with col2:
            st.image(display_histogram(hist), width=520)
        
        st.header('')
        st.write(
            "Berdasarkan hasil analisis menggunakan model klasifikasi melanoma, kami telah memprediksi sifat dari melanoma yang terdapat pada gambar yang Anda unggah. "
            "Hasil prediksi ini memberikan informasi penting tentang apakah melanoma tersebut bersifat jinak atau ganas."
        )
        st.header(f'Hasil Prediksi: Melanoma {hasil}')
        if hasil == 'Ganas':
            st.write(
                "Hasil prediksi menunjukkan bahwa melanoma bersifat ganas. "
                "Sangat disarankan untuk segera periksakan diri Anda ke dokter kulit atau ahli dermatologi untuk pemeriksaan lebih lanjut dan mendapatkan perawatan yang sesuai. "
                "Hindari menggaruk atau mencoba menghilangkan area tersebut sendiri. "
                "Pastikan juga kulit terlindungi dari sinar matahari langsung dengan menggunakan pakaian pelindung atau sunscreen."
            )
        else:
            st.write(
                "Hasil prediksi menunjukkan bahwa melanoma bersifat jinak. "
                "Disarankan untuk tetap memantau perkembangan kondisi dan berkonsultasi dengan dokter kulit secara berkala untuk memastikan tidak ada perubahan yang mencurigakan. "
                "Tetap pantau kondisi kulit Anda secara rutin. "
                "Berikut beberapa langkah yang bisa Anda lakukan:\n"
                "- Hindari paparan sinar matahari berlebihan, terutama di siang hari.\n"
                "- Gunakan sunscreen dengan SPF 30 atau lebih setiap hari.\n"
                "- Amati perubahan ukuran, warna, atau bentuk pada area kulit yang mencurigakan.\n"
                "- Jika ada perubahan yang tidak biasa, segera konsultasikan ke dokter kulit."
            )


elif selected_menu == "About App":
    # Judul halaman
    st.title("Tentang Aplikasi")
    
    # Penjelasan aplikasi
    st.subheader("Deskripsi Aplikasi")
    st.write("""
        Aplikasi ini dibuat sebagai bagian dari proyek akhir mata kuliah Pengolahan Citra dan Grafika. Aplikasi ini bertujuan untuk membantu mendeteksi dan menganalisis jenis melanoma pada kulit, baik melanoma jinak maupun ganas, menggunakan teknik pengolahan citra dan model machine learning.
    """)
    
    st.markdown("**Aspek yang digunakan untuk deteksi antara lain:**")
    st.markdown("""
        1. **Segmentasi**  
        Menggunakan Otsu Thresholding untuk memisahkan objek (lesi) dari latar belakang gambar.
        2. **Deteksi Tepi**  
        Menggunakan algoritma Canny untuk mendeteksi tepi lesi setelah segmentasi, mengurangi gangguan latar belakang.
        3. **Kompakness (Perimeter dan Area)**  
        Menghitung perimeter dan area lesi berdasarkan kontur hasil segmentasi.
        4. **Asimetri**  
        Mengukur perbedaan bentuk lesi dengan membandingkan sisi kanan-kiri atau atas-bawah lesi.
        5. **Histogram**  
        Menganalisis distribusi intensitas piksel dalam gambar untuk menilai tekstur dan gradasi warna.
        6. **Aspek Warna**  
        Menghitung rata-rata hue, saturasi, dan value dalam ruang warna HSV untuk menilai keberagaman warna.
        7. **Irregularity Tepi**  
        Mengukur ketidakteraturan tepi lesi berdasarkan kontur hasil segmentasi.
        """)

    st.markdown("**Teknologi yang Digunakan dalam Pembuatan Aplikasi:**")
    st.markdown("""
        1. **Streamlit**  
        Untuk membuat antarmuka pengguna (UI) yang sederhana dan interaktif.
        2. **Python**  
        Sebagai bahasa pemrograman utama untuk pengolahan citra, analisis data, dan pembuatan model prediksi.
        3. **OpenCV**  
        Untuk pemrosesan citra, seperti segmentasi gambar, deteksi tepi, dan perhitungan area serta histogram.
        4. **Scikit-learn**  
        Untuk membuat model machine learning yang dilatih menggunakan dataset dari hasil pemrosesan citra open-cv.
        """)

    # Deskripsi tim
    st.subheader("Tim Pengembang")
    st.write("""
        Tim Pengembang Aplikasi Melanoma Detection yang terdiri dari mahasiswa jurusan Mekatronika dan Kecerdasan Buatan yang bersemangat untuk menerapkan teknologi kecerdasan buatan dalam dunia kesehatan.
        
        Anggota Tim:
        - Anita Neza Dewanti
        - Yudi Pratama Putra
        - Miqdad Fauzan Ghozwatulhaq
        - Permana Fadillah Sulaeman
        - Hazza Musyaffa
    """)