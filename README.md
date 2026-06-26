# Sistem Klasifikasi Sampah Menggunakan YOLOv8

## Deskripsi Proyek

Sistem Klasifikasi Sampah adalah aplikasi berbasis Computer Vision yang dirancang untuk mengidentifikasi dan mengklasifikasikan jenis sampah secara otomatis menggunakan model YOLOv8 Classification. Sistem ini bertujuan membantu proses pengelolaan sampah dengan memberikan informasi mengenai kategori sampah serta rekomendasi penanganan yang sesuai.

Aplikasi ini menggunakan Streamlit sebagai antarmuka pengguna sekaligus menjalankan proses inferensi model secara langsung. Dengan begitu, aplikasi tidak bergantung pada URL API atau server backend terpisah.

---

## Fitur Utama

* Upload gambar sampah melalui antarmuka web.
* Klasifikasi jenis sampah secara otomatis menggunakan YOLOv8.
* Menampilkan tingkat kepercayaan (confidence score).
* Menampilkan probabilitas seluruh kelas.
* Memberikan rekomendasi penanganan sampah berdasarkan hasil prediksi.
* Prediksi langsung dari aplikasi Streamlit tanpa request HTTP ke backend.

---

## Kategori Sampah

Model mampu mengklasifikasikan sampah ke dalam kategori berikut:

| Kategori       | Deskripsi                                                                   |
| -------------- | --------------------------------------------------------------------------- |
| Organic        | Sampah yang berasal dari sisa makhluk hidup dan dapat terurai secara alami. |
| Recyclable     | Sampah yang dapat didaur ulang menjadi produk baru.                         |
| Non-Recyclable | Sampah yang tidak dapat didaur ulang melalui proses umum.                   |
| Hazardous      | Sampah berbahaya dan beracun yang memerlukan penanganan khusus.             |

---

## Arsitektur Sistem

```text
Pengguna
    |
    v
Streamlit App
    |
    v
Model YOLOv8 Classification
    |
    v
Hasil Prediksi
```

---

## Teknologi yang Digunakan

### Aplikasi

* Streamlit
* Ultralytics YOLOv8
* Pillow

### Machine Learning

* YOLOv8 Classification
* Python

---

## Struktur Proyek

```text
APK_Sampah/
|
├── streamlit_app.py
├── backend.py
├── best.pt
├── requirements.txt
├── data.yaml
├── prepare_dataset.py
├── train/
├── val/
└── README.md
```

---

## Instalasi

### Clone Repository

```bash
git clone https://github.com/username/waste-classification-system.git
cd waste-classification-system
```

---

Install dependensi:

```bash
pip install -r requirements.txt
```

Jalankan Streamlit:

```bash
streamlit run streamlit_app.py
```

Aplikasi akan berjalan pada:

```text
http://localhost:8501
```

---

## Tampilan Aplikasi

### Halaman Utama

Tambahkan screenshot pada folder:

```text
screenshots/home.png
```

### Hasil Prediksi

Tambahkan screenshot pada folder:

```text
screenshots/result.png
```

---

## Pengembangan Selanjutnya

Beberapa pengembangan yang dapat dilakukan pada sistem ini:

* Integrasi dengan aplikasi mobile.
* Deteksi menggunakan kamera secara real-time.
* Deployment berbasis cloud.
* Penambahan kategori sampah yang lebih beragam.
* Integrasi dengan sistem pengelolaan sampah pintar.
* Dukungan multi-bahasa.

---

## Tujuan Proyek

Proyek ini dikembangkan sebagai implementasi Computer Vision dan Deep Learning untuk mendukung pengelolaan sampah yang lebih efektif serta meningkatkan kesadaran masyarakat dalam melakukan pemilahan sampah berdasarkan kategorinya.

---

## Penulis

**Nama:** 

Andrian Faturohman 		103052400011

Jeff Raymond Santoso 103052400020

M. Rasyid Alburaidy  		103052400023

Amir Hamzah Khairurrijal  	103052400077

Farrel Krisna Saputra 		103052400087
**Program Studi:** S1 Data Science
**Universitas:** Telkom University
**Tahun:** 2026
