---
title: Final Projects Bootcamp Data Science
date: 2024-02-10
draft: false
categories: Portofolio
tags:
  - Data
---

# Disclaimer!

> Halaman ini adalah proyek yang saya ajukan sebagai siswa dalam program Bootcamp Data Science di kelas.work. Baik konteks dan skenario merupakan bagian untuk pembelajaran dan simulasi dalam Bootcamp.

# Overview

Web page screenshot

![cover - by Web 2 PDF](/images/docs-google-com-presentation-d-1zW0IV0oYhs_e3JpBUDP0erida9tgBHbv-edit-usp-sharing-ouid-116791460179490765452-rtpof-true-sd-true.jpg)

{{ < callout > }}

Hasil yang diharapkan: [Presentasi Project](https://docs.google.com/presentation/d/1zW0IV0oYhs_e3JpBUDP0erida9tgBHbv/edit?usp=drivesdk&ouid=116802401847442666522&rtpof=true&sd=true)

{{ <\callout > }}

<br>

# Pendahuluan

Dokumentasi ini merangkum empat studi kasus utama dalam data science, mulai dari analisis statistik hingga implementasi pembelajaran mesin dan deep learning. Setiap studi kasus mencakup pendekatan unik untuk memecahkan permasalahan berbasis data, memberikan wawasan, dan menawarkan solusi praktis. Proyek ini bertujuan untuk mengeksplorasi berbagai permasalahan nyata yang ada dalam dunia data, seperti mengetahui efektivitas dari suatu kegiatan, analisis harga dalam pasar, mesin yang dapat memberikan rekomendasi kepada pengguna hingga mempelajari bahasa isyarat menggunakan klasifikasi gambar.

<br>

# Detail Proses

## Studi Kasus 1: Sales Force Training

**Latar Belakang**

Perusahaan X ingin meningkatkan penjualan mereka. Dari data penjualan sebelumnya menunjukkan bahwa penjualan rata-rata yaitu $100 per transaksi. Setelah melakukan training kepada pekerja sales, data penjualan terbaru (yang diambil dari 25 sampel pekerja sales) tersimpan dalam tabel data di bawah ini :

![Sumber: Bootcamp](/images/Screenshot_20241217-150446_1.png)

**Rumusan Masalah**

1. Apakah terdapat peningkatan signifikan dalam jumlah transaksi setelah pelatihan?
2. Bagaimana pengaruh pelatihan pekerja sales terhadap rata-rata penjualan per transaksi di Perusahaan X?

**Tujuan**

1. Mengaplikasikan analisis dari uji statistika untuk menentukan apakah harus menolak atau menerima hipotesis nol (H₀) berdasarkan hasil uji.
2. Menilai apakah pelatihan sales force memiliki dampak signifikan terhadap rata-rata penjualan.

**Highlight Proyek**

- **Metode**: Statistik deskriptif dan inferensial (One-Sample T-Test).
- **Hasil Utama**: Tidak terdapat peningkatan signifikan dalam rata-rata transaksi setelah pelatihan.
- **Insight**: Pelatihan belum dan/atau bukan strategi yang efektif untuk meningkatkan rata-rata jumlah transaksi pekerja sales di perusahaan X.

<br><br>

## Studi Kasus 2: Housing Price

**Latar Belakang**

Dalam era informasi saat ini, banyaknya pilihan yang tersedia mengenai suatu properti, seperti ukuran, lokasi, dan kualitas bangunan, dapat memberikan wawasan yang berharga jika dianalisis dengan tepat. Data-data itu, ketika digabungkan dan dianalisis, dapat membantu dalam memahami dinamika pasar serta memberikan panduan untuk pembelian. Hal tersebut menunjukkan bahwa variabel seperti lokasi dan ukuran tanah memainkan peran penting dalam menentukan harga properti, tetapi hubungan ini belum dipahami sepenuhnya.

**Rumusan Masalah**

1. Apa saja faktor-faktor yang mempengaruhi harga jual suatu properti?
2. Apakah tiap faktor atau variabel memiliki hubungan yang saling memengaruhi terhadap harga jual?

**Tujuan**

Eksplorasi berbagai faktor yang memengaruhi harga properti menggunakan dataset “Housing Price”

**Highlight Proyek**

- **Metode**: Exploratory Data Analysis dan Transformasi Data (Log Transformation).
- **Hasil Utama**: Lokasi dengan fasilitas lengkap menjadi faktor dominan, dan luas bangunan memiliki korelasi tinggi terhadap harga properti.
- **Insight**: Segmentasi pasar properti berdasarkan fasilitas dan ukuran bangunan terlihat jelas.

<br><br>

## Studi Kasus 3: Machine Learning

**Latar Belakang**

Proyek ini menggunakan dataset dari 73.516 pengguna dan 12.294 judul anime untuk menganalisis preferensi penonton dan memberikan rekomendasi anime yang relevan.

**Rumusan Masalah**

Apa saja faktor yang memengaruhi  pengguna dalam memberikan rating terhadap suatu anime?

**Tujuan**

1. Membangun model yang dapat memberikan rekomendasi kepada pengguna
2. Memberikan pengguna rekomendasi anime berdasarkan pola interaksi pengguna sebelumnya.
3. Memberikan pengguna rekomendasi anime berdasarkan anime (konten) yang telah sebelumnya telah disukai oleh pengguna.

**Highlight Proyek**

- **Metode**:

    1. Collaborative Filtering untuk mengidentifikasi pola rating pengguna.
    2. Content-Based Filtering untuk menentukan kesamaan antar anime berdasarkan genre.

- **Hasil Utama**:

    1. Sistem rekomendasi berhasil menampilkan anime dengan rating relevan, seperti Noragami dan Bleach.
    2. Genre thriller dan kompleks mendapatkan rating tertinggi, mencerminkan preferensi penonton terhadap cerita mendalam.

- **Insight**: Rekomendasi mencerminkan preferensi pengguna, baik secara acak maupun berdasarkan pilihan pengguna. Namun, pengujian lebih lanjut diperlukan untuk mengatasi data dengan distribusi tidak seimbang

<br><br>

## Studi Kasus 4: Deep Learning

**Latar Belakang**

American Sign Language (ASL) adalah bahasa visual yang digunakan oleh komunitas tunarungu dan individu dengan gangguan pendengaran di Amerika Utara. Bahasa ini memiliki tata bahasa yang khas, memadukan gerakan tangan, ekspresi wajah, dan gerakan tubuh untuk menyampaikan makna. ASL bukan hanya digunakan untuk mengeja huruf atau angka, tetapi juga untuk mengekspresikan ide, perasaan, dan konsep yang lebih kompleks. Sebagai bahasa yang kaya dan beragam, ASL berbeda dari bahasa lisan yang lebih umum digunakan.

**Rumusan Masalah**

1. Bagaimana model deep learning dapat mempelajari dan mengenali pose tangan dalam ASL?
2. Apakah hasil dari kinerja model memang dapat diterapkan di kehidupan nyata?

**Tujuan**

1. Membangun model klasifikasi gambar untuk mengenali huruf dalam bahasa isyarat ASL.
2. Mengoptimalkan performa model menggunakan teknik augmentasi data dan hyperparameter tuning. 

**Highlight Proyek**

- **Metode**: CNN menggunakan Sequential API dengan hyperparameter tuning.
- **Hasil Utama**:

    1. Model mencapai akurasi validasi 100% setelah 10 epoch, namun risiko overfitting perlu diperhatikan.
    2. Model kesulitan mengenali kelas tertentu seperti R dan N.

- **Insight**:

    1. Model berkinerja baik untuk sebagian besar kelas, namun memerlukan perbaikan untuk kelas-kelas dengan performa rendah.
    2. Teknik augmentasi data atau peningkatan jumlah data untuk kelas tertentu dapat meningkatkan akurasi.

<br><br>

# Kesimpulan dan Penutup

Dokumentasi ini mencakup berbagai pendekatan dalam data science untuk memecahkan tantangan nyata:

- Analisis Statistik dan Eksploratif dalam _Sales Force Training_ dan _Housing Price_.
- Implementasi Machine Learning untuk _Anime Recommender System_.
- Pemanfaatan Deep Learning untuk klasifikasi bahasa isyarat dalam _ASL Classification_.

Setiap studi kasus tidak hanya memberikan solusi praktis, tetapi juga insight yang dapat digunakan untuk pengembangan strategi dan inovasi di masa depan.
