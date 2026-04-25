# ABC Store: Analisis Segment Toserba

status: Published
type: Post
date: April 19, 2026 9:59 PM
tags: data
category: project

# Disclaimer!

> Halaman ini adalah dokumentasi yang saya buat untuk salah satu proyek selama mengikuti Bootcamp Data Science di [kelas.work](http://kelas.work). Saya tidak, dalam bentuk apapun, terikat dengan perusahaan/badan usaha yang ada dalam studi kasus ini dan baik ABC Store maupun skenario yang ada merupakan langkah yang saya ambil untuk membuat proyek ini berjalan sebagaimana adanya dalam DuDi.
> 

<aside>
ℹ️

Dataset dapat ditemukan di [Kaggle](https://www.kaggle.com/datasets/ilkeryildiz/online-retail-listing).

</aside>

---

# Overview

Analisis RFM (Recency, Frequency, Monetary) adalah metode yang digunakan untuk memahami perilaku pelanggan dengan mengelompokkan mereka berdasarkan aktivitas belanja mereka. Dalam proyek ini, analisis RFM diterapkan dengan panduan dari mentor, bertujuan untuk memberikan insight yang dapat membantu pengambilan keputusan bisnis.

## Metodologi

— sekaligus bagian-bagian dalam dokumentasi berbentuk artikel ini.

1. Businesses Understanding
2. Pengumpulan dan Preprocecing Data
3. Menghitung dan Mengeksplorasi nilai RFM
4. Segmentasi Pelanggan
5. Evaluasi Pola yang ditemukan
6. Implementasi Hasil.

---

# 1. Businesses Understanding

## Latar Belakang

**Masalah apa yang dihadapi?**

ABC Store adalah toko serba ada yang menyediakan berbagai produk berkualitas dengan harga bervariasi. Toko ini dikenal dengan konsep personal branding yang sederhana namun tegas dalam setiap layanan dan produknya. ABC Store mengutamakan kepuasan pelanggan dengan menyediakan beragam kebutuhan sehari-hari—mulai dari kosmetik dan riasan wajah, peralatan kantor, mainan anak, hingga peralatan olahraga dan alat berat. Fokus pada keberagaman produk menjadikan ABC Store sebagai destinasi utama untuk belanja terpadu dalam satu layanan.

Sejak 2009, ABC Store telah menerapkan sistem informasi yang mencatat transaksi pelanggan. Sistem ini mencakup data seperti Invoice, StockCode, Description, dan Country untuk periode Desember 2009 hingga 2011. Setiap transaksi dicatat dalam sistem, termasuk transaksi yang dibatalkan (ditandai dengan kode 'C' pada kolom Invoice). Berdasarkan data yang terkumpul, ABC Store ingin melakukan analisis segmentasi pelanggan yang menjangkau pelanggan untuk mengembangkan strategi penjualan yang lebih efektif pada tahun 2012.

**Rumusan Masalah**

1. Produk apa yang sering dibeli berdasarkan segmentasi pelanggan?
2. Bagaimana pengaruh pembelian produk terhadap retensi pelanggan?
3. Bagaimana cara mengenali pelanggan yang paling berharga untuk meningkatkan loyalitas mereka?
4. Apakah transaksi yang dibatalkan memiliki pengaruh yang signifikan terhadap hasil dan kinerja model?

**Tujuan**

1. Mengidentifikasi pelanggan berdasarkan nilai dan tingkat loyalitas.
2. Menyesuaikan strategi pemasaran berdasarkan segmentasi RFM untuk menjangkau pelanggan secara lebih personal dan efektif.
3. Meningkatkan penjualan dan pendapatan.

**Bagaimana cara menggunakan data untuk menyelesaikan masalah tersebut?**

Setelah masalah didefinisikan dengan jelas, saatnya menentukan model yang akan digunakan. Pada studi kasus ABC Store ini, saya menggunakan pendekatan:

- **RFM (Recency, Frequency, Monetary) Analysis** - digunakan untuk segmentasi pelanggan berdasarkan tiga faktor utama:
    - **Recency (R)**: Seberapa baru pelanggan melakukan pembelian.
    - **Frequency (F)**: Seberapa sering pelanggan melakukan pembelian.
    - **Monetary (M)**: Seberapa banyak uang yang dibelanjakan pelanggan.
- **Clustering (Pengelompokan)** - digunakan untuk mengelompokkan pelanggan berdasarkan kesamaan karakteristik mereka, seperti frekuensi pembelian, total pengeluaran, atau produk yang dibeli. Algoritma clustering seperti K-Means atau Hierarchical Clustering digunakan untuk mengidentifikasi kelompok-kelompok yang memiliki pola serupa.

**Model algoritma machine learning apa yang digunakan?**

Untuk segmentasi pelanggan, saya menggunakan **K-Means clustering**. Berikut kelebihan dan kekurangan dari algoritma ini:

- **Kelebihan**
    - Sederhana dan mudah dipahami
    - Efisien untuk dataset besar
    - Menyesuaikan dengan tipe data numerik
- **Kelemahan**
    - Sensitif terhadap outliers
    - Memerlukan penentuan jumlah cluster
    - Tidak baik untuk data non-sferis

Jadi, saya akan menggunakan **K-Means clustering** untuk studi kasus segmentasi pelanggan di ABC Store, dengan mempertimbangkan kekuatan dan keterbatasannya.

---

# 2. Pengumpulan dan Preprocecing Data

Data Preparation, Import Library atau modul yang digunakan. Dalam proyek ini saya menggunakan beberapa library atau modul seperti:

```python
# basic function
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import datetime

# cluster dan segmentasi pelanggan, udah masuk modeling
!pip install yellowbrick
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from yellowbrick.cluster import KElbowVisualizer
from yellowbrick.cluster import SilhouetteVisualizer
```

Namun untuk tahap satu hingga tiga, yaitu saat ini, saya menggunakan yang dasar terlebih dahulu. Hal itu untuk menghemat penggunaan memori selama analisis awal, agar tidak menggangu ataupun memperlambat tampilan maupun sistem yang digunakan. Jika sudah selesai, baik manapun pilihan Anda, import atau sambungkan dataset ke dalam file notebook. Dataset dapat Anda temukan di informasi awal atau properti pada dokumentasi ini, scroll melalui link ini.

Untuk Goggle Colab (yang saya gunakan):

```python
from google.colab import drive
drive.mount('/content/drive')
```

```python
# load dataset
path = "/content/drive/MyDrive/Classroom Data Science/RFM/online_retail_listing.csv"

data = pd.read_csv(path, delimiter=';', encoding='latin-1')
data.head()
```

`Path` adalah variabel yang digunakan untuk menampung jalur file dataset. Jalur file tersebut disimpan dalam bentuk string agar ketika dataset yang disimpan dalam `data` dapat langsung diproses menggunakan baris parameter yang singkat dan tidak memakan tempat.

Kemudian untuk Jupyter, VsCode atau layanan lainnya yang sejenis, Anda dapat langsung mengubah isi dari variabel `path` menjadi lokasi directory Anda menyimpan dataset. Jangan melakukan *mount* Google Drive jika Anda menyimpan dataset pada directory lokal.

Misalnya, Anda dapat memasukkan `path` sebagai berikut:

```python
# load dataset
path = "c:/RFM/online_retail_listing.csv"

data = pd.read_csv(path, delimiter=';', encoding='latin-1')
data.head()
```

Saat dataset sudah tersambung dengan benar, akan tampil 5 baris awal dari dataset:

![Output dari data.head()](ABC%20Store%20Analisis%20Segment%20Toserba/5a863738-9ff5-481b-9e94-fbbe36fc6612.png)

Output dari data.head()

Untuk melihat kondisi lainnya, jalankan `data.tail()` dan `data.sample(5)` untuk melihat baris terbawah dan baris yang diambil secara acak. Anda dapat mengatur jumlah barisnya sebanyak yang Anda inginkan.

![Output dari data.sample(5)](ABC%20Store%20Analisis%20Segment%20Toserba/Screenshot_2025-01-28_140226.png)

Output dari data.sample(5)

![Output dari data.tail()](ABC%20Store%20Analisis%20Segment%20Toserba/Screenshot_2025-01-28_140430.png)

Output dari data.tail()

Belum ditemukan kolom Invoice yang mengandung kode ‘C’. Itu artinya, transaksi yang dibatalkan berada di luar area yang ditentukan dan bukan berada di transaksi awal, akhir maupun acak. Mengingikasikan, mungkin saja transaksi yang dibatalkan tidak memiliki pola tertentu dan bersifat random, sehingga memerlukan analisis sendiri untuk menemukannya.

Jalankan juga `data.info()` dan `data.describe()` untuk mencaritahu informasi statistik dan general dari data.

![Output dari data.info()](ABC%20Store%20Analisis%20Segment%20Toserba/Screenshot_2025-01-28_140648.png)

Output dari data.info()

![Output dari data.describe() untuk tipedata numerikal](ABC%20Store%20Analisis%20Segment%20Toserba/Screenshot_2025-01-28_140844.png)

Output dari data.describe() untuk tipedata numerikal

Dari penggunaan fungsi describe(), tanpa sadar pembagian dataset sebenarnya telah dilakukan. Karena untuk data dengan tipe seperti object dan sejenisnya, pemanggilan tidak dapat dilakukan secara langsung dan harus menggunakan describe()- pemanggilan langsung dapat dilakukan, namun hasilnya amembingunkan bagi sebagian orang dan saya mengambil jalan yang aman- dengan memilih kolom-kolom tertentu dari DataFrame untuk diterapkan pada fungsi `describe()`.

![Output dari data.describe() untuk tipedata kategorikal/string/object](ABC%20Store%20Analisis%20Segment%20Toserba/Screenshot_2025-01-28_141037.png)

Output dari data.describe() untuk tipedata kategorikal/string/object

Kemudian, dari beberapa kode yang telah dijalankan, adapun informasi yang didapatkan:

1. Variabel dalam dataset
    - Invoice - Nomor Invoice yang menjadi bukti transaksi
    - StockCode - Kode stock barang yang menjadi pengenal di sistem internal
    - Description - Nama barang/produk dalam display yang mendeskripsikan item tersebut
    - Quantity - Banyaknya barang yang dibeli oleh pelanggan dalam sebuah transaksi
    - InvoiceDate - Tanggal invoice dibuat pada saat pembelian terjadi
    - Price - Harga satuan dari barang yang dibeli
    - Customer ID - Kode unique untuk pelanggan yang melakukan transaksi
    - Country - Negara asal seorang pelanggan yang melakukan transaksi.
    
2. Indikasi yang ditimbulkan
    - Terdapat perbedaan jumlah nilai non null atas keseluruhan isi dari Description dan Customer ID, mengindikasikan adanya nilai yang hilang dari variabel tersebut.
    - Terdapat kesalahan tipedata, di mana variabel seperti InvoiceDate dan Price memiliki tipedata yang tidak tepat jika dibandingkan dengan isi yang ada dalam variabel-variabel tersebut.
    - Terdapat nilai minus pada variabel quantity dan kode C pada variabel Invoice, yang tidak masuk akal, dimana mungkin saja dapat mengindikasikan bahwa terdapat data yang tidak akurat pada keseluruhan isi data.

Untuk lebih jelasnya, akan dilakukan analisis terpisah dari informasi awal yang didapatkan.

***

## EDA Tahap Awal (Data Cleaning)

Untuk mengeksplorasi dan meneliti mengenai dataset dan variabel yang ada pada dataset, maka akan dilakukan Eksplorasi data awal yang meliputi:

- analisis masing-masing variabel (univariat)
- analisis satu variabel terhadap variabel lainnya (bivariat)

Tetapi sebelum melakukan analisis univariat dan bivariat, karena data sering kali mengandung noise seperti missing values, duplikasi atau bahkan tipe data yang tidak sesuai, pembersihan data harus dilakukan terlebih dahulu. Data yang tidak bersih dapat menyebabkan bias dan insight yang menyesatkan. Analisis univariat dan bivariat membutuhkan data yang bersih agar hasilnya akurat dan relevan.

***

### A. Menangani Inaccurate Data

Berdasarkan indikasi yang ditemukan  sebelumnya, terdapat beberapa column atau variabel yang memiliki data yang tidak akurat. Baik dari segi isi maupun tipedata-nya, variabel-variabel tersebut harus dibersihkan. 

Pertama-tama, saya akan melihat column Invoice yang mengandung kode C. Kode C ini secara kontekstual hanya berarti ‘cancelled’ atau ‘dibatalkan’ dan secara teori tidak memiliki pengaruh signifikan terhadap kinerja model untuk analisis RFM. Karena proses cleaning hanya memengaruhi data input (dataset) dan setelah setelah dataset di cleaning, model tetap berjalan tetap berjalan sesuai algoritmanya dan hasil klasterisasi (pengelompokan) mungkin tidak terlihat terlalu berbeda jika transaksi yang dibatalkan jumlahnya kecil.

Namun, jika transaksi yang dibatalkan jumlahnya besar, hasil analisis dan klasterisasi dapat berubah signifikan— yakni hanya karena model belajar dari data yang lebih valid.

**Mengapa ini penting?** Karena transaksi yang dibatalkan tidak mencerminkan aktivitas pembelian pelanggan secara nyata. Jika transaksi ini tidak dibuang, **akan terjadi** bias dalam perhitungan, **terutama** pada Monetary, karena transaksi yang dibatalkan akan memberikan nilai yang tidak valid.

```python
data[data["Invoice"].str.contains("C")]
```

Output:

![Screenshot 2025-01-28 143941.png](ABC%20Store%20Analisis%20Segment%20Toserba/Screenshot_2025-01-28_143941.png)

Seperti yang saya sebutkan di atas, adanya kode C dalam data tidak memiliki keberadaan yang begitu signifikan. Setidaknya 0.22% dari baris yang ada dalam dataset adalah pesanan yang dibatalkan.

```python
data = data[~data["Invoice"].str.contains("C", na = False)]data
```

Hasilnya adalah seperti ini:

![Screenshot 2025-01-28 144128.png](ABC%20Store%20Analisis%20Segment%20Toserba/Screenshot_2025-01-28_144128.png)

Selanjutnya adalah variabel InvoiceDate dan Price yang memiliki kesalahan tipedata dan format. 

- InvoiceDate - Tanggal invoice dibuat atau pada saat pembelian terjadi
    - Tipedata-nya: Object
    - Value/isi: Tanggal Pembelian, waktu
    - Yang seharusnya: Datetime
    - Pertimbangan: Tidak masuk akal jika tanggal dan waktu pembelian produk dinyatakan dalam tipedata object (yang biasanya untuk string atau categorical), dapat menghambat analisis mendatang karena nilai Recency tidak bisa ditemukan dalam tipedata object.

- Price - Harga satuan dari barang yang dibeli
    - Tipedata: Object
    - Value/isi: Harga barang, mata uang
    - Yang seharusnya: Float
    - Pertimbangan: Dalam satuan ukur uang, mata uang menunjukkan nilai dari suatu barang atau jasa. Sebagai satuan hitung atau *unit of account,* uang yang terdiri dari angka memiliki unit utama dan unit pecahan (desimal). Sehingga membuat tipedata floating menjadi pilihan yang sesuai, karena fleksibel untuk unit utama dan unit pecahan. Menggunakan tanda titik (.) sebagai pemisah desimal alih-alih tanda koma (,).

Setelah diperbaiki tipedata dan format-nya, hasilnya seperti:

![Menggunakan info() untuk melihat informasi umum setelah diperbaiki.](ABC%20Store%20Analisis%20Segment%20Toserba/Screenshot_2025-01-28_144317.png)

Menggunakan info() untuk melihat informasi umum setelah diperbaiki.

Nilai minus (kurang dari 0) pada variabel Quantity yang tidak masuk akal. Karena dapat mengartikan bahwa Customer ID tidak membeli produk apapun dalam transaksi.

```python
data = data[data["Quantity"] > 0]
data.info()
```

Karena terdapat produk dengan kuantitas di bawah 0, maka tidak menutup kemungkinan jika variabel lainnya juga terdapat hal yang demikian. Maka akan dilakukan pemeriksaan lagi terhadap variabel Price, yang memiliki kemungkinan terbesar karena merupakan satuan angka.

```python
data = data[data["Price"] > 0]
data.info()
```

![Hasil setelah data yang tidak akurat ditangani.](ABC%20Store%20Analisis%20Segment%20Toserba/Screenshot_2025-01-28_145122.png)

Hasil setelah data yang tidak akurat ditangani.

Berdasarkan *general information* di atas, terdapat perbedaan sebesar 2.42% dari *general information* yang pertama kali di run dalam notebook. Yakni sekitar 25 ribu nilai yang tidak valid telah disingkirkan dari dataset. Terlebih, 21 ribu missing value dari Description telah ditemukan penyebabnya! dan bahkan juga telah dihapuskan! Yakni karena missing value tersebut berkaitan dengan variabel lainnya yang tidak valid, seperti variabel Price dan Quantity. 

Dan sisanya, yakni variabel Customer ID yang memiliki missing value, akan menjadi awal dari bagian Missing Value.

***

### B. Menangani Missing Value

Untuk menangani missing value, entah menghapus atau dengan menggunakan metode penanganan missing value yang lain. Misalnya, seperti imputasi atau bahkan membuat model untuk menanganinya, saya perlu mengetahui terlebih dahulu variabel mana saja yang terdapat missing value-nya.

Walaupun hanya menyisakan satu variabel saja, hal ini tetap perlu untuk dilakukan.

![Perbandingan missing value dengan keseluruhan data pada dataset.](ABC%20Store%20Analisis%20Segment%20Toserba/Screenshot_2025-01-28_145527.png)

Perbandingan missing value dengan keseluruhan data pada dataset.

Dari keseluruhan data pada dataset, perbandingan dengan nilai yang kosong adalah sebesar 2.8%. Walaupun persentase itu menunjukkan jumlah yang kecil, apa penyebab nilai nan tetap di caritahu.

Sebagai perbandingan, lihatlah dua sampel yang diambil ini:

![Yang memiliki nilai NaN.](ABC%20Store%20Analisis%20Segment%20Toserba/Screenshot_2025-01-28_150427.png)

Yang memiliki nilai NaN.

![Yang tidak memiliki nilai NaN.](ABC%20Store%20Analisis%20Segment%20Toserba/Screenshot_2025-01-28_150617.png)

Yang tidak memiliki nilai NaN.

Dari indeks yang ditunjukkan, walaupun terdapat sebuah baris yang memiliki nilai NaN, hampir semua baris lainnya terisi dengan benar. Hal ini menunjukkan bahwa transaksi benar-benar ada dan memang masalahnya ada pada variabel tersebut.

![Informasi statistik dari missing value.](ABC%20Store%20Analisis%20Segment%20Toserba/Screenshot_2025-01-28_151017.png)

Informasi statistik dari missing value.

![Setelah dilakukan penghapusan.](ABC%20Store%20Analisis%20Segment%20Toserba/78d11645-b43b-493b-ac6a-9f6d0b60f310.png)

Setelah dilakukan penghapusan.

Dari hasil sejauh ini, adapun analisa saya:

- Sebagian besar transaksi memiliki jumlah barang (Quantity) kecil, dengan rentang 1 hingga 1.820 item, tetapi mayoritas hanya memesan 1 sampai 2 unit barang per transaksi (median = 1, kuartil ketiga = 2).
- Transaksi pada rentang harga rendah mendominasi, sehingga besar kemungkinan banyak transaksi bersifat kecil-kecilan atau dilakukan oleh pelanggan tanpa identitas tetap sehingga data dianggap tidak begitu relevan. Adapun kemungkinan lainnya ialah kesalahan manusia, kesalahan tekniis pada sistem (tidak memiliki logical test untuk syarat sebuah transaksi), atau masalah pengumpulan data yang tidak sempurna.
- Pendekatan paling tepat untuk menangani kondisi data yang kosong adalah menghapusnya. Karena data identitas yang hilang tidak dapat direkonstruksi dengan akurat dan berisiko merusak kualitas analisis jika diestimasi. Selain itu, celah dalam sistem atau proses pengumpulan data perusahaan, seperti ketidakkonsistenan pencatatan atau kerentanan sistem, dapat menjadi penyebab utama data yang hilang. Oleh karena itu, penghapusan data kosong tidak hanya memastikan keandalan analisis tetapi juga menyoroti perlunya peningkatan sistem pengelolaan data di perusahaan.

***

### C. Menangani Duplicated Value

```python
data.drop_duplicates(keep='first', inplace=True)
data.info()
```

 

***

### D. Feature Engineering

Setelah proses pembersihan data selesai, langkah feature engineering dilakukan untuk mempersiapkan data agar siap digunakan dalam model machine learning. 

**clean_data**

Pada tahap ini saya memisahkan data yang sudah bersih dari data awal menjadi variabel/dataset clean_data.

**OrderValue**

adalah fitur baru yang merupakan hasil kombinasi dari dua variabel, yaitu Price dan Quantity, yang saling berhubungan erat. Berdasarkan konsep monetary sendiri, fitur ini menunjukkan total kontribusi pelanggan dalam bentuk nilai pesanan mereka. OrderValue didapat dengan mengalikan harga satuan produk dengan jumlah produk yang dibeli.

**nums, cats dan datetime**

Memisahkan dataset menjadi beberapa variabel atau sub-dataset. Dalam kasus ini, saya menggunakan tiga pemisah dataset berdasarkan tipedatanya, yakni numerikal, kategorikal dan datetime (data dengan format tanggal).

***

### E. Analisis Univariat

**nums**

![Hist Invoice](ABC%20Store%20Analisis%20Segment%20Toserba/image.png)

Hist Invoice

![Hist Quantity](ABC%20Store%20Analisis%20Segment%20Toserba/image%201.png)

Hist Quantity

![Hist Price](ABC%20Store%20Analisis%20Segment%20Toserba/image%202.png)

Hist Price

![Hist Customer ID](ABC%20Store%20Analisis%20Segment%20Toserba/image%203.png)

Hist Customer ID

![Hist OrderVallue](ABC%20Store%20Analisis%20Segment%20Toserba/image%204.png)

Hist OrderVallue

**cats**

![Barh StockCode](ABC%20Store%20Analisis%20Segment%20Toserba/image%205.png)

Barh StockCode

![Barh Description](ABC%20Store%20Analisis%20Segment%20Toserba/image%206.png)

Barh Description

![Barh Country](ABC%20Store%20Analisis%20Segment%20Toserba/image%207.png)

Barh Country

**datetime**

![Lineplot InvoiceDate from datetime](ABC%20Store%20Analisis%20Segment%20Toserba/download_(39).png)

Visualisasi variabel datetime.

### F. Analisis Bivariat

**Numerikal ‘n Numerikal**

![image.png](ABC%20Store%20Analisis%20Segment%20Toserba/image%208.png)

Terdapat dua pasangan variabel yang menunjukkan korelasi positif. Untuk memperoleh variabel seperti OrderValue, terlebih dahulu diperlukan variabel Quantity dan Price.

- Hubungan yang kuat antara OrderValue dan Quantity menunjukkan bahwa Quantity memiliki pengaruh signifikan terhadap besarnya OrderValue. Namun, karena nilai korelasi tidak mencapai 1, meskipun keduanya berkorelasi positif, hubungan tersebut tidak bersifat mutlak atau deterministik.
- Sementara itu, hubungan antara OrderValue dan Price tergolong lemah dan hampir tidak terlihat. Korelasi keduanya berada dalam rentang 0.0 hingga 0.3, yang mendekati nol, menandakan bahwa hubungan linear antara kedua variabel sangat lemah. Meskipun demikian, adanya nilai korelasi yang lebih besar dari nol mengindikasikan kemungkinan adanya hubungan lain yang bersifat non-linear.

**Kategorikan ‘n Numerikal**

- **Langkah cepat untuk Section ini**
    - [**Invoice ‘n Description**](https://www.notion.so/ABC-Store-Analisis-Segment-Toserba-5b3c813f9c548242b0d201e1fe3ffa39?pvs=21)
        - [Top 10 Product Teratas](https://www.notion.so/ABC-Store-Analisis-Segment-Toserba-5b3c813f9c548242b0d201e1fe3ffa39?pvs=21)
        - Top Produk Terbawah
    - **OrderValue ‘n Customer ID**
    

Sejauh ini, variabel numerikal yang berpotensi memiliki hubungan dengan variabel kategorical adalah Invoice (jumlah transaksi) dan OrderValue (Total Pembelian). Sebagai variabel Independent yang memengaruhi variabel seperti Description dan Customer ID, kedua variabel tersebut sangat menarik untuk dibahas, karena berkenaan dengan *business interests* dan secara hakikat, adalah dasar dari Monetary dan pemetaan segment pelanggan.

**Invoice ‘n Description**

Berkenaan dengan pemahaman bisnis dalam analisis ini, [salah satunya](https://www.notion.so/ABC-Store-Analisis-Segment-Toserba-5b3c813f9c548242b0d201e1fe3ffa39?pvs=21) yaitu:

1. Produk apa yang sering dibeli berdasarkan segmentasi pelanggan?
2. Bagaimana pengaruh pembelian produk terhadap retensi pelanggan?

Maka akan dilakukan pembagian segment produk dalam ABC Store.

Sebenarnya pembagian ini sudah dilakukan pada tahap analisis univariat, yaitu pada bagian `Description` pada analisis variabel kategori. Namun, di bagian itu, kita hanya fokus pada jumlah maksimum produk berdasarkan kemunculan produk itu sendiri di bagian `Description`, dan bukan jumlah total invoice yang dibuat. Secara logis, Anda dapat menentukan mengapa suatu produk laku keras tidak hanya dengan melihat inventaris Anda (dalam hal ini variabel `Description`), tetapi juga dengan melihat bukti kuat yang Anda miliki.

Jadi, untuk menjawab masalah yang ada, mengenai segment produk yang sering dibeli dan pengaruh pembelian produk terhadap retensi pelanggan, saya menggunakan analisis Bivariat (di bawah ini).

**Top 10 Produk Teratas**

| index | Product | Transaction |
| --- | --- | --- |
| 5045 | WHITE HANGING HEART T-LIGHT HOLDER | 4982 |
| 3767 | REGENCY CAKESTAND 3 TIER | 3302 |
| 292 | ASSORTED COLOUR BIRD ORNAMENT | 2666 |
| 2391 | JUMBO BAG RED RETROSPOT | 2611 |
| 3153 | PARTY BUNTING | 2087 |
| 2608 | LUNCH BAG  BLACK SKULL\. | 2023 |
| 3842 | REX CASH+CARRY JUMBO SHOPPER | 1923 |
| 2621 | LUNCH BAG SPACEBOY DESIGN | 1873 |
| 4611 | STRAWBERRY CERAMIC TRINKET BOX | 1859 |
| 2245 | HOME BUILDING BLOCK WORD | 1844 |

Untuk mempermudah interpretasi, saya juga membuat visualisasi sederhana:

![17385083910698043856428067849345.png](ABC%20Store%20Analisis%20Segment%20Toserba/17385083910698043856428067849345.png)

*White Hanging Heart T-light Holder* adalah salah satu barang terbaik yang ditawarkan oleh ABC Store, diikuti oleh *Regency Cakestand 3 Tier* dan delapan produk lainnya. Namun, jika ABC Store memiliki 10 produk teratas, pasti ada juga produk yang kurang sukses. Untuk analisis yang lebih mendalam, masalah atau tantangan dapat memberikan informasi tambahan pada RFM.

**Top 10 Produk Terbawah**

Adalah analisis antara dua variabel yang dibuat untuk mencapai tujuan dari analisis dan bisnis dari pemahaman bisnis yang telah didefinisikan mengenai ‘bagaimana cara memahami pelanggan berdasarkan nilainya’.

| index | Description | Transaction |
| --- | --- | --- |
| 757 | BROWN COZY SQUARE PHOTO ALBUM | 1 |
| 355 | BAG FOR CHILDREN VINTAGE PINK | 1 |
| 354 | BAG FOR CHILDREN VINTAGE BLUE | 1 |
| 1772 | FOLKART HEART CHRISTMAS DECORATIONS | 1 |
| 1750 | FOLDING SHIRT TIDY | 1 |
| 1147 | CRACKED GLAZE EARRINGS BROWN | 1 |
| 1149 | CRACKED GLAZE NECKLACE BROWN | 1 |
| 359 | BAKING MOULD CUPCAKE CHOCOLATE | 1 |
| 3972 | S/4 BLACK DISCO PARTITION PANEL | 1 |
| 4960 | WALL ART , THE MAGIC FOREST | 1 |

Karena semua produk di atas memiliki jumlah pembelian yang sama, maka tidak efektif jika divisualisasikan. Sebaliknya, alasan mengapa produk-produk-produk tersebut memiliki performa yang rendah, dapat memberikan wawasan tambahan mengenai strategi pemasaran yang lebih efektif.

Pertama-tama, buatlah pertanyaan sederhana untuk masalah ini.

> Apakah ada hubungannya dengan tren waktu?
> 

Yang selanjutnya, akan memperluas jangkauan analisis ini, dimana bukan hanya tentang produk yang laku dan tidak laku saja, tetapi tentang … bagaimana bisa ada produk yang tidak laku?

Mari mulai!

Karena jumlah pembelian terkecil-nya sudah diketahui yakni 1, maka hanya product dengan sekali pembelian saja yang akan diambil. Selain itu, penggunaan sample berupa hanya 10 produk saja sudah tidak relevan, karena untuk memahami dan mencaritahu mengapa sampai ada produk dengan hanya sekali pembelian, penting untuk melihatnya secara keseluruhan.

![image.png](ABC%20Store%20Analisis%20Segment%20Toserba/image%209.png)

Interpretasi:

- Terdapat puncak penjualan di awal pembentukan ABC Store pada Desember 2009 dengan lebih dari 60 transaksi. Ini mungkin disebakan dari rasa antusiasme dan penasaran pelanggan mengenai perusahaan yang baru di buka. Namun, terdapat penurunan drastis di awal 2010, tetapi masih ada lonjakan pada bulan januari dan maretnya dan fluktuasi terjad sepanjang 2010 hingga 2011 dengan beberapa bulan memiliki sedikit transaksi.
- Kemudian September 2010 menunjukkan lonjakan signifikan lainnya, kemungkinan ada faktor musiman atau promosi. Pada 2011, distribusi transaksi lebih merata, meskipun jumlahnya tetap rendah dibandingkan dengan akhir 2009 dan awal 2010.

Jadi, memang terdapat tren pada waktu awal pencatatan transaksi. Akan tetapi, itu tidak memiliki pengaruh yang begitu besar, karena secara bertahap mulai terjadi penurunan terhadap pembelian. Dan tidak juga dapat dikatakan sebagai pola musiman, karena walaupun bulan pertama perusahaan menggunakan sistem informasi terjadi lonjakan transaksi terhadap barang-barang yang dinilai kurang laku, itu hanya sekali dan fluktuasi tetap berjalan dengan beberapa perubahan kecil.

> Apakah karena uang (harganya)?
> 

Sekarang setelah mengetahui bahwa terdapat variasi dalam waktu pembelian dan tren di periode awal pencatatan, maka untuk memperjelas pola dari produk yang memiliki data jual yang rendah dan untuk mengetahui apakah produk-produk tersebut memiliki sedikit penjualan karena harganya yang tidak kompetitif, murni karena tren ‘penasaran’ pelanggan yang sudah tidak relevan atau memang produk tersebut yang kurang diminati di pasaran, saya akan membandingkan harga dari tiap produk yang dibeli dengan harga rata-rata dari keseluruhan transaksi.

Dalam langkah ini, adapun maksud dari harga yang tidak kompetitif ialah yang melebihi rata-rata harga. Di mana dengan asumsi produk tersebut terlalu mahal atau harganya terlalu tinggi dari cakupan harga normal, pendekatan ini tidak hanya akan mengungkap ‘Produk ini harganya terlalu tinggi!’, tetapi melihat pola lain seperti negara atau pelanggan yang menjanjikan untuk gambaran analisis RFM dan bahkan pola waktu yang berhubungan dengan langkah sebelumnya.

![Screenshot 2025-02-04 162130.png](ABC%20Store%20Analisis%20Segment%20Toserba/Screenshot_2025-02-04_162130.png)

Di sini Anda dapat melihat bahwa  ada lima produk kurang kompetitif yang ditemukan di penampilan awal. Dan sisanya adalah produk yang kiranya memiliki harga yang sama dengan rata-rata dan yang lebih rendah. Tentu saja itu tidak dapat dijadikan acuan dan hanya gambaran awal mengetahui dua kelompok yang sudah dibuat.

Kemudian, secara keseluruhan perbandingannya adalah seperti ini:

![image.png](ABC%20Store%20Analisis%20Segment%20Toserba/image%2010.png)

Persentase dari harga yang terlalu tinggi adalah 35.23%, lebih rendah dari yang diharapkan. Kalau begitu, mari perluas kembali jangkauannya.

![barh country of low product ](ABC%20Store%20Analisis%20Segment%20Toserba/1738669369951777217431667300287.png)

barh country of low product 

Inggris (UK) tetap menjadi negara yang paling dominan dan menjanjikan. Negara tersebut adalah target pasar yang sangat besar. Dari sini, Anda dapat melihat bahwa produk-produk tersebut bukan hanya sekedar produk yang tidak laku saja. Apa yang ingin saya katakan adalah produk-produk yang tidak laku itu, tidak sepenuhnya tidak laku. Seberapa kecilnya itu tetap ada yang beli, dimana nilai terkecilnya adalah apa yang kita analisis saat ini.

Sebagai contoh, perhatikan visualisasi di bawah ini:

![Barh Customer ID of low product](ABC%20Store%20Analisis%20Segment%20Toserba/17386760643283777386366097863412.png)

Barh Customer ID of low product

Apa yang Anda lihat adalah gambaran lain yang diperoleh mengenai RFM.

Dari visualisasi tersebut kita tidak hanya tahu bahwa pelanggan dengan ID 14156.0 melakukan transaksi atau beli barang sebanyak 12 buah. Tetapi, 12 barang yang dibeli oleh ID 14156.0 adalah barang yang berasal dari kelas atau kategori tidak laku. Anda harus ingat bahwa nilai terendah dalam rentang transaksi adalah satu (1) dan bukan nol (0), jadi kita tidak benar-benar mencari produk yang benar-benar tidak laku. Tetapi mencari tahu pola ‘kenapa’-nya dari produk yang kurang memiliki daya jual. Dalam hal ini, tren waktu dan harga tidak memiliki pengaruh dan pola yang begitu signifikan. Sehingga, dapat diartikan bahwa terdapat kekurangan dari segi produk (di luar harga dan tren waktu), yang memerlukan perhatian khusus terhadap kontrol kualitas.

Ngomong-ngomong, jika Anda ingin melihat distribusi data-nya lebih dalam, Anda bisa menggunakan kode berikut ini:

```python
# Filter produk dengan harga terlalu rendah
low_price_products = result_data[result_data['Result'] == 'Harga sama atau rendah']
# Filter produk dengan harga terlalu tinggi
high_price_products = result_data[result_data['Result'] == 'Harga terlalu tinggi']
```

**OrderValue ‘n Customer ID**

Seperti yang sebelumnya dibahas, analisis dua variabel ini juga berkenaan dengan kepentingan bisnis yang ada. Yakni bagaimana untuk mengenali para pelanggan berdasarkan nilai mereka. Dan bagaimana cara untuk mengetahui sebuah nilai adalah dengan menilainya.

Untuk itu, kita harus melihatnya berdasarkan dua variabel potensial yang telah diketahui yaitu `Description`, yang sebelumnya sudah dibahas dan `Customer ID`. Sebagai pengingat, sebelumnya Anda telah mengetahui [bahwa terdapat](https://www.notion.so/ABC-Store-Analisis-Segment-Toserba-5b3c813f9c548242b0d201e1fe3ffa39?pvs=21) hubungan linear yang positif antara variabel numerikal. Dari analisis itu kita tahu bahwa `Quantity` dan `Price` adalah variabel yang memengaruhi `OrderValue`, dimana sebagai hasilnya kita dapat memakai `OrderValue` sebagai kandidat dari variabel numerik.

| index | Customer ID | TotalSpend |
| --- | --- | --- |
| 5674 | 18102.0 | 569501.5 |
| 2268 | 14646.0 | 516874.5 |
| 1781 | 14156.0 | 313437.62 |
| 2529 | 14911.0 | 285118.84 |
| 5035 | 17450.0 | 244784.25 |
| 1324 | 13694.0 | 192509.53 |
| 5094 | 17511.0 | 164753.55 |
| 67 | 12415.0 | 144458.37 |
| 4280 | 16684.0 | 141740.79 |
| 2676 | 15061.0 | 122493.16 |

Visualisasi adalah sebagai berikut:

![17386791146709036879437863958284.png](ABC%20Store%20Analisis%20Segment%20Toserba/17386791146709036879437863958284.png)

Kita bisa membandingkan hasilnya dengan [visualisasi sebelumnya](https://www.notion.so/ABC-Store-Analisis-Segment-Toserba-5b3c813f9c548242b0d201e1fe3ffa39?pvs=21), di mana pada produk yang kurang laku terdapat informasi mengenai para pelanggan yang membeli produk-produk tersebut dan dalam visualisasi ini adalah apa yang menyangkut dari keseluruhan transaksi.

Jika dilihat terdapat beberapa pelanggan yang berada dalam dua pembagian. Misalnya, pelanggan dengan ID 150156.0 yang dalam produk berdaya jual rendah menjadi pelanggan dengan pembelian tertinggi, dalam kelompok ini juga masuk dalam10 besar pelanggan dengan TotalSpend terbanyak.

---

# 3. Menghitung dan Mengeksplorasi nilai RFM

## Recency

Mengukur seberapa lama atau seberapa baru suatu tindakan/interaksi terakhir di lakukan. Untuk mencari nilai dari Recency, Frequency dan Monetary, pertama-tama harus mengelompokkan data-data yang ada berdasarkan nilai independent utama atau data yang dijadikan acuan untuk analisis RFM, yakni `Customer ID`.

```python
customer_data = clean_data.pivot_table(index="Customer ID",
                values=["InvoiceDate", "OrderValue"],
                aggfunc={"InvoiceDate": [min, max, pd.Series.nunique], "OrderValue": sum})
customer_data.head()
```

Output:

![Screenshot_20250205-200906.png](ABC%20Store%20Analisis%20Segment%20Toserba/Screenshot_20250205-200906.png)

Setelah membuat pivot table untuk mempermudah analisis, hal lain yang perlu dilakukan adalah mengubah nama kolom menjadi apa yang mewakili isinya.

```python
# Memberi nama pada columns customer_data
customer_data.columns = ["LastInvoiceDate", "FirstInvoiceDate", "Frequency", "MonetaryValue"]
```

Setelah itu, buat tanggal acuan untuk menghitung nilai Recency. Di mana pada tanggal itu yang kita gunakan adalah today, sebuah tanggal referensi yang melambangkan hari ini atau tanggal terakhir dari dataset.

```python
# Misalnya, untuk mendefinisikan 'today' sebagai tanggal terakhir di dalam dataset kolom LastInvoiceDate ini:

today = customer_data["LastInvoiceDate"].max() # acuan referensi tanggalnya
```

Kemudian hitung nilai Recency-nya.

```python
# Hitung recency: interval (hari) antara tanggal transaksi terakhir dan hari ini
customer_data["Recency"] = (today - customer_data["LastInvoiceDate"]) / np.timedelta64(1, 'D')
customer_data.head()
```

Output:

![Screenshot_20250205-204943_2.png](ABC%20Store%20Analisis%20Segment%20Toserba/Screenshot_20250205-204943_2.png)

![17387635040195054611869954118812.png](ABC%20Store%20Analisis%20Segment%20Toserba/17387635040195054611869954118812.png)

![17387639988723747220305542745068.png](ABC%20Store%20Analisis%20Segment%20Toserba/17387639988723747220305542745068.png)

Analisis:

Lebih dari 50% pelanggan berada di bawah rentang 100 hari, yang menunjukkan bahwa separuh pelanggan melakukan transaksi sekitar 3 bulan yang lalu. Kemudian, dengan adanya recency sebesar 6 bulan sampai ± 2 tahun, yakni berada di atas rentang 100 hari namun memiliki frequency pelanggan yang rendah, mengidentifikasikan bahwa sebagian besar pelanggan masih cukup aktif melakukan transaksi dan sedikitnya pelanggan yang tidak melakukan transaksi dalam waktu lama, menunjukkan skews positif (outlier yang menarik ada di sebelah kanan).

## MonetaryValue

![17387687270372333982061694328252.png](ABC%20Store%20Analisis%20Segment%20Toserba/17387687270372333982061694328252.png)

![17387688161035903825493534435682.png](ABC%20Store%20Analisis%20Segment%20Toserba/17387688161035903825493534435682.png)

Bedasarkan visualisasi di atas, terdapat nilai ekstrem yang dapat memengaruhi Frequency. Di mana nilai yang ekstrim tersebut adalah apa yang mewakili kontribusi pelanggan yang terlalu besar dan membuat pelanggan dengan kontribusi pembelian sedang atau normal menjadi tidak terlihat.

Untuk itu harus dilakukan penanganan yang tepat. Yaitu penghapusan.

Akan tetapi, tidak semuanya akan dihapuskan karena dalam beberapa konteks bisnis, nilai yang terlalu tinggi atau rendah dari yang lain dapat menggambarkan kondisi yang sebenarnya dan dianggap relevan. Bisa juga berpotensi menghilangkan wawasan penting dari data. Karenanya, berdasarkan visualisasi di atas, metode yang sesuai adalah dengan menggunakan persentil ke-99 sebagai cara untuk menghilangkan nilai yang ekstrim.

Cara ini bekerja dengan menghapus sebesar 1% nilai yang ekstrim atau outlier dari data dan mempertahankan sisanya sebagai bagian yang bersih. 

```python
# Hitung persentil 99 dari MonetaryValue dan simpan kedalam percentile_99
percentile_99 = np.percentile(customer_data["MonetaryValue"], 99)

# Filter data yang memiliki MonetaryValue di bawah atau sama dengan persentil 99
customer_data = customer_data[customer_data["MonetaryValue"] <= percentile_99] # filter 1% outlier dengan mempertahankan 99% data asli.
```

![17388113077138140572327618637705.png](ABC%20Store%20Analisis%20Segment%20Toserba/17388113077138140572327618637705.png)

![17388113681962857138156745131820.png](ABC%20Store%20Analisis%20Segment%20Toserba/17388113681962857138156745131820.png)

Analisis:

Secara garis besar, terdapat perbedaan dalam distribusi data. Di mana setelah melakukan penghapusan outlier menggunakan metode moderat seperti persentil ke-99, variasi dan persebaran data lebih terlihat dibandingkan dengan keadaan data sebelum dilakukan penghapusan. Dalam kasus ini, bar plot dan box plot menjadi lebih mudah untuk di baca, karena setelah outlier ditangani, customer yang termasuk ke dalam kategori normal jadi terlihat dan tidak tertutup oleh 1% customer yang sangat tinggi.

## Frequency

![17388126195481440958274198727471.png](ABC%20Store%20Analisis%20Segment%20Toserba/17388126195481440958274198727471.png)

![17388127401512617331774327226027.png](ABC%20Store%20Analisis%20Segment%20Toserba/17388127401512617331774327226027.png)

Analisis:
Sebagian besar pelanggan melakukan kurang dari 20 transaksi, dengan rata-rata 5,44 transaksi dan tidak lebih dari 3 atau lebih transaksi yang dilakukan oleh sebagian besar pelanggan. Namun, keberadaan outlier berupa pelanggan yang bertransaksi dalam jumlah lebih tinggi dari rata-rata, khususnya kelompok dalam kelompok 75%, menunjukkan bahwa hanya sedikit yang bertransaksi dalam jumlah banyak dan banyak pelanggan yang bertransaksi jarang. Nilai outlier menunjukkan bahwa ada sekelompok pelanggan  yang kaya atau mampu secara ekonomi tetapi segmen pelanggan mereka kecil. Di sisi lain, ada beberapa transaksi yang dilakukan oleh pelanggan dengan segmentasi besar.

Jadi, secara garis besar … **Insight** yang didapatkan melalui analisis RFM sejauh ini adalah:

Ketiga variabel menunjukkan distribusi yang condong ke kanan akibat keberadaan outlier. Dimana:

- **Recency**: Terbentuk dua kelompok pelanggan yang berbeda—sebagian besar masih aktif melakukan transaksi baru-baru ini, sementara sisanya jarang atau sudah lama tidak bertransaksi.
- **Frequency & Monetary Value**: Pola pembelian pelanggan terbagi menjadi dua segmen utama—kelompok besar dengan frekuensi transaksi rendah dan kontribusi pembelian kecil, serta kelompok kecil dengan transaksi yang lebih sering dan nilai pembelian yang lebih tinggi.

Analisis ini menunjukkan adanya perbedaan perilaku pelanggan yang dapat menjadi dasar segmentasi dan strategi pemasaran yang lebih tepat sasaran.

---

# 4. Segmentasi Pelanggan

Selanjutnya, data dan informasi yang ditemukan dari analisis sejauh ini akan digunakan untuk memverifikasi kelompok-kelompok pelanggan yang sudah terbentuk dan ditemukan menjadi lebih jelas lagi. Karena, jika sebelumnya beberapa kelompok sudah diketahui seperti banyaknya pelanggan yang bertransaksi kecil atau di bawah rata-rata, atau barang-barang mewah yang tetap memiliki target pasarnya. Pada tahap ini, semua hal itu akan diperjelas agar strategi pemasaran yang sesuai dapat dibentuk sesuai dengan konteks bisnis dan keadaan data yang sebenarnya.

## Modelling

Sebelum melanjutkan ke tahap modeling yang lebih jauh, terdapat beberapa persiapan yang harus dilakukan, yakni menyiapkan data sudah siap untuk proses modeling dan optimalisasi dataset tersebut. Salah satu cara untuk menentukan jumlah cluster pada modeling, yang digunakan pada kasus ini, adalah Elbow Method.

Ketika melakukan clustering, perlu untuk menentukan jumlah kelompok (kluster) yang ideal. Jika jumlah kluster terlalu sedikit, data dalam satu kluster menjadi terlalu bervariasi (kurang seragam). Jika jumlahnya terlalu banyak, model menjadi terlalu rumit. Elbow Method membantu menemukan jumlah kluster yang seimbang, di mana data cukup seragam di setiap kluster tanpa membuat model terlalu rumit.

Karena itu, jika Anda belum menginstal [library atau package](https://www.notion.so/ABC-Store-Analisis-Segment-Toserba-5b3c813f9c548242b0d201e1fe3ffa39?pvs=21)-nya, silahkan install terlebih dahulu.

Setelah semua requirement terpasang dengan lancar, pertimbangkan  seberapa besar data yang Anda punya dan hasil apa yang Anda harapkan dari Elbow—bagaimana dengan jumlah kluster yang optimal?

Karena jika data RFM Anda sangat besar, proses klustering pada setiap iterasi memerlukan waktu untuk menghitung jarak dan membuat kluster— komputasi-nya lama! Selain itu, pikirkan tentang tujuan analisis atau bisnis. Misalnya, jika Anda ingin memisahkan pelanggan berdasarkan kategori gender maupun perilaku belanja mereka, atau menggunakan itu semua sebagai dasar untuk strategi pemasaran yang lebih efektif lagi. Pilih rentang k yang relevan, seperti k=2 untuk pilihan kelompok kecil, 3 - 5 jika Anda ingin membuat berdasarkan segmentasi pelanggan atau dalam kasus saya k = 10. Pratik terbaik adalah gunakan nilai k yang kecil terlebih dahulu, misalnya mulailah dengan 2 sampai 10 lalu analisis grafik elbow untuk menentukan titik yang optimal.

Jika sudah, jalankan kode Anda dan biarkan model yang bekerja.

Ngomong-ngomong, Ini adalah kode saya:

```python
model_elbow = KElbowVisualizer(KMeans(random_state=1000), k=10)
model_elbow.fit(rfm_data)
model_elbow.show()
```

Output-nya:

![download (45).png](ABC%20Store%20Analisis%20Segment%20Toserba/download_(45).png)

Dalam dataset ini, titik yang optimal adalah 4. Di mana pada grafik, line menunjukkan penurunan yang tajam hingga k=4 lalu melandai, maka 4 adalah jumlah kluster yang optimal.

Lalu, definisikan model KMeans Anda dan terapkan labels kluster yang dihasilkan oleh model KMeans sebagai columns baru pada `rfm_data` dan lihat bagaimana persebaran setelahnya.

![Screenshot 2025-01-31 201535.png](ABC%20Store%20Analisis%20Segment%20Toserba/Screenshot_2025-01-31_201535.png)

Setelah mendefinisikan model KMeans, saya menambahkan beberapa metrik untuk mendukung kluster k=4-apakah itu benar-benar jumlah kluster yang saya perlukan untuk data saya? apakah saya butuh angka itu?

Jadi saya menambahkan Silhouette Score dan Calinski Harabasz Index dalam analisis saya.Itu adalah bebarapa keputusan di luar rencana saya. Dan hasilnya adalah seperti ini.

![Screenshot 2025-01-31 202602.png](ABC%20Store%20Analisis%20Segment%20Toserba/Screenshot_2025-01-31_202602.png)

Interpretasi:

- Jumlah klaster (k=4) yang ditentukan dari metode Elbow didukung oleh evaluasi Silhouette Score dan Calinski-Harabasz Index.
- Dengan Silhouette Score sebesar 0.68, klasterisasi cukup baik, meskipun mungkin masih ada tumpang tindih kecil di area tertentu.
- Calinski-Harabasz Index yang tinggi menunjukkan bahwa data sudah terbagi dengan baik antar-klaster.

![Screenshot 2025-02-07 163515.png](ABC%20Store%20Analisis%20Segment%20Toserba/Screenshot_2025-02-07_163515.png)

Dari hasilnya, saya jadi merasa yakin dan menggunakan k=4 untuk segment pelangggan ABC Store. Barulah setelah itu saya menambahkan label yang diperoleh dari model sebagai kolom baru untuk `rfm_data` yang saya beri nama `cluster_elbow`. `cluster_elbow` ini untuk mengetahui persebarannya, melihat bagaimana persebaran pelanggan berdasarkan pola rfm yang sebelumnya diamati, saya menggunakan catplot.

![catplot visualisasi](ABC%20Store%20Analisis%20Segment%20Toserba/image%2011.png)

catplot visualisasi

Dari visualisasi tersebut, Anda dapat mengetahui bahwa jumlah pelanggan lebih banyak berada di kluster satu. Kluster tersebut terlihat dominan jika dibandingkan dengan kluster lainnya, namun segmentasi tidak hanya diputuskan dari hal itu. Ada hal lain. Jadi saya menggunakan visualisasi lain agar pola dari keempat kluster terlihat dengan lebih baik. Yaitu berdasarkan tujuan analisis-nya- menyesuaikan strategi pemasaran dan meningkatkan revenue yang ada, maka saya membuat pasangan rfm untuk melihat nilai pelanggan sebagaimana adanya, menyesuaikan strategi pemasaran berdasarkan segmentasi RFM serta untuk meningkatkan pendapatan atau revenue perusahaan, maka adapun pasangan yang di analisis adalah Recency ‘n Frequency, Recency ‘n  MonetaryValue dan Frequency ‘n MonetaryValue.

## Segmentasi Pelanggan

![Recency ‘n Frequency](ABC%20Store%20Analisis%20Segment%20Toserba/image%2012.png)

Recency ‘n Frequency

![Recency ‘n MonetaryValue](ABC%20Store%20Analisis%20Segment%20Toserba/image%2013.png)

Recency ‘n MonetaryValue

![Frequency ‘n MonetaryValue](ABC%20Store%20Analisis%20Segment%20Toserba/image%2014.png)

Frequency ‘n MonetaryValue

Analisis:

- Recency ‘n Frequency
    - Kluster 0: Memiliki Frequency pembelian yang sangat rendah (< 20), namun setiap pelanggan ada disetiap kelas Recency.
    - Kluster 1:  Kurang terlihat karena tersebar secara acak dan kepadatan pelanggan yang sangat kecil, tetapi jika bisa diambil beberapa pola kecil (yang ada); kelompok dominan yang terbentuk ialah yang berada di rentang Recency yang sangat kecil, yakni 10 bulan terakhir dengan rata-rata transaksi yang besar (10 sampai >60).
    - Kluster 2: Memiliki persebaran yang hampir mirip dengan kluster sebelumnya, tetapi terdapat sedikit perluasan segment dimana nilai Recency cenderung lebih tersebar secara merata dan hampir menyamai  pembagian kluster 0, namun tidak dengan kepadatannya. Sedangkan nilai Frequency yang dominan ada pada rentang 10 - 40 transaksi.
    - Kluster 3: Dalam jumlah kepadatan pelanggan, kluster ini  berada di urutan terakhir kedua setelah kluster 1. Dimana memiliki rentang Recency yang rendah secara dominan, namun tetap ada history dalam rentang lain dan memiliki pola Frequency tinggi yang hampir mirip dengan kluster 1.
    
- Recency 'n MonetaryValue
    - Kluster 0: Memiliki kepadatan dan rentang yang hampir sama dengan segment sebelumnya, dimana pada kluster ini pengeluaran dan frequency yang dilakukan pelanggan berada pada skala yang kecil, tetapi recency-nya ada pada setiap rentang pembagian waktu.
    - Kluster 1: Kluster ini adalah kluster yang paling dominan dari segi pengeluaran, berbanding lurus dengan jumlah transaksi pada perbandingan sebelumnya di mana kluster 1 ialah kluster dengan rentang transaksi yang lebih tinggi. Kebanyakan pelanggan pada kluster ini adalah mereka yang sering sampai cukup lama tidak aktif dalam melakukan transaksi. Perlu upaya untuk mempertahankan pelanggan dalam kluster ini.
    - Kluster 2: Kluster ini cukup padat dan berada di rentang yang normal, yakni lebih tinggi dari kluster 0 dan lebih rendah dari kluster 3. Dimana walaupun lebih rendah dari kluster 3 dan 4, padatnya kluster ini lebih tinggi daripada dua kluster tersebut, sehingga menunjukkan bahwa pelanggan dalam kluster 2—yakni adalah pelanggan dengan pengeluaran sedang hingga menengah yang memiliki kecendrungan untuk aktif baru-baru ini.
    - Kluster 3: Kluster ini adalah yang menjadi pembatas antara kluster dengan pengeluaran normal dan yang sangat tinggi. Sehingga dapat dilihat bahwa pelanggan dalam kluster ini adalah mereka yang berada dalam rentang pengeluaran 7000 - 15000 dan cenderung aktif, tetapi tidak banyak user dalam kluster ini.
- Frequency ‘n MonetaryValue
    - Kluster 0: Pelanggan pada kluster ini hanya ada di rentang yang sangat kecil, di mana hubungan antara jumlah transaksi dan pengeluaran yang dilakukan berbanding lurus, yakni sangat rendah dan di bawah kluster lainnya.
    - Kluster 1: Pelanggan pada kluster ini adalah mereka yang berada di puncak transaksi dan pengeluaran terbanyak. Dimana transaksi yang tersebar pada kluster ini cukup merata dan ada pada hampir kelas Frequency. Selain itu, walaupun terdapat beberapa pelanggan yang memiliki kesamaan dengan kluster lainnya dalam hal rentang Frequency,  kontribusi yang dikeluarkan kluster ini lebih besar. Mengindikasikan pembelian barang-barang mewah dan kondisi ekonomi kaya dalam kluster ini. Akan tetapi, kepadatan dalam kluster ini tidak banyak jika dibandingkan dengan kluster lainnya.
    - Kluster 2: Pelanggan pada kluster ini memiliki kecendrungan untuk melakukan sedikit hingga transaksi dalam jumlah normal. Dimana walaupun rentang jumlah  transaksi pada kluster ini tidak jauh berbeda dari kluster 0, pengeluaran yang dilakukan pelanggan dalam kluster ini lebih tinggi daripada kluster tersebut. Sehingga, dapat diartikan bahwa pelanggan dalam kluster ini membeli produk yang sedikit lebih mahal namun di bawah batas wajar dari barang-barang mewah.
    - Kluster 3: Persebaran pada kluster ini tidak lebih padat ataupun tidak begitu renggang dibandingkan dengan kluster lainnya. Di mana sebagian besar pelanggan tersebar cukup merata, dengan beberapa perluasan segment dari kluster 2 dan ada di tingkat yang lebih rendah dari kluster 1, mengindikasikan bahwa pelanggan dalam kluster ini membeli produk yang sedikit lebih banyak dari kluster 2 namun dengan harga yang lebih tinggi.

Sehingga dapat disimpulkan bahwa:

- Kluster 0: Recency Tinggi (merata), Frequency dan MonetaryValue yang sangat rendah— Pelanggan dalam seluruh kelas Recency dengan pembelian dan pengeluaran yang kecil. Paling padat.
- Kluster 1: Recency sangat rendah, Frequency dan MonetaryValue yang sangat tinggi— Pelanggan aktif dengan daya beli dan pengeluaran yang tinggi. Paling renggang.
- Kluster 2:  Recency tinggi,  Frequency  dan MonetaryValue yang rendah. Cenderung agak padat dibanding dengan kluster 1 dan 3— Pelanggan dengan daya beli yang kecil dan sudah lama tidak aktif.
- Kluster 3: Recency rendah (cukup merata), Frequency dan cenderung tinggi. Kepadatan yang cenderung normal ke bawah (sedikit lebih tinggi dari kluster 1)— Pelanggan yang cenderung aktif dengan daya beli dan pengeluaran menegah ke atas.

Dari hasil itu, secara statistik-nya adalah sebagai berikut:

> Toggle list di bawah adalah yang akan menambahkan block code baru berupa hasil dari fungsi description() dari kluster.
> 
> 
> 
> - Kluster Descriptive
>     
>     ```python
>     Describe Cluster 0 for Selected Columns:
>                Recency    Frequency  MonetaryValue
>     count  4471.000000  4471.000000    4471.000000
>     mean    241.335073     2.899799     745.190820
>     std     213.335697     2.371867     594.804859
>     min       0.018750     1.000000       2.900000
>     25%      44.113889     1.000000     266.675000
>     50%     183.938194     2.000000     559.970000
>     75%     405.968403     4.000000    1099.860000
>     max     733.138889    25.000000    2369.550000
>     
>     Describe Cluster 1 for Selected Columns:
>               Recency   Frequency  MonetaryValue
>     count   84.000000   84.000000      84.000000
>     mean    44.505556   31.309524   19898.352857
>     std     86.272514   20.257961    3600.954030
>     min      0.128472    3.000000   15172.140000
>     25%      5.723438   19.000000   16924.955000
>     50%     17.032639   28.000000   19531.535000
>     75%     40.264583   38.000000   22813.400000
>     max    570.850000  122.000000   26885.540000
>     
>     Describe Cluster 2 for Selected Columns:
>               Recency   Frequency  MonetaryValue
>     count  970.000000  970.000000     970.000000
>     mean    77.884832   10.623711    3998.704287
>     std    113.747662    6.323711    1250.806112
>     min      0.000000    1.000000    2380.130000
>     25%     12.121528    7.000000    2954.985000
>     50%     33.522569   10.000000    3712.940000
>     75%     82.085069   13.000000    4862.862500
>     max    686.070833  100.000000    6963.800000
>     
>     Describe Cluster 3 for Selected Columns:
>               Recency   Frequency  MonetaryValue
>     count  276.000000  276.000000     276.000000
>     mean    49.313519   20.692029    9965.781489
>     std    106.107148   10.993357    2162.076766
>     min      0.021528    1.000000    6995.360000
>     25%      6.072917   13.000000    8202.375000
>     50%     15.039583   19.000000    9490.070000
>     75%     34.991667   27.000000   11694.025000
>     max    648.915278   71.000000   14896.870000
>     ```
>     

- Kluster 0 : Recency sangat rendah. Frequency dan MonetaryValue yang terendah.
- Kluster 1 : Recency sangat rendah, Frequency dan MonetaryValue sangat tinggi.
- Kluster 2 : Recency tinggi,  Frequency  dan MonetaryValue yang rendah.
- Kluster 3 : Recency rendah, Frequency dan MonetaryValue tinggi.

---

# 5. Evaluasi Pola yang ditemukan

![image.png](ABC%20Store%20Analisis%20Segment%20Toserba/image%2015.png)

Analisa:

- Recency (Kiri)
    - Kluster 0 memiliki nilai max tertinggi dibanding dengan kluster lain, menunujukan bahwa pelanggan dalam kluster ini sudah lama tidak melakukan transaksi.
    - Kluster dengan nilai lebih kecil memiliki pelanggan yang melakukan transaksi baru-baru ini. Dengan urutan; kluster 0 (Merah) > kluster 2 (Biru) > kluster 3 (Ungu) > kluster 1 (Hijau).
- Frequency (Tengah)
    - Frequency tertinggi ada pada kluster 1, yang menunjukkan pelanggan dalam kluster ini lebih sering dalam melakukan transaksi.
    - Dengan urutan; kluster 1 (Hijau) > cluster 3 (Ungu) > cluster 2(Biru) > kluster 0 (Merah), kluster 0 adalah kelompok pelanggan yang paling jarang dalam melakukan transaksi.
- MonetaryValue (Kanan)
    - Berbanding lurus dengan nilai Frequency-nya, kluster 1 memiliki nilai max tertinggi dibanding dengan kluster lainyya, menunjukkan bahwa pelanggan dalam kluster ini adalah mereka yang paling banyak melakukan pengeluaran.
    - Dengan urutan; Kluster 1 (Hijau) > Kluster 3 (Ungu) > Kluster 2 (Biru) > Kluster 0 (Merah), berarti pelanggan dalam kluster 0 adalah yang paling sedikit menghabiskan uang.
    

***

Berdasarkan beberapa pendekatan yang diambil, maka adapun **insight** yang dapat diperoleh dari analisis ini adalah sebagai berikut:

| Kluster | R | F | M | Karakteristik | Segmentasi Pelanggan | Strategi |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | Sangat tinggi | Sangat rendah | Sangat rendah | Pelanggan yang sudah sangat lama tidak melakukan transaksi lagi, bahkan kemungkinan besar sudah berpindah ke kompetitor. | Lost Customers | Lakukan Re-engagement Campaign. |
| 1 | Sangat rendah | Sangat tinggi | Sangat tinggi | Pelanggan setia yang sering berbelanja dan memiliki nilai transaksi yang tinggi. | Loyal Customers | Berikan program dan layanan loyalitas  |
| 2 | Tinggi | Rendah | Rendah | Pelanggan yang sebelumnya aktif tetapi sudah lama tidak bertransaksi. Mereka memiliki risiko tinggi untuk meninggalkan bisnis atau beralih ke kompetitor. | At-Risk Customers | Berikan penawaran (e.x. diskon) ekslusif  |
| 3 | Rendah | Tinggi | Tinggi | Pelanggan yang mungkin tidak sering berbelanja, tetapi setiap transaksi yang mereka lakukan bernilai sangat besar. | High-Value Customers | Berikan kesempatan akses ke program ekslusif   seperti layanan yang ada pada kelas loyal |

---

# 6. Implementasi Hasil

Setelah memperoleh insight dan segmentasi pelanggan, terdapat beberapa hal yang menjadi perhatian utama. Analisis RFM dan K-Means Clustering memberikan pemahaman yang lebih mendalam mengenai kelompok pelanggan serta pola belanja mereka, yang berpengaruh terhadap strategi bisnis dan profitabilitas. Namun, keterbatasan dalam penerapan atau ilustrasi nyata dari strategi pemasaran membuat analisis ini terasa kurang mencerminkan aplikasinya di dunia nyata. Oleh karena itu, diperlukan langkah konkret untuk menerjemahkan hasil analisis ke dalam strategi yang dapat diimplementasikan secara langsung. Di antaranya :

## Lost Customers

**Strategi yang direkomendasikan:** Melakukan Re-engagement ‘Campaign’— yaitu kampanye pemasaran yang dirancang khusus untuk melakukan re-engagement— yakni tindakan umum atau upaya menarik kembali pelanggan, pengguna, atau audiens yang sebelumnya aktif tetapi kemudian menjadi tidak aktif.

Dalam kampanye ini, ada beberapa upaya terstruktur dengan strategi yang lebih jelas:

- Personalized Outreach!— Kirimkan pesan personal kepada pelanggan dengan penawaran ekslusif.
- Win-Back Offers—Menawarkan penawaran eksklusif agar pelanggan yang sudah lama tidak aktif tertarik kembali.
- Retargeting & Social Media Ads— Menggunakan iklan digital untuk menjangkau kembali pelanggan yang sudah lama tidak aktif.
- Feature Reintroduction— Memperkenalkan fitur baru atau peningkatan layanan yang bisa menarik kembali pelanggan.
- Free Trial or Product Sampling— Memberikan trial gratis atau sampel produk dari Early Access Programs untuk mendorong pelanggan mencoba kembali.
- Product Recommendations Based on Purchase History— Mempersonalisasi rekomendasi produk berdasarkan kebiasaan belanja sebelumnya.

## Loyal Customers

**Strategi yang direkomendasikan:** Memberikan Program dan Layanan Loyalitas— yakni strategi pemasaran yang dirancang untuk mendorong pelanggan agar tetap setia pada suatu merek, produk, atau layanan

Dari strategi ini terdapat sup-proyek yang dapat dijalankan, yakni:

- Personalized Communication & Appreciation— Mengirim pesan yang menunjukkan apresiasi atas kesetiaan mereka.
- Exclusive Access & Early Access Program— Member mendapatkan akses lebih awal ke produk baru atau diskon spesial.
- Referral Program— Pelanggan yang mengajak teman untuk bergabung mendapatkan hadiah atau diskon.
- Surprise & Delight Strategy— Memberikan kejutan seperti hadiah ulang tahun atau voucher tanpa pemberitahuan sebelumnya.
- Poin Reward (Point-Based Loyalty Program)— Pelanggan mengumpulkan poin dari transaksi yang bisa ditukar dengan hadiah atau diskon.
- Exclusive Discounts & Incentives— Menawarkan diskon atau poin loyalitas ekstra untuk mendorong transaksi ulang.
- Product Recommendations Based on Purchase History.

## At-Risk Customers

**Strategi yang direkomendasikan:** Melakukan Re-engagement sebelum pelanggan mulai kehilangan ketertatikan terhadap brand dan produk.

Dalam strategi ini, terdapat sup-proyek yang dapat dijalankan, yaitu:

- Personalized Communication— Menyesuaikan pesan berdasarkan preferensi pelanggan agar tetap engaged.
- VIP & Early Access Programs— Memberikan akses sementara ke  Exclusive Access.
- Retargeting & Remarketing Ads— Menampilkan iklan kepada pelanggan yang sudah jarang berinteraksi.
- Product Recommendations Based on Purchase History

## High-Value Customers

**Strategi yang direkomendasikan:** VIP Exclusive Access & Early Access Program— Berikan kesempatan akses ke program ekslusif seperti layanan yang ada pada kelas loyal, dalam hal ini:

- Personalized Communication & Appreciation untuk meningkatkan kesetiaan pelanggan terhadap brand dan produk.
- Targeted Discounts & Upselling— Menawarkan produk tambahan atau diskon berdasarkan kebiasaan belanja mereka.
- Referral Program untuk mengajak pelanggan baru
- Poin Reward (Point-Based Loyalty Program) untuk mendapatkan diskon dan meningkatkan kelas pelanggan.
- Product Recommendations Based on Purchase History

---

# Penutup

Dengan memahami pola pembelian tiap segmen pelanggan, ABC Store dapat merancang strategi pemasaran yang lebih efektif untuk meningkatkan retensi pelanggan dan meningkatkan profitabilitas.

- Kode ‘C’ pada variabel Invoice [tidak memiliki](https://www.notion.so/ABC-Store-Analisis-Segment-Toserba-5b3c813f9c548242b0d201e1fe3ffa39?pvs=21) pengaruh signifikan kepada data hasil maupun model *machine learning*.
- Dalam konteks bisnis dan pemanfaatan yang mendalam untuk ABC Store, diperlukan Quality Control (QC) [untuk](https://www.notion.so/ABC-Store-Analisis-Segment-Toserba-5b3c813f9c548242b0d201e1fe3ffa39?pvs=21) memastikan kualitas produk atau layanan yang dihasilkan perusahaan tetap sama.
- [Diperlukan](https://www.notion.so/ABC-Store-Analisis-Segment-Toserba-5b3c813f9c548242b0d201e1fe3ffa39?pvs=21) variabel mengenai kategori produk untuk memperkaya informasi seputar produk dan  segmentasi pelanggan (e.x. Category).
- Lost Customers perlu strategi re-engagement agar kembali melakukan transaksi.
- Loyal Customers harus diberikan program loyalitas agar tetap setia.
- At-Risk Customers bisa ditargetkan dengan promosi khusus sebelum mereka benar-benar churn.
- High-Value Customers harus diperlakukan sebagai VIP, dengan strategi eksklusif agar terus berbelanja.