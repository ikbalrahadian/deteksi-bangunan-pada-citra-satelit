# Deep Learning : Deteksi Bangunan Pada Citra Satelit

---

Pada laman ini akan dijelaskan alur pembuatan model untuk mendeteksi objek bangunan pada citra satelit menggunakan pretrained model **YOLOv3**. Deteksi bangunan pada model akan menggunakan beberapa baris kode sederhana dari **ImageAI** sebagai pustaka python untuk membangun sistem dengan kemampuan deep learning dan penglihatan komputer. Tahapan training model dilakukan menggunakan komputer virtual dari **Google Colab**. 

### Daftar Isi
- <a href="#persiapan" > :white_square_button: Persiapan</a>
- <a href="#pembuatansetdata" > :white_square_button: Pembuatan set data</a>
- <a href="#pembuatanskripkode1" > :white_square_button: Pembuatan skrip kode training model</a>
- <a href="#trainingmodel" > :white_square_button: Training model</a>
- <a href="#pembuatanskripkode2" > :white_square_button: Pembuatan skrip kode deteksi bangunan</a>
- <a href="#deteksibangunan" > :white_square_button: Deteksi bangunan</a>


### Persiapan
<div id="persiapan"></div>

Untuk memulai pembuatan model deteksi bangunan beberapa hal yang perlu disiapkan adalah sebagai berikut:
- **Citra satelit resolusi tinggi** sebagai bahan untuk membuat set data. Citra satelit dapat diunduh secara gratis dari laman *Open Data Program - DigitalGlobe* : (http://www.digitalglobe.com/ecosystem/open-data) untuk beberapa daerah terdampak bencana. Pada kasus ini citra satelit yang digunakan adalah wilayah Palu, Sulawesi Tengah yang mengalami tsunami pada tahun 2018.
- **Pretrained model YOLOv3** sebagai model yang telah dilatih sebelumnya menggunakan set data COCO dapat diunduh pada laman *pjreddie* : (https://pjreddie.com/media/files/yolov3.weights). Pretrained model ini akan dilatih untuk mendeteksi bangunan menggunakan set data baru.
- **Labelimg** sebagai alat anotasi grafis dapat dicari dan diunduh pada laman : (https://github.com/tzutalin/labelImg). Labelimg digunakan sebagai Supervised Learning dengan memberikan anotasi pada sampel bangunan yang terdapat pada set data citra satelit. Anotasi akan disimpan sebagai file XML untuk masing-masing citra pada set data.
- **VS Code** sebagai teks editor untuk melakukan pembuatan skrip kode training model dan deteksi bangunan dapat diunduh pada laman *Visual Studio Code* : (https://code.visualstudio.com/).
- **Akun google baru** untuk menyediakan gdrive yang memiliki cukup kapasitas untuk menampung hasil training model yang dapat dibuat pada laman *Google* : (https://accounts.google.com/signup).


### Pembuatan set data
<div id="pembuatansetdata"></div>

Untuk membuat model deteksi bangunan diperlukan set data sebagai sampel. Set data berisi kumpulan citra satelit yang berisi gambar bangunan. Sampel akan digunakan pada proses training model untuk mempelajari dan mengenali karakteristik bangunan pada citra satelit. Pembuatan set data dilakukan dengan tahapan sebagai berikut:
1. Menyiapkan citra satelit berukuran 600x600 piksel yang berisi objek bangunan, jumlah citra satelit yang direkomendasikan adalah 300 (menghasilkan akurasi deteksi >75%). Penamaan citra satelit disarankan mengikuti urutan angka (misal: 1.jpg, 2.jpg, ..., 300.jpg).
2. Melakukan anotasi objek bangunan pada setiap citra satelit di set data menggunakan labelimg. Anotasi dilakukan dengan memberikan kotak pembatas dan label/nama pada setiap sampel bangunan. Anotasi akan menghasilkan file berekstensi XML untuk setiap citra pada set data (misal: 1.xml, 2.xml, ..., 300.xml).
3. Setelah seluruh pasangan citra satelit dan anotasi objek bangunan selesai dibuat, kemudian buat folder baru untuk menyimpan set data dengan nama *Bangunan*.
4. Dalam folder *Bangunan* kemudian buat dua folder baru dengan nama *train* dan *validation*.
5. Pada masing-masing folder *train* dan *validation* kemudian buat dua folder baru dengan nama *images* dan *annotations*.
6. Masukkan 80% set data ke dalam folder *train* dan 20% set data ke dalam folder *validation*. Dalam hal ini 80% pasangan citra (.jpg) dan anotasi (.xml) masing-masing dimasukkan ke dalam folder *images* dan *annotations* pada folder *train*, dan 20% pasangan citra (.jpg) dan anotasi (.xml) masing-masing dimasukkan ke dalam folder *images* dan *annotations* pada folder *validation*.
6. Setelah hal di atas selesai dilakukan, akan terlihat struktur folder set data sebagai berikut:
   ```
   >> train        >> images       >> 1.jpg
                                      2.jpg
                                      3.jpg
                                      ...
                   >> annotations  >> 1.xml
                                      2.xml
                                      3.xml
                                      ...
                                      
   >> validation   >> images       >> ...
                                      298.jpg
                                      299.jpg
                                      300.jpg
                   >> annotations  >> ...
                                      298.xml
                                      299.xml
                                      300.xml
   ```
