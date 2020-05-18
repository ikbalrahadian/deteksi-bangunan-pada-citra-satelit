# Deep Learning : Deteksi Bangunan Pada Citra Satelit

---

Pada laman ini akan dijelaskan alur pembuatan model untuk mendeteksi objek bangunan pada citra satelit menggunakan pretrained model **YOLOv3**. Deteksi bangunan pada model akan menggunakan beberapa baris kode sederhana dari **ImageAI** sebagai pustaka python untuk membangun sistem dengan kemampuan deep learning dan penglihatan komputer. Tahapan training model dilakukan menggunakan komputer virtual dari **Google Colab**. 

### Daftar Isi
<a href="#persiapan" > :white_square_button: Persiapan</a>
<a href="#pembuatansetdata" > :white_square_button: Pembuatan set data</a>
<a href="#pembuatanskripkode1" > :white_square_button: Pembuatan skrip kode training model</a>
<a href="#trainingmodel" > :white_square_button: Training model</a>
<a href="#pembuatanskripkode2" > :white_square_button: Pembuatan skrip kode deteksi bangunan</a>
<a href="#deteksibangunan" > :white_square_button: Deteksi bangunan</a>

### Persiapan
<div id="persiapan"></div>

Untuk memulai pembuatan model deteksi bangunan beberapa hal yang perlu disiapkan adalah sebagai berikut:
1. **Citra satelit resolusi tinggi** sebagai bahan untuk membuat set data. Citra satelit dapat diunduh secara gratis dari laman *Open Digital Program - DigitalGlobe* : (http://www.digitalglobe.com/ecosystem/open-data) untuk beberapa daerah terdampak bencana. Pada kasus ini citra satelit yang digunakan adalah wilayah Palu, Sulawesi Tengah yang mengalami tsunami pada tahun 2018.
2. **Pretrained model YOLOv3** sebagai model yang telah dilatih sebelumnya menggunakan set data COCO. Pretrained model ini akan dilatih untuk mendeteksi bangunan menggunakan set data baru. 
3. **Labelimg** sebagai alat anotasi grafis dapat dicari dan diunduh pada laman *PyPI* : (https://github.com/tzutalin/labelImg). Labelimg digunakan untuk Supervised Learning dengan memberikan anotasi pada sampel bangunan yang terdapat pada set data citra satelit. Anotasi akan disimpan sebagai file XML untuk masing-masing citra pada set data.
4. **VS Code** sebagai teks editor untuk melakukan pembuatan skrip kode training model dan deteksi bangunan dapat diunduh pada laman *Visual Studio Code* : (https://code.visualstudio.com/).
5. **Akun google baru** untuk menyediakan gdrive yang memiliki cukup kapasitas untuk menampung hasil training model yang dapat dibuat pada laman *Google* : (https://accounts.google.com/signup).
