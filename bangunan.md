# Deep Learning : Deteksi Bangunan Pada Citra Satelit

---

Pada laman ini akan dijelaskan alur pembuatan model untuk mendeteksi objek bangunan pada citra satelit menggunakan pretrained model **YOLOv3**. Deteksi bangunan pada model akan menggunakan beberapa baris kode sederhana dari **ImageAI** sebagai pustaka python untuk membangun sistem dengan kemampuan deep learning dan penglihatan komputer. Tahapan training model dilakukan menggunakan komputer dari **Google Colab**. 

## Daftar Isi
- <a href="#persiapan" > :white_square_button: Persiapan</a>
- <a href="#pembuatansetdata" > :white_square_button: Pembuatan set data</a>
- <a href="#pembuatanskripkode1" > :white_square_button: Pembuatan skrip kode training model</a>
- <a href="#trainingmodel" > :white_square_button: Training model</a>
- <a href="#pembuatanskripkode2" > :white_square_button: Pembuatan skrip kode deteksi bangunan</a>
- <a href="#deteksibangunan" > :white_square_button: Deteksi bangunan</a>


## Persiapan
<div id="persiapan"></div>

Untuk memulai pembuatan model deteksi bangunan beberapa hal yang perlu disiapkan adalah sebagai berikut:
- **Citra satelit resolusi tinggi** sebagai bahan untuk membuat set data. Citra satelit dapat diunduh secara gratis dari laman *Open Data Program - DigitalGlobe* : (http://www.digitalglobe.com/ecosystem/open-data) untuk beberapa daerah terdampak bencana. Pada kasus ini citra satelit yang digunakan adalah wilayah Palu, Sulawesi Tengah yang mengalami tsunami pada tahun 2018.
- **Pretrained model YOLOv3** sebagai model yang telah dilatih sebelumnya menggunakan set data COCO dapat diunduh pada laman berikut : (https://github.com/ikbalrahadian/deteksi-objek/releases). Pretrained model diunduh dari laman *pjreddie* dan telah dikonversi dari format .weight ke dalam format .h5 yang dapat dibuka pada Keras API. Pretrained ini akan dilatih untuk mendeteksi bangunan menggunakan set data baru.
- **Labelimg** sebagai alat anotasi grafis dapat dicari dan diunduh pada laman : (https://github.com/tzutalin/labelImg). Labelimg digunakan sebagai Supervised Learning dengan memberikan anotasi pada sampel bangunan yang terdapat pada set data citra satelit. Anotasi akan disimpan sebagai file XML untuk masing-masing citra pada set data.
- **VS Code** sebagai teks editor untuk melakukan pembuatan skrip kode training model dan deteksi bangunan dapat diunduh pada laman *Visual Studio Code* : (https://code.visualstudio.com/).
- **Akun Google baru** untuk menyediakan gdrive yang memiliki cukup kapasitas untuk menampung hasil training model yang dapat dibuat pada laman *Google* : (https://accounts.google.com/signup).

## Pembuatan Set Data
<div id="pembuatansetdata"></div>

Untuk membuat model deteksi bangunan diperlukan set data sebagai sampel. Set data berisi kumpulan citra satelit yang berisi gambar bangunan. Sampel akan digunakan pada proses training model untuk mempelajari dan mengenali karakteristik bangunan pada citra satelit. Pembuatan set data dilakukan dengan tahapan sebagai berikut:
1. Menyiapkan citra satelit berukuran 600x600 piksel yang berisi objek bangunan, jumlah citra satelit yang direkomendasikan adalah >300 (menghasilkan akurasi deteksi >75%). Penamaan citra satelit disarankan mengikuti urutan angka (misal: 1.jpg, 2.jpg, 3.jpg, ... dst.).

   <img src="https://github.com/ikbalrahadian/deteksi-objek/blob/master/dataset_image.JPG" width="900">

2. Melakukan anotasi objek bangunan pada setiap citra satelit di set data menggunakan labelimg. Anotasi dilakukan dengan memberikan kotak pembatas dan label/nama pada setiap sampel bangunan. Anotasi akan menghasilkan file berekstensi XML untuk setiap citra pada set data (1.xml, 2.xml, 3.xml, ... dst.).

   <img src="https://github.com/ikbalrahadian/deteksi-objek/blob/master/labelling.png" width="900">

3. Setelah seluruh pasangan citra satelit dan anotasi objek bangunan selesai dibuat, kemudian buat folder baru untuk menyimpan set data dengan nama ***bangunan***.
4. Dalam folder ***bangunan*** kemudian buat dua folder baru dengan nama ***train*** dan ***validation***.
5. Pada masing-masing folder ***train*** dan ***validation*** kemudian buat dua folder baru dengan nama ***images*** dan ***annotations***.
6. Masukkan 80% set data ke dalam folder ***train*** dan 20% set data ke dalam folder ***validation***. Dalam hal ini 80% pasangan citra (.jpg) dan anotasi (.xml) masing-masing dimasukkan ke dalam folder ***images*** dan ***annotations*** pada folder ***train***, dan 20% pasangan citra (.jpg) dan anotasi (.xml) masing-masing dimasukkan ke dalam folder ***images*** dan ***annotations*** pada folder ***validation***.
6. Setelah hal di atas selesai dilakukan, akan terlihat struktur folder set data sebagai berikut:
   ```
       >> bangunan----------->> train---------->> images------------>> 1.jpg
             |                    |                                    ...
             |                    |                                    240.jpg
             |                     ------------>> annotations------->> 1.xml
             |                                                         ...
             |                                                         240.xml
              -------------->> validation------>> images------------>> 241.jpg
                                  |                                    ...
                                  |                                    300.jpg
                                   ------------>> annotations------->> 241.xml
                                                                       ...
                                                                       300.xml
   ```


## Pembuatan Skrip Kode Training Model
<div id="pembuatanskripkode1"></div>

Penulisan skrip kode dilakukan dengan menggunakan **VS Code** dengan isi skrip kode sebagai berikut:
```python
from imageai.Detection.Custom import DetectionModelTrainer
trainer = DetectionModelTrainer()
trainer.setModelTypeAsYOLOv3()
trainer.setDataDirectory(data_directory="bangunan")
trainer.setTrainConfig(object_names_array=["bangunan"], batch_size=4, num_experiments=100, train_from_pretrained_model="pretrained-yolov3.h5")
trainer.trainModel()
```

Kode skrip diatas memiliki penjelasan untuk setiap baris sebagai berikut:
1. Importing kelas training model deteksi dari pustaka python ImageAI
2. Pendefinisian model trainer : deteksi
3. Pendefinisian tipe jaringan deep learning : YOLOv3
4. Pendefinisian direktori set data yang digunakan : folder 'bangunan'
5. Pendefinisian konfigurasi model trainer dalam parameter berikut:
   - **object_names_array** : berisi nama objek dalam set data
   - **batch_size** : berisi ukuran batch dalam proses training
   - **num_experiments** : berisi jumlah iterasi jaringan melakukan training set data
   - **train_from_pretrained_model** : berisi pretrained model yang akan digunakan
6. Memulai proses training model

Skrip kode untuk melakukan training model kemudian diberi nama *training* dan disimpan dalam format .py (training.py). 


## Training Model
<div id="trainingmodel"></div>

Sebelum melakukan training model, dibuat **akun Google baru** terlebih dahulu. Menggunakan akun baru yang telah dibuat, kemudian training model dilakukan dengan menggunakan komputer dari **Google Colab** melalui tahapan berikut:
1. Mengunggah set data (folder 'bangunan'), pretrained model YOLOv3 ('yolov3.h5'), dan skrip kode ('training.py') ke dalam google drive.
   <img src="https://github.com/ikbalrahadian/deteksi-objek/blob/master/sc1.png" width="900">
2. Masuk ke laman (https://colab.research.google.com/), kemudian pilih *NEW NOTEBOOK*.
   <img src="https://github.com/ikbalrahadian/deteksi-objek/blob/master/sc2.png" width="900">
3. Ganti nama file menjadi main.ipynb, kemudian tulis daftar perintah berikut pada google colab:
   - menghubungkan google drive pada komputer
     ```python
     from google.colab import drive
     drive.mount('/content/drive')
     ```
   - masuk ke direktori google drive pada komputer
     ```python
     cd drive/My\ Drive
     ```
   - instalasi perangkat yang dibutuhkan
     ```python
     !pip install tensorflow-gpu==1.13.1
     !pip install tensorflow_estimator==1.13.0
     !pip install imageai --upgrade
     !pip install opencv-python
     !pip install keras
     ```
   - melakukan perintah training model
     ```python
     !python training.py
     ```
   Beberapa daftar perintah diatas akan terlihat pada google colab sebagai berikut:
   <img src="https://github.com/ikbalrahadian/deteksi-objek/blob/master/sc3.png" width="900">
4. Setelah daftar perintah tersebut ditulis, kemudian pilih **Connect** ke komputer google colab menggunakan **runtime GPU**.
   <img src="https://github.com/ikbalrahadian/deteksi-objek/blob/master/sc5.png" width="900">
5. Setelah terhubung ke komputer google colab, kemudian play satu-persatu daftar perintah yang telah dibuat pada bagian (3) dengan menekan tombol play di sebelah skrip kode yang telah ditulis.
   <img src="https://github.com/ikbalrahadian/deteksi-objek/blob/master/sc4.png" width="900">
6. Proses training model akan terlihat pada google colab sebagai berikut. Lamannya proses training model bergantung pada jumlah iterasi yang dilakukan.
   <img src="https://github.com/ikbalrahadian/deteksi-objek/blob/master/sc6.png" width="900">
7. Setelah seluruh epoch training model selesai dilakukan, akan dihasilkan folder *cache*, *json*, *logs*, dan *models* pada folder *bangunan*. Model hasil training dari setiap epoch akan berada pada foler *models*, dan konfigurasi model yang dihasilkan akan berada pada folder *json* terlihat seperti gambar sebagai berikut.
   <img src="https://github.com/ikbalrahadian/deteksi-objek/blob/master/sc7.png" width="900">
   
   
## Pembuatan Skrip Kode Deteksi Bangunan
<div id="pembuatanskripkode2"></div>

Skrip kode deteksi bangunan ditulis untuk melakukan proses deteksi bangunan pada input citra satelit area studi. Skrip kode ditulis menggunakan **VS Code** dengan format .py dan disimpan dengan nama *prediction* (prediction.py). Berikut adalah isi skrip kode deteksi bangunan dengan beberapa keterangan kode pada setiap baris diberi simbol #.

```python

# import pustaka
from imageai.Detection.Custom import CustomObjectDetection
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import xlsxwriter
import os

# membuat direktori baru
if (os.path.isdir('output')) == False:
    os.mkdir('output')
if (os.path.isdir('coordinate')) == False:
    os.mkdir('coordinate')
if (os.path.isdir('summary')) == False:
    os.mkdir('summary')
if (os.path.isdir('entirety')) == False:
    os.mkdir('entirety')

# konfigurasi deteksi objek
detector = CustomObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath("masukkan_nama_file_model_yang_akan_digunakan_disini")
detector.setJsonPath("detection_config.json")
detector.loadModel()

# plot semua koordinat
valuex_dot_all = 0
valuey_dot_all = 0
plotx_dot_all = []
ploty_dot_all = []
number_image_row = 6
number_image_col = 0
check_img_last = Image.open("./input/image_6.jpg")
width_image_last, height_image_last = check_img_last.size

# membuat file all
workbook_all = xlsxwriter.Workbook('./entirety/plot_all.xlsx')
worksheet_all = workbook_all.add_worksheet()
# konfigurasi header pada excel (baris pertama)
worksheet_all.write(0, 0, 'No')
worksheet_all.write(0, 1, 'Confidence Level')
worksheet_all.write(0, 2, 'X1')
worksheet_all.write(0, 3, 'Y1')
worksheet_all.write(0, 4, 'X2')
worksheet_all.write(0, 5, 'Y2')
worksheet_all.write(0, 6, 'X')
worksheet_all.write(0, 7, 'Y')

# membuat file review all
workbook_review_all = xlsxwriter.Workbook('./entirety/review_all.xlsx')
worksheet_review_all = workbook_review_all.add_worksheet()
# konfigurasi header pada excel (baris pertama)
worksheet_review_all.write(0, 0, 'No')
worksheet_review_all.write(0, 1, 'Image')
worksheet_review_all.write(0, 2, 'No Building')
worksheet_review_all.write(0, 3, 'Confidence Level')
worksheet_review_all.write(0, 4, 'X1')
worksheet_review_all.write(0, 5, 'Y1')
worksheet_review_all.write(0, 6, 'X2')
worksheet_review_all.write(0, 7, 'Y2')
worksheet_review_all.write(0, 9, 'Image')
worksheet_review_all.write(0, 10, 'Buildings Detected')
worksheet_review_all.write(0, 11, 'Average Confidence Level')
row_excel_all = 1
sum_data_percentage_all = 0
average_percentage_all = 0

# looping sebanyak jumlah input citra = 77 citra
for j in range(1, 78):
    # konfigurasi output
    detections = detector.detectObjectsFromImage(display_object_name=False, display_percentage_probability=False, input_image="./input/image_" + str(j) + ".jpg", output_image_path="./output/output_" + str(j) + ".jpg")
    workbook = xlsxwriter.Workbook('./coordinate/output_' + str(j) + '.xlsx')
    worksheet = workbook.add_worksheet()
    # konfigurasi header pada excel (baris pertama)
    worksheet.write(0, 0, 'No')
    worksheet.write(0, 1, 'Confidence Level')
    worksheet.write(0, 2, 'X1')
    worksheet.write(0, 3, 'Y1')
    worksheet.write(0, 4, 'X2')
    worksheet.write(0, 5, 'Y2')
    # konfigurasi posisi output
    row_excel = 1
    sum_data_percentage = 0
    plotx_dot = []
    ploty_dot = []
    valuex_dot = 0
    valuey_dot = 0
    plotx_percent = []
    ploty_percent = []
    for i in range(100):
        plotx_percent.append(i)
        ploty_percent.append(0)

    # membaca ukuran gambar dalam pixel yang ditaruh ke variabel width dan height
    img = Image.open("./input/image_" + str(j) + ".jpg")
    width_image, height_image = img.size

    # posisi gambar dalam matriks 7 x 11
    # |1 8  15 22 29 36 43 50 57 64 71|
    # |2 9  16 23 30 37 44 51 58 65 72|
    # |3 10 17 24 31 38 45 52 59 66 73|
    # |4 11 18 25 32 39 46 53 60 67 74|
    # |5 12 19 26 33 40 47 54 61 68 75|
    # |6 13 20 27 34 41 48 55 62 69 76|
    # |7 14 21 28 35 42 49 56 63 70 77|

    # number image dalam (col,row)
    # (0,6)...................(10,6)
    # (0,5)...................(10,5)
    # (0,4)...................(10,4)
    # (0,3)...................(10,3)
    # (0,2)...................(10,2)
    # (0,1)...................(10,1)
    # (0,0)...................(10,0)

    # konfigurasi pergeseran input citra
    move_vertical = 0
    move_horizontal = 0
    if  number_image_row <= 0:
        number_image_row = 7
        number_image_col = number_image_col + 1 # perpindahan kolom setelah baris 7
    else:
        number_image_col = number_image_col
        
    number_image_row = number_image_row - 1

    # mengatur pergeseran ke samping sejauh posisi (dalam pixel)
    move_horizontal = number_image_col*width_image_last

    # pergeseran image terdeteksi bila berada di posisi paling bawah
    if number_image_row <= 0:
        move_vertical = 0
    else:
        move_vertical = height_image_last + (number_image_row-1)*height_image #mengatur pergeseran keatas sejauh posisi dia dalam pixel
    
    # looping sebanyak bangunan yang terdeteksi
    for detection in detections:
        valuex_dot = (detection["box_points"][0] + detection["box_points"][2]) / 2
        valuey_dot = ((detection["box_points"][1] + detection["box_points"][3]) / -2) + height_image
        plotx_dot.append(valuex_dot)
        ploty_dot.append(valuey_dot)

        valuex_dot_all = valuex_dot + move_horizontal
        valuey_dot_all = valuey_dot + move_vertical
        plotx_dot_all.append(valuex_dot_all)
        ploty_dot_all.append(valuey_dot_all)

        # plot semua koordinat pada plot_all.xlsx
        worksheet_all.write(row_excel_all, 0, row_excel_all)
        worksheet_all.write(row_excel_all, 1, int(detection["percentage_probability"]))
        worksheet_all.write(row_excel_all, 2, detection["box_points"][0] + move_horizontal)
        worksheet_all.write(row_excel_all, 3, detection["box_points"][1] + move_vertical)
        worksheet_all.write(row_excel_all, 4, detection["box_points"][2] + move_horizontal)
        worksheet_all.write(row_excel_all, 5, detection["box_points"][3] + move_vertical)
        worksheet_all.write(row_excel_all, 6, valuex_dot_all)
        worksheet_all.write(row_excel_all, 7, valuey_dot_all)

        # plot persentase
        ploty_percent[int(detection["percentage_probability"])] = ploty_percent[int(detection["percentage_probability"])] + 1

        # menjumlahkan data persentase semuanya (untuk mencari rata-rata)
        sum_data_percentage = sum_data_percentage + int(detection["percentage_probability"])

        # menulis pada excel output
        worksheet.write(row_excel, 0, row_excel)
        worksheet.write(row_excel, 1, int(detection["percentage_probability"]))
        worksheet.write(row_excel, 2, detection["box_points"][0])
        worksheet.write(row_excel, 3, detection["box_points"][1])
        worksheet.write(row_excel, 4, detection["box_points"][2])
        worksheet.write(row_excel, 5, detection["box_points"][3])

        # menjumlahkan seluruh data persentase  
        sum_data_percentage_all = sum_data_percentage_all + int(detection["percentage_probability"])

        # menulis di excel all
        worksheet_review_all.write(row_excel_all, 0, row_excel_all)
        worksheet_review_all.write(row_excel_all, 1, j)
        worksheet_review_all.write(row_excel_all, 2, row_excel)
        worksheet_review_all.write(row_excel_all, 3, int(detection["percentage_probability"]))
        worksheet_review_all.write(row_excel_all, 4, detection["box_points"][0])
        worksheet_review_all.write(row_excel_all, 5, detection["box_points"][1])
        worksheet_review_all.write(row_excel_all, 6, detection["box_points"][2])
        worksheet_review_all.write(row_excel_all, 7, detection["box_points"][3])
        
        # sebagai row tambahan di excel dan sebagai indikator jumlah data
        row_excel += 1 # untuk file satuan
        row_excel_all += 1 # untuk file all

    # mendapat nilai rata-rata persentase
    if sum_data_percentage != 0:
        average_percentage = sum_data_percentage / (row_excel-1)
    else:
        average_percentage = 0

    # menuliskan nilai rata-rata di output_x.xlsx 
    worksheet.write(row_excel + 1, 0, 'Average Percentage: ')
    worksheet.write(row_excel + 1, 1, average_percentage)
    workbook.close()  # menutup excel

    # menulis average data pada review_all.xlsx
    worksheet_review_all.write(j, 9, j)
    worksheet_review_all.write(j, 10, (row_excel-1))
    worksheet_review_all.write(j, 11, average_percentage)

    # input gambar awal dan akhir
    img_input = mpimg.imread("./input/image_" + str(j) + ".jpg")
    img_output = mpimg.imread("./output/output_" + str(j) + ".jpg")

    # membagi window menjadi 4 bagian, bentuk dalam array 2 dimensi, 00-(kiri-atas) 01-(kanan-atas) 10-(kiri-bawah) 11-(kanan-bawah)
    fig, loc = plt.subplots(2, 2)

    # memplot window 00
    loc[0, 0].imshow(img_input)  # plot gambar
    loc[0, 0].set_title('Citra Satelit')  # title

    # memplot window 01
    loc[0, 1].plot(plotx_dot, ploty_dot, 'ro', ms=0.5)
    loc[0, 1].set_title('Sebaran Bangunan')
    loc[0, 1].set_ylim(0, height_image)
    loc[0, 1].set_xlim(0, width_image)

    # memplot window 10
    loc[1, 0].imshow(img_output)
    loc[1, 0].set_title('Bangunan Terdeteksi')

    # memplot window 11
    loc[1, 1].plot(plotx_percent, ploty_percent)
    loc[1, 1].set_title('Tingkat Kepercayaan Deteksi Bangunan')

    # menampilkan tulisan jumlah dan rata-rata dibawah window 10
    jumlah = 'Jumlah bangunan terdeteksi: ' + str(row_excel-1) + ' bangunan'
    average = 'Rerata tingkat kepercayaan deteksi bangunan: ' + str("%.2f" % average_percentage) + ' %'
    loc[1, 0].text(0, 1300, jumlah)  # 0,1300 adalah posisi dalam pixel
    loc[1, 0].text(0, 1350, average)
    
    # save window
    fig.set_size_inches((11, 11), forward=False)
    fig.savefig('./summary/output_' + str(j) + '.png', dpi=(300))
    plt.close()

average_percentage_all = sum_data_percentage_all / (row_excel_all - 1)
worksheet_review_all.write(j+1, 9, 'Summary')
worksheet_review_all.write(j+1, 10, (row_excel_all-1))
worksheet_review_all.write(j+1, 11, average_percentage_all)
workbook_review_all.close()  # menutup excel

worksheet_all.write(row_excel_all + 1, 0, 'Average Percentage: ')
worksheet_all.write(row_excel_all + 1, 1, average_percentage_all)
workbook_all.close()  # menutup excel

#membuat window baru untuk plot semua koordinat
fig_all, loc_all = plt.subplots()
loc_all.plot(plotx_dot_all, ploty_dot_all, 'ro', ms=0.5)
loc_all.set_title('Sebaran Bangunan Terdeteksi')
loc_all.set_ylim(0, 6500)
loc_all.set_xlim(0, 10900)
fig_all.savefig('./entirety/plot_all.png', dpi=(300))

```

Skrip kode diatas ditulis untuk mendeteksi bangunan dengan input berupa 77 citra satelit pada area studi dengan masing-masing ukuran input 1200x1200 piksel. 77 citra satelit input terlihat pada gambar berikut.
<img src="https://github.com/ikbalrahadian/deteksi-objek/blob/master/sc8.png" width="900">
dengan penamaan dan urutan posisi file input sebagai berikut.
```
image_1 image_8  image_15 image_22 image_29 image_36 image_43 image_50 image_57 image_64 image_71
image_2 image_9  image_16 image_23 image_30 image_37 image_44 image_51 image_58 image_65 image_72
image_3 image_10 image_17 image_24 image_31 image_38 image_45 image_52 image_59 image_66 image_73
image_4 image_11 image_18 image_25 image_32 image_39 image_46 image_53 image_60 image_67 image_74
image_5 image_12 image_19 image_26 image_33 image_40 image_47 image_54 image_61 image_68 image_75
image_6 image_13 image_20 image_27 image_34 image_41 image_48 image_55 image_62 image_69 image_76
image_7 image_14 image_21 image_28 image_35 image_42 image_49 image_56 image_63 image_70 image_77
```

Untuk menggunakan skrip kode deteksi bangunan diatas, ganti "masukkan_nama_file_model_yang_akan_digunakan_disini" pada bagian # konfigurasi deteksi objek dengan nama file dari model yang dihasilkan pada proses training (misal: detection_model-ex-062--loss-0033.151.h5).

## Deteksi Bangunan
<div id="deteksibangunan"></div>

Deteksi bangunan dilakukan dengan menjalankan skrip kode deteksi bangunan yang telah dibuat sebelumnya (prediction.py). Proses ini akan mendeteksi bangunan yang ada pada masing-masing citra satelit input. Deteksi bangunan dilakukan dengan membuat struktur folder dan file pada google drive terlebih dahulu sebagai berikut.
```
       >> deteksi_bangunan--->> input------->> image_1.jpg
             |                                 image_2.jpg
             |                                 ...
             |                                 image_77.jpg
             |
              --------------->> detection_config.json (dari jolder json)
             |
              --------------->> file_model_yang_digunakan.h5 (dari folder models)
             |
              --------------->> prediction.py (skrip kode deteksi bangunan)
```

Setelah struktur folder dan file pada google drive telah dibuat, kemudian deteksi bangunan pada input citra satelit dilakukan dengan menjalankan skrip kode prediction.py menggunakan google colab dengan langkah sebagai berikut.
1. 
