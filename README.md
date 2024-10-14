# Building a Machine Learning Model : Fiber Optic Monitoring
## Dokumentasi ini menjelaskan langkah-langkah untuk membangun model machine learning untuk monitoring kabel fiber optic. Model ini digunakan untuk mendeteksi redaman sinyal serta memprediksi kerusakan kabel optik

### Step 1: Exploratory Data Analisys

Data yang digunakan merupakan kumpulan dataset pengukuran kabel optic menggunakan OTDS yang diperoleh dari  pengukuran 

Perbandingan Data General dengan Data KeyEvent:

Data 1
![image](https://github.com/user-attachments/assets/93d66e6f-278d-4cb3-9f46-4362e7f3fe8f)

Data 2
![image](https://github.com/user-attachments/assets/c9cbfd32-3026-4666-a7e6-ab5118157617)

Data 3
![image](https://github.com/user-attachments/assets/8b11dd0c-1e10-4b30-b721-40dd1cfe9789)

Terjadi perubahan yang tidak begitu signifikan pada setiap Event

### Pada dataset : RFTS Port 05.sor

Menampilkan visualisasi Tracedata sebagai berikut:
![image](https://github.com/user-attachments/assets/f009274a-03a0-44d7-839d-f15aabde7948)

Keterangan:

Mapping Label Encoding pada kolom 'type':
* 0F9999LS {auto} loss/drop/gain: 0
* 1E9999LS {auto} reflection: 1
* 1F99992P [unknown type 1F99992P]: 2
* 1F9999LS {auto} reflection: 3

| event | distance | splice_loss | refl_loss | Attenuation_slope | Pulse Width | Fiber Length (km) | Wavelength (nm) | Noise Floor | Averaging Time (sec) | type |
|-------|----------|-------------|-----------|-------------------|-------------|-------------------|-----------------|-------------|----------------------|------|
| 1     | 0.000    | 1.120       | -59.70    | 0.000             | 10.0        | 81.440095         | 1550.0          | 55000       | 20.0                 | 2    |
| 2     | 0.027    | 0.000       | -63.25    | 0.000             | 10.0        | 81.440095         | 1550.0          | 55000       | 20.0                 | 2    |
| 3     | 1.374    | 0.400       | 0.00      | 0.199             | 10.0        | 81.440095         | 1550.0          | 55000       | 20.0                 | 0    |
| 4     | 1.963    | 0.195       | 0.00      | 0.202             | 10.0        | 81.440095         | 1550.0          | 55000       | 20.0                 | 0    |
| 5     | 2.989    | 0.044       | 0.00      | 0.242             | 10.0        | 81.440095         | 1550.0          | 55000       | 20.0                 | 0    |
| 6     | 5.049    | -0.037      | 0.00      | 0.246             | 10.0        | 81.440095         | 1550.0          | 55000       | 20.0                 | 0    |
| 7     | 5.641    | 0.101       | 0.00      | 0.255             | 10.0        | 81.440095         | 1550.0          | 55000       | 20.0                 | 0    |
| 8     | 12.933   | 0.250       | -49.34    | 0.259             | 10.0        | 81.440095         | 1550.0          | 55000       | 20.0                 | 3    |
| 9     | 15.638   | 0.220       | 0.00      | 0.188             | 10.0        | 81.440095         | 1550.0          | 55000       | 20.0                 | 0    |
| 10    | 17.751   | 0.798       | -37.07    | 0.196             | 10.0        | 81.440095         | 1550.0          | 55000       | 20.0                 | 3    |
| 11    | 18.489   | 0.176       | 0.00      | 0.117             | 10.0        | 81.440095         | 1550.0          | 55000       | 20.0                 | 0    |
| 12    | 20.381   | 0.103       | 0.00      | 0.235             | 10.0        | 81.440095         | 1550.0          | 55000       | 20.0                 | 0    |
| 13    | 22.039   | 0.125       | 0.00      | 0.282             | 10.0        | 81.440095         | 1550.0          | 55000       | 20.0                 | 0    |
| 14    | 23.786   | 0.401       | 0.00      | 0.327             | 10.0        | 81.440095         | 1550.0          | 55000       | 20.0                 | 0    |
| 15    | 25.126   | 0.341       | 0.00      | 0.186             | 10.0        | 81.440095         | 1550.0          | 55000       | 20.0                 | 0    |
| 16    | 25.791   | 0.116       | 0.00      | 0.166             | 10.0        | 81.440095         | 1550.0          | 55000       | 20.0                 | 0    |
| 17    | 27.419   | 0.000       | -17.51    | 0.241             | 10.0        | 81.440095         | 1550.0          | 55000       | 20.0                 | 1    |

Diambil data terakhir pada Event untuk menspesifikan data grafik, maka diperoleh:
![image](https://github.com/user-attachments/assets/1f9fe218-b812-4faa-bab7-d41d50c5f711)

Maka diperoleh tracedata sebagai berikut:
![image](https://github.com/user-attachments/assets/9cdfecc0-aaff-491d-a88b-337230a505fb)

Menambahkan titik perubahan intensitas sinyal sebagai berikut:
![image](https://github.com/user-attachments/assets/20cd76d7-cabc-45da-aac1-636fc8eda1bb)

Diperoleh data sebagai berikut:
![image](https://github.com/user-attachments/assets/00bda3e3-a795-444c-98f6-e8693c0c17c9)

Keterangan:

* Terdapat penurunan bertahap dalam intensitas sinyal, seperti terlihat dari titik awal di sekitar 27 dB hingga sekitar 17 dB pada akhir jalur (sekitar 25 km).
* Kehilangan sambungan dan pantulan terlihat jelas pada beberapa titik, misalnya di sekitar 2 km dan 15 km, yang dapat menjadi indikasi perbaikan atau pemeliharaan yang dibutuhkan pada sambungan tersebut.

Grafik ini menunjukkan distribusi sinyal dan kehilangan sepanjang serat optik. Secara keseluruhan, terdapat beberapa titik sambungan dan pantulan yang menimbulkan penurunan sinyal.



