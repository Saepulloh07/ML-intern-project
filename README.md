# Building a Machine Learning Model : Fiber Optic Monitoring
## Dokumentasi ini menjelaskan langkah-langkah untuk membangun model machine learning untuk monitoring kabel fiber optic. Model ini digunakan untuk mendeteksi redaman sinyal serta memprediksi kerusakan kabel optik

### Step 1: Data Collection

Data yang digunakan merupakan kumpulan dataset pengukuran kabel optic menggunakan OTDS yang diperoleh dari : https://www.kaggle.com/datasets/johannesreber/otdr-trace-training-data/data

Perbandingan Data General dengan Data KeyEvent:

Data 1
![image](https://github.com/user-attachments/assets/93d66e6f-278d-4cb3-9f46-4362e7f3fe8f)

Data 2
![image](https://github.com/user-attachments/assets/c9cbfd32-3026-4666-a7e6-ab5118157617)

Data 3
![image](https://github.com/user-attachments/assets/8b11dd0c-1e10-4b30-b721-40dd1cfe9789)

Terjadi perubahan yang tidak begitu signifikan pada setiap Event

###Pada dataset :2022_06_01_1625_275ns_60sec_no_3.sor

Menampilkan visualisasi Tracedata sebagai berikut:
![image](https://github.com/user-attachments/assets/7df6fb1c-42db-47ee-b2de-6faf1c18f097)

Keterangan:

Mapping Label Encoding pada kolom 'type':
0E9999LS {auto} loss/drop/gain: 0
0F9999LS {auto} loss/drop/gain: 1
1F9999LS {auto} reflection: 2

| event | distance | splice_loss | refl_loss | slope | Pulse Width | type |
|-------|----------|-------------|-----------|-------|-------------|------|
| 1     | 0.000    | 0.000       | -60.821   | 0.000 | 275.0       | 2    |
| 2     | 10.987   | 0.111       | 0.000     | 0.196 | 275.0       | 1    |
| 3     | 12.005   | 0.000       | 0.000     | 0.191 | 275.0       | 0    |
| 4     | 13.529   | 0.000       | -61.644   | 0.000 | 275.0       | 2    |
| 5     | 14.025   | 0.000       | -61.216   | 0.000 | 275.0       | 2    |
| 6     | 15.028   | 0.000       | -56.456   | 0.000 | 275.0       | 2    |
| 7     | 16.022   | 0.000       | -58.567   | 0.000 | 275.0       | 2    |
| 8     | 16.088   | 0.000       | -66.083   | 0.000 | 275.0       | 2    |
| 9     | 17.016   | 0.000       | -55.054   | 0.000 | 275.0       | 2    |
| 10    | 18.013   | 0.000       | -51.892   | 0.000 | 275.0       | 2    |

Maka diperoleh tracedata sebagai berikut:
![image](https://github.com/user-attachments/assets/557bd33c-3d9b-4e89-ae64-2eb897432c10)
