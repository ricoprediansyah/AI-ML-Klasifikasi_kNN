# ML-2 Classification using k-NN

# STUDI KASUS: Customer Churn
# Terdapat dataset konsumen dengan 2 kategori churn, yaitu:
# pelanggan aktif (0) dan
# pelanggan yang sudah berhenti (1)

#import library
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

#Load Dataset
df = pd.read_csv('customer_churn_dataset.csv', index_col=['customer_id'])

# Data Preprocessing
# Library ML umumnya hanya bisa memproses data numerik. Karena hal itu, kita perlu mengubah kolom bernilai kategorik menjadi nilai numerik.

# Pada kolom product nilai atribut:
# Kartu A didefinisikan sebagai 0
# Kartu B didefinisikan sebagai 1, dan
# Kartu C didefinisikan sebagai 2

df['product'] = df['product'].map({'Kartu A': 0,'Kartu B': 1, 'Kartu C':2})
df.head()

# Split Feature (X) dan Label (Y)
# Selanjutnya, pisahkan X dan Y dari dataset. Ubah DataFrame menjadi numpy array dengan method .values.

# atribut predictor/feature
X = df.iloc[:, :-1].values
#atribut targer/label
Y = df.iloc[:, -1].values

# Normalisasi Data
# K-NN merupakan metode yang didasarkan pada tingkat kedekatan antar-atribut.
# Oleh karena itu, sangat disarankan untuk melakukan normalisasi data agar perhitungan jarak menjadi lebih valid.

scaler = StandardScaler()
scaler = scaler.fit(X)
X = scaler.transform(X)

print(X) #sudah di normalisasikan

# Splitting Data
# Untuk menghasilkan model yang baik, perlu dilakukan pengujian yang baik
# Aturan umum yang berlaku adalah data training dan testing harus dipisahkan terlebih dahulu. Dengan kata lain tidak boleh menggunakan data yang sama untuk training dan testing.
# Kali ini kita membagi dataset dengan komposisi sebesar 80% data training dan 20% data testing.
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape, y_train.shape)
print ('Train set:', X_test.shape, y_test.shape)

# Modeling: Klasifikasi k-Nearest Neighbor (k-NN)
# Training Model
# Pertama, tentukan nilai k. Kita akan coba dengan nikai k=7:

k = 7
# Train Model
model_knn = KNeighborsClassifier(n_neighbors = k)
model_knn.fit(X_train, y_train)

# Predicting
# Kita sudah dapat menggunakan model yang telah di-training untuk memprediksikan data
y_pred = model_knn.predict(X_test)
print(y_pred)

# Perhitungan Akurasi
# Pada klasifikasi, nilai akurasi klasifikasi secara sederhana dapat dihitung dengan:
# membandingkan kelas hasil prediksi model dengan kelas sebenarnya/actual class.
print("Train set Accuracy :", metrics.accuracy_score(y_train, model_knn.predict(X_train)))
print("Train set Accuracy :", metrics.accuracy_score(y_test, y_pred))

# Accuracy train set dan test set tidak terpaut jauh. Accuracy keduanya juga sangat tinggi.
# Bisa dikatakan model dengan k=7 sudah Good Fit.

# Note: Pada praktiknya kita harus memilih metrics yang cocok untuk kasus customer churn, apakah accuracy, precision, atau recall. Supaya lebih mudah, kita hanya memilih metric accuracy.
# Klasifikasi Data Baru
# Model yang sudah good fit bisa kita gunakan untuk memprediksi data baru.

product = int(input('Input Product: '))
reload_1 = float(input('Input Reload 1: '))
reload_2 = float(input('Input Reload 2: '))
video = float(input('Input Video: '))
music = float(input('Input Music: '))
games = float(input('Input Games: '))
chat_1 = float(input('Input Chat 1: '))
chat_2 = float(input('Input Chat 2: '))
socmed_1 = float(input('Input Socmed 1: '))
socmed_2 = float(input('Input Socmed 2: '))
internet = float(input('Input Internet: '))
days_active = int(input('Input Days Active: '))
tenure = float(input('Input Tenure: '))

new_data = [[product, reload_1, reload_2, video, music, games, chat_1, chat_2, socmed_1, socmed_2, internet, days_active, tenure]]

#prediksi data baru
hasil_prediksi = model_knn.predict(new_data)

#cetak hasil prediksii
if hasil_prediksi == 0:
  print('\n Customer diprediksi tidak berhenti berlangganan (tetap aktif)')
elif hasil_prediksi == 1:
  print('\n Customer diprediksi berhenti berlangganan')


# MENCARI NILAI K TERBAIK
# Performa dari model K-NN sangat bergantung dengan nilai K.
# Lalu bagaimana memilih nilai K yang terbaik? Jawabnya, kita harus lakukan pengujian.
# Kode berikut ini bertujuan untuk melakukan pengujian nilai k dari k=1 sampai dengan k=10.
Ks = 11
accuracies = []
precisions = []
recalls = []

#loops dari k=1 sampai dengan k=10
for k in range (1, Ks):
  knn = KNeighborsClassifier(n_neighbors = k)
  
  #training model
  knn.fit(X_train, y_train)

  #membuat prediksi dari testing data
  y_hat = knn.predict(X_test)

  #menyimpan hasil pengujian
  accuracies.append(metrics.accuracy_score(y_test, y_hat))
  precisions.append(metrics.precision_score(y_test, y_hat))
  recalls.append(metrics.recall_score(y_test, y_hat))

  #cetak hasil pengujian
  print('k =', k)
  print('Accuracy =', metrics.accuracy_score(y_test, y_hat))
  print('Precision =', metrics.precision_score(y_test, y_hat))
  print('Recall =', metrics.recall_score(y_test, y_hat), '\n')

# Kita juga bisa menampilkan grafik / plot-nya untuk mempermudah pembacaan hasil pengujian
# line chart
plt.plot(range(1,Ks), accuracies, 'r')
plt.plot(range(1,Ks), precisions, 'g')
plt.plot(range(1,Ks), recalls, 'b')
plt.legend(('Accuracy', 'Precision', 'Recall'))
plt.ylabel('Score')
plt.xlabel('Number of Neighbor (k)')
plt.tight_layout()
plt.show()

max_accuracy = max(accuracies)
max_precision = max(precisions)
max_recall = max(recalls)

print("Akurasi terbaik adalah   ", max_accuracy, "dengan nilai k=", accuracies.index(max_accuracy)+1)
print("Precision terbaik adalah ", max_precision, "dengan nilai k=", precisions.index(max_precision)+1)
print("Recall terbaik adalah    ", max_recall, "dengan niali k=", recalls.index(max_recall)+1)

# Kesimpulan
# Supaya lebih mudah, kita akan pilih model terbaik berdasarkan metric accuracy.
# Berdasarkan metric accuracy, model terbaik adalah model dengan nilai k=4.
# Setelah diketahui nilai k terbaik, kita perlu melatih ulang model menggunakan nilai k tersebut