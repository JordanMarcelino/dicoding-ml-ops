# Submission 1: ML Pipeline - Social Media Articles Stress Detection

Nama: Jordan Marcelino

Username dicoding: jordanmarz

![Stress](https://storage.googleapis.com/kaggle-datasets-images/4641718/7903032/0c0c023182219681b65e2774a47682d5/dataset-cover.jpg?t=2024-03-21-08-29-49)

|                         | Deskripsi                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Dataset                 | [Stress Detection from Social Media Articles](https://www.kaggle.com/datasets/mexwell/stress-detection-from-social-media-articles/data)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| Masalah                 | Mental health adalah istilah yang sudah tidak asing lagi untuk didengar di era modern ini. Istilah ini sering digunakan oleh kaum muda untuk membahas kondisi psikologis & emosional seseorang. Banyak hal yang dapat memengaruhi kondisi mental health seseorang, media sosial adalah salah satunya. Maraknya penggunaan media sosial saat ini menyebabkan banyaknya orang menaruh jati dirinya pada hal tersebut, sehingga kondisi mental health seseorang akan sangat bergantung pada media sosial. Pesan atau komen yang diberikan oleh seseorang pada media sosial dapat secara tidak langsung mencerminkan kondisi mental health orang tersebut. Untuk dapat mengetahui kondisi mental health seseorang melalui pesan atau komen yang disampaikan memerlukan pengetahuan yang lebih dibidang psikologis dan bahasa atau memerlukan expert dibidangnya. |
| Solusi machine learning | Model machine learning dapat membantu menyelesaikan permasalahan yang diangkat dengan mengidentifikasi pola-pola dalam teks yang berkaitan dengan kondisi stres. Ini bisa termasuk pola kata-kata atau frasa tertentu yang sering muncul dalam konteks yang berkaitan dengan stres, ekspresi emosional yang khas, atau bahkan struktur kalimat yang mengindikasikan kecemasan atau tekanan. Model klasifikasi dapat memproses teks yang baru dan secara otomatis mengklasifikasikannya sebagai positif (tidak stres) atau negatif (stres). Ini akan membantu dalam mengidentifikasi konten-konten yang mungkin mengindikasikan adanya stres di antara berbagai artikel media sosial.                                                                                                                                                                         |
| Metode pengolahan       | Dataset terdiri dari 4 kolom, tetapi yang digunakan pada proyek ini hanya fitur Body_Title dan label, selain itu akan dihapus. Kemudian dataset dibagi menjadi data training dan evaluation dengan rasio 80%:20%, dan dilakukan transformasi fitur menjadi lowercase serta mengubah label menjadi integer                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| Arsitektur model        | Arsitektur model yang digunakan terdiri dari layer Embedding, Bidirectional LSTM, Fully Connected Layer, dan Output Layer dengan aktivasi sigmoid. Optimizer yang digunakan adalah Adam dengan loss function BinaryCrossentropy                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| Metrik evaluasi         | Metrik evaluasi yang digunakan yaitu ExampleCount, AUC, FalsePositives, TruePositives, FalseNegatives, TrueNegatives, dan BinaryAccuracy                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| Performa model          | Berdasarkan nilai metrik Binary Accuracy, model menghasilkan nilai 93% pada training set dan 90% pada evaluation set. Hasil tersebut sudah menunjukkan bahwa model memiliki performa yang baik                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |