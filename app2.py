#!/usr/bin/env python
# coding: utf-8

# In[3]:

from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import pandas as pd
import numpy as np
import string 
import re #regex library

# import word_tokenize & FreqDist from NLTK
from nltk.tokenize import word_tokenize 
from nltk.probability import FreqDist
from nltk.corpus import stopwords
# !pip3 install swifter
# !pip3 install PySastrawi

from collections import Counter

import ast
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score, precision_score, f1_score, recall_score, confusion_matrix
# from svmutil import svm_problem, svm_parameter, svm_train, svm_predict
# from libsvm.svmutil import *
from sklearn.svm import SVC
import logging

# import Sastrawi package
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import swifter

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.DEBUG)
model = None  # Global model
scaler = None

def initialize_model():
    # Initialize with polynomial kernel
    return SVC(kernel='poly', degree=3, coef0=1, C=1, gamma='scale')
    


@app.route('/sequential_svm/train', methods=['POST'])
def train_sequential_svm():
    global model, scaler
    # Membaca dataset
    df = pd.read_csv("pelabelan.csv", header=0)
    # df = pd.read_csv("hasil_vector_matrix.csv", header=0)
    
    # Memisahkan fitur dan label
    X = df.drop(columns=['Unnamed: 0', 'aktual', 'sentimen']).values
    y = df['sentimen'].values

    # Menstandarisasi fitur
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Menginisialisasi model
    model = initialize_model()

    # Melatih model
    model.fit(X, y)

    return "Pelatihan Sequential SVM berhasil"

from sklearn.feature_extraction.text import TfidfVectorizer


@app.route('/predict', methods=['POST'])
def predict():
    global model, scaler  # Pastikan scaler juga diakses secara global
    if model is None:
        logging.error("Model belum dilatih.")
        return "Model belum dilatih.", 400

    # Menerima data dari request
    data = request.get_json()
    if not data or 'text' not in data:
        logging.error("Data tidak valid: Tidak ada teks dalam data.")
        return "Data tidak valid", 400

    try:
        # Preprocessing teks input
        processed_text = preprocess_text(data['text'])
        logging.debug(f"Teks diproses: {processed_text}")

        # Mengubah teks menjadi format yang sesuai untuk model menggunakan TF-IDF
        vectorized_text = tfidf_predict(processed_text)
        logging.debug(f"Teks tervektorisasi: {vectorized_text}")

        # Memastikan bahwa vektor fitur tidak melebihi 100 fitur
        # if vectorized_text.shape[0] > 100:
        #     vectorized_text = vectorized_text[:100]  # Memotong vektor jika lebih dari 100 fitur
        #     logging.debug("Vektor fitur dipotong menjadi 100 fitur.")
        
        if len(vectorized_text) < scaler.mean_.shape[0]:
            vectorized_text = np.pad(vectorized_text, (0, scaler.mean_.shape[0] - len(vectorized_text)), 'constant')
            logging.debug(f"Vektor fitur dipanjangkan menjadi {len(vectorized_text)} fitur.")
        
        elif len(vectorized_text) > scaler.mean_.shape[0]:
            vectorized_text = vectorized_text[:scaler.mean_.shape[0]]
            logging.debug(f"Vektor fitur dipotong menjadi {len(vectorized_text)} fitur.")

        # Standarisasi fitur sebelum prediksi
        vectorized_text = scaler.transform([vectorized_text])  # Pastikan menggunakan transform bukan fit_transform
        logging.debug("Teks terstandarisasi {vectorized_text} .")

        # Melakukan prediksi menggunakan model yang sudah dilatih
        prediction = model.predict(vectorized_text)
        logging.debug(f"Prediksi: {prediction}")

        # Menghitung kernel dari data uji
        kernel_matrix = model.decision_function(vectorized_text)
        logging.debug(f"Matriks kernel data uji: {kernel_matrix}")

        # Mendapatkan nilai alpha (ai) dan support vectors (yi)
        dual_coef = model.dual_coef_
        support_vectors = model.support_vectors_
        logging.debug(f"Nilai alpha (ai): {dual_coef}")
        logging.debug(f"Support vectors (yi): {support_vectors}")

        # Mengonversi prediksi numerik ke label sentimen
        sentiment = 'Positif' if prediction[0] == 1 else 'Negatif'
        logging.info(f"Sentimen yang diprediksi: {sentiment}")

        return jsonify({"sentimen": sentiment, "kernel_matrix": kernel_matrix.tolist(), "alpha_values": dual_coef.tolist(), "support_vectors": support_vectors.tolist()})
    except Exception as e:
        logging.error(f"Kesalahan dalam pemrosesan prediksi: {str(e)}")
        if isinstance(e, TypeError) and "float() argument" in str(e):
            logging.error("Tipe data yang diberikan tidak sesuai, pastikan data yang diberikan adalah numerik.")
            return jsonify({"error": "Kesalahan dalam pemrosesan prediksi", "message": "Tipe data yang diberikan tidak sesuai, pastikan data yang diberikan adalah numerik."}), 500
        return jsonify({"error": "Kesalahan dalam pemrosesan prediksi", "message": str(e)}), 500


def preprocess_text(text):
    # Mengubah teks menjadi huruf kecil
    text = text.lower()

    # Menghapus karakter khusus, tautan, dan non-ASCII
    text = re.sub(r"\\t|\\n|\\u|\\", " ", text)
    text = text.encode('ascii', 'ignore').decode('ascii')
    text = re.sub(r"(@[A-Za-z0-9]+)|(\w+:\/\/\S+)", " ", text)
    text = text.replace("http://", " ").replace("https://", " ")

    # Menghapus angka
    text = re.sub(r"\d+", "", text)

    # Menghapus tanda baca
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Menghapus spasi berlebih
    text = re.sub(r'\s+', ' ', text).strip()

    # Menghapus karakter tunggal
    text = re.sub(r"\b[a-zA-Z]\b", "", text)

    # Tokenisasi
    tokens = word_tokenize(text)

    # Menghapus stopwords
    list_stopwords = set(stopwords.words('indonesian'))
    list_stopwords.update(["yg", "dg", "rt", "dgn", "ny", "d", 'klo', 
                           'kalo', 'amp', 'biar', 'bikin', 'bilang', 
                           'gak', 'ga', 'krn', 'nya', 'nih', 'sih', 
                           'si', 'tau', 'tdk', 'tuh', 'utk', 'ya', 
                           'jd', 'jgn', 'sdh', 'aja', 'n', 't', 
                           'nyg', 'hehe', 'pen', 'u', 'nan', 'loh', 'rt',
                           '&amp', 'yah'])
    tokens = [word for word in tokens if word not in list_stopwords]

    # Stemming
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    stemmed_tokens = [stemmer.stem(word) for word in tokens]

    # Menggabungkan token kembali menjadi string
    return ' '.join(stemmed_tokens)

# # Contoh penggunaan fungsi
# text_input = "Ini adalah contoh teks untuk testing preprocessing!"
# processed_text = preprocess_text(text_input)
# print(processed_text)

def tfidf_predict(processed_text):
    # Konversi teks yang diproses menjadi list token
    input_tokens = word_tokenize(processed_text)

    # Load data untuk mendapatkan DF dan IDF yang sudah ada
    TWEET_DATA = pd.read_csv("Text_Preprocessing.csv", usecols=["tweet_tokens_stemmed"])
    TWEET_DATA["tweet_list"] = TWEET_DATA["tweet_tokens_stemmed"].apply(ast.literal_eval)

    # Menghitung DF dari data yang ada
    def calc_DF(tfDict):
        count_DF = Counter()
        for document in tfDict:
            count_DF.update(document)
        return count_DF

    DF = calc_DF(TWEET_DATA["tweet_list"].apply(Counter))

    n_document = len(TWEET_DATA) + 1  # Termasuk dokumen input
    max_proses = 100  # Batas maksimum proses
    n_document = min(n_document, max_proses)  # Batasi jumlah dokumen yang diproses

    # Menghitung IDF
    def calc_IDF(__n_document, __DF):
        IDF_Dict = {}
        for term in __DF:
            IDF_Dict[term] = np.log(__n_document / (__DF[term] + 1))
        return IDF_Dict

    IDF = calc_IDF(n_document, DF)

    # Menghitung TF untuk input
    TF_dict = Counter(input_tokens)
    for term in TF_dict:
        TF_dict[term] = TF_dict[term] / len(input_tokens)

    # Menghitung TF-IDF untuk input
    def calc_TF_IDF(TF, IDF):
        TF_IDF_Dict = {}
        for term in IDF:
            TF_IDF_Dict[term] = TF.get(term, 0) * IDF[term]
        return TF_IDF_Dict

    TF_IDF_dict = calc_TF_IDF(TF_dict, IDF)

    # Mengonversi TF-IDF dict ke vektor berdasarkan urutan DF
    TF_IDF_vector = np.array([TF_IDF_dict.get(term, 0) for term in sorted(DF.keys())])
    return TF_IDF_vector

@app.route('/sequential_svm/predict', methods=['GET'])
@cross_origin()
def predict_sequential_svm():
    global model
    try:
        # Mengambil data uji dari parameter GET
        X_test_string = request.args.get('X_test')
        # Mengubah string menjadi array numpy
        X_test = np.array(ast.literal_eval(X_test_string))
        # Melakukan prediksi
        y_pred = model.predict(X_test)
        # Mengubah label numerik kembali ke bentuk sentimen
        sentiment_labels = ['Negatif' if label == 0 else 'Positif' for label in y_pred]
        return jsonify(predictions=sentiment_labels)
    except Exception as e:
        logging.error("Kesalahan dalam memproses permintaan: %s", e)
        return jsonify({"error": "Kesalahan dalam memproses permintaan", "message": str(e)}), 500

@app.route('/sequential_svm/evaluate', methods=['POST'])
def evaluate_sequential_svm():
    global model
    print("Initializing model...")
    model = initialize_model()
    if model is None:
        print("Failed to initialize model!")
    else:
        print("Model successfully initialized.")

    df = pd.read_csv("pelabelan.csv", header=0)
    X = df.drop(columns=['0', 'aktual', 'sentimen']).values
    y = df['sentimen'].values

    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

    # Train and predict
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Calculate metrics
    cm = confusion_matrix(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    accuracy = np.mean(y_pred == y_test)

    return jsonify(accuracy=accuracy, confusion_matrix=cm.tolist(),
                   classification_report=classification_rep,
                   precision=precision, recall=recall, f1_score=f1)

# def initialize_model():
#     # Membuat instance dari SVC dengan kernel linear
#     return SVC(kernel='linear')


# # Route untuk training sequential SVM
# @app.route('/sequential_svm/train', methods=['POST'])
# def train_sequential_svm():
#     global model
#     # Membaca dataset
#     df = pd.read_csv("hasil_vector_matrix.csv", header=0)

#     # Pisahkan fitur dan label
#     X = df.drop(columns=['tweet_id', 'sentimen']).values
#     y = df['sentimen'].values

#     # Inisialisasi model
#     model = initialize_model()

#     # Pembelajaran bertahap dengan polynomial kernel
#     for i in range(len(X)):
#         # Update model dengan batch data yang baru
#         # Example: -t 1 for polynomial, -d 3 for degree 3, -r 1 for coef0 of 1, -g 1 for gamma 1
#         model = svm_train(y[i:i+1].tolist(), X[i:i+1].tolist(), '-t 1 -d 3 -r 1 -g 1 -c 1 -w+')

#     return "Sequential SVM trained successfully"


# # Route untuk prediksi dengan sequential SVM
# @app.route('/sequential_svm/predict', methods=['POST'])
# def predict_sequential_svm():
#     # Ambil data uji dari request
#     data = request.json
#     X_test = np.array(data['X_test'])

#     # Melakukan prediksi dengan model yang telah dilatih
#     y_pred, _, _ = svm_predict([0] * len(X_test), X_test.tolist(), model)

#     return jsonify(predictions=y_pred.tolist())

# # Route untuk evaluasi model
# @app.route('/sequential_svm/evaluate', methods=['POST'])
# # @app.route('/sequential_svm/evaluate', methods=['POST'])
# def evaluate_sequential_svm():
#     global model
#     print("Menginisialisasi model...")
#     model = initialize_model()
#     if model is None:
#         print("Gagal menginisialisasi model!")
#     else:
#         print("Model diinisialisasi dengan sukses.")
#     # Membaca dataset
#     df = pd.read_csv("hasil_vector_matrix.csv", header=0)

#     # Pisahkan fitur dan label
#     X = df.drop(columns=['tweet_id', 'sentimen']).values
#     y = df['sentimen'].values

#     # Split data menjadi data latih dan data uji
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#     # Melakukan prediksi dengan model yang telah dilatih
#     y_pred, _, _ = svm_predict(y_test.tolist(), X_test.tolist(), model)

#     # Menghitung confusion matrix
#     cm = confusion_matrix(y_test, y_pred)

#     # Menghitung classification report, precision, recall, dan F1-score
#     classification_rep = classification_report(y_test, y_pred)
#     precision = precision_score(y_test, y_pred)
#     recall = recall_score(y_test, y_pred)
#     f1 = f1_score(y_test, y_pred)

#     # Menghitung akurasi
#     accuracy = np.mean(np.array(y_pred) == np.array(y_test))

#     return jsonify(accuracy=accuracy, confusion_matrix=cm.tolist(),
#                    classification_report=classification_rep,
#                    precision=precision, recall=recall, f1_score=f1)



#API SMO
# @app.route('/svm')
# def SVM():
#     # Membaca dataset
#     df = pd.read_csv("hasil_vector_matrix.csv", header=0)

#     # Menggunakan LabelEncoder untuk mengonversi label menjadi numerik
#     label_encoder = LabelEncoder()
#     df['sentimen'] = label_encoder.fit_transform(df['sentimen'])

#     # Pisahkan fitur dan label
#     X = df.drop(columns=['tweet_id', 'sentimen'])
#     y = df['sentimen']

#     # Split data menjadi data latih dan data uji
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
#     y_train_list = y_train.tolist()
#     y_test_list = y_test.tolist()


#     # Membuat struktur data SVM
#     prob = svm_problem(y_train_list, X_train.values.tolist())

#     # Mendefinisikan parameter SVM
#     param = svm_parameter('-s 0 -t 0 -c 1')  # -s 0 untuk SVC, -t 0 untuk kernel linear, -c adalah parameter penalti

#     # Melatih model SVM
#     model = svm_train(prob, param)

#     # Melakukan prediksi pada data uji
#     y_pred, _, _ = svm_predict(y_test_list, X_test.values.tolist(), model)

#     # Menghitung confusion matrix
#     cm = confusion_matrix(y_test, y_pred)

#     # Menghitung classification report, precision, recall, dan F1-score
#     classification_rep = classification_report(y_test, y_pred)
#     precision = precision_score(y_test, y_pred)
#     recall = recall_score(y_test, y_pred)
#     f1 = f1_score(y_test, y_pred)

#     # Menghitung akurasi
#     accuracy = np.mean(np.array(y_pred) == np.array(y_test))

#     # Mendapatkan nilai TN, FP, FN, TP
#     tn, fp, fn, tp = cm.ravel()

#     # Mengonversi nilai int64 menjadi int
#     tn = int(tn)
#     fp = int(fp)
#     fn = int(fn)
#     tp = int(tp)

#     return jsonify(accuracy=accuracy, confusion_matrix=cm.tolist(),
#                    tn=tn, fp=fp, fn=fn, tp=tp,
#                    classification_report=classification_rep,
#                    precision=precision, recall=recall, f1_score=f1)


# #API SVM
@app.route('/svm')
def SVM():
    try:
        df = pd.read_csv("pelabelan.csv", header=0)

        label_encoder = LabelEncoder()
        df['sentimen'] = label_encoder.fit_transform(df['sentimen'])

        # Pisahkan fitur dan label
        X = df.drop(columns=['Unnamed: 0', 'aktual', 'sentimen'])
        y = df['sentimen']
        
        XNum = X.astype(np.float64)
        yNum = y.astype(np.float64)

        # Memeriksa apakah terdapat lebih dari satu kelas dalam data
        if len(np.unique(yNum)) <= 1:
            return "Error: Jumlah kelas harus lebih dari satu; hanya mendapatkan 1 kelas"

        # Split data menjadi data latih dan data uji
        X_train, X_test, y_train, y_test = train_test_split(XNum, yNum, test_size=0.4, random_state=42)

        # Inisialisasi model SVM
        svm_model = SVC(kernel='poly')
        
        # Mendapatkan nilai gamma, lambda, dan complexity dari request jika tersedia
        gamma = float(request.args.get('gamma', 'scale'))
        C = float(request.args.get('lambda', 1.0))
        coef0 = float(request.args.get('complexity', 0.0))

        # Mengatur parameter model SVM
        svm_model.set_params(gamma=gamma, C=C, coef0=coef0)

        # Melatih model SVM
        svm_model.fit(X_train, y_train)
        
        # Melakukan prediksi pada data uji
        y_pred = svm_model.predict(X_test)
        
        labels = np.unique(y_train)  # Ensure all possible labels are included
        cm = confusion_matrix(y_test, y_pred, labels=labels)
        
        # Mendapatkan nilai TN, FP, FN, TP
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            tn, fp, fn, tp = 0, 0, 0, 0  # Handle cases where not all labels are present
    
        # # Menghitung confusion matrix
        # cm = confusion_matrix(y_test, y_pred)
        
        # # Mendapatkan nilai TN, FP, FN, TP
        # tn, fp, fn, tp = cm.ravel() 
        
        # Menghitung classification report, AUC, precision, recall, dan F1-score
        classification_rep = classification_report(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        
        # Mengonversi nilai int64 menjadi int
        tn = int(tn)
        fp = int(fp)
        fn = int(fn)
        tp = int(tp)
        
        # Menghitung akurasi
        accuracy = np.mean(y_pred == y_test)

        # Logging
        app.logger.info("Model SVM telah dilatih dan diuji.")
        app.logger.info(f"Confusion Matrix: {cm.tolist()}")
        app.logger.info(f"TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")
        app.logger.info(f"Classification Report: {classification_rep}")
        app.logger.info(f"Precision: {precision}, Recall: {recall}, F1 Score: {f1}")

        return jsonify(accuracy=accuracy, confusion_matrix=cm.tolist(),
                    tn=tn, fp=fp, fn=fn, tp=tp,
                    classification_report=classification_rep,
                    precision=precision, f1_score=f1, recall=recall)
    except Exception as e:
        app.logger.error(f"Kesalahan dalam memproses permintaan SVM: {str(e)}")
        return jsonify({"error": f"Kesalahan dalam memproses permintaan SVM: {str(e)}"}), 500
# API KNN
@app.route('/knn')
def KNN():
    df = pd.read_csv("hasil_vector_matrix.csv", header=None)

    X = df.iloc[1:, :-1].values
    y = df.iloc[1:, -2].values
    XNum = X.astype(np.float64)
    yNum = y.astype(np.float64)

    similarities = cosine_similarity(XNum, XNum)

    k = 3

    nearest_neighbors = np.zeros((len(XNum), k), dtype=int)

    for i in range(len(XNum)):
        nearest_indices = np.argsort(similarities[i])[-k-1:-1][::-1]
        nearest_neighbors[i] = yNum[nearest_indices]

    predictions = np.zeros(len(XNum), dtype=int)

    for i in range(len(XNum)):
        prediction = np.argmax(np.bincount(nearest_neighbors[i]))
        predictions[i] = prediction

    accuracy = np.mean(predictions == yNum)
    ranking = np.argsort(similarities, axis=1)[:, ::-1]

    return jsonify(accuracy=accuracy, ranking=ranking.tolist())



# API tambah tweet to CSV
@app.route('/add-tweet', methods=['POST'])
def addTweet():
    data = request.form
    
    df = pd.read_csv('data_tweet.csv', encoding='latin1')

    current_rows = df.shape[0]
    
    tweet_text = data['tweet']
    
    df.loc[current_rows, 'rawContent'] = tweet_text
    df.loc[current_rows, 'status'] = 0

    df.to_csv('data_tweet.csv', index=False)
    
    return "berhasil ditambahkan" 

# mencari persentase untuk hasil negatif dan positif
@app.route('/get-persentase-sentimen', methods=['GET'])
def getPersentaseSentimen():
    # Membaca file Excel
    df = pd.read_csv('pelabelan.csv')

    # Ambil kolom "sentimen"
    sentimen = df['sentimen']

    # Hitung jumlah prediksi positif dan negatif
    positive_count = sentimen[sentimen == 'positif'].count()
    negative_count = sentimen[sentimen == 'negatif'].count()

    # Hitung persentase perbandingan antara prediksi positif dan negatif
    total_count = len(sentimen)
    positive_percentage = (positive_count / total_count) * 100
    negative_percentage = (negative_count / total_count) * 100

    # Tampilkan hasil persentase perbandingan

    return jsonify({"Persentase_positif": positive_percentage, "Persentase_negatif": negative_percentage})

# API Chart
@app.route('/count-data-training')
def countDataTraining():
       # Baca file CSV
    df = pd.read_csv("data_tweet.csv")

    # Hitung jumlah ulasan untuk setiap rating
    rating_counts = df['status'].value_counts().sort_index()

    # Ubah hasil ke dalam format yang dapat dijsonifikasi
    result = [{"rating": str(rating), "count": int(count)} for rating, count in rating_counts.items()]

    # Kembalikan data dalam format JSON
    return jsonify(result)

    
# API data training
@app.route('/data-training')
def dataTraining():
    TWEET_DATA = pd.read_csv("data_tweet.csv", usecols=['rawContent', 'status'])
    # filtered_data = TWEET_DATA[TWEET_DATA['status'] == 1]
    data = TWEET_DATA['rawContent'].to_list()
    
    return jsonify(data)
     
    #   # Membaca data tweet dari file CSV
    # TWEET_DATA = pd.read_csv("data_tweet.csv", usecols=['rawContent', 'status'])
    
    # # Mengonversi data ke dalam format dictionary
    # data = TWEET_DATA.to_dict(orient='records')
    
    # # Mengembalikan data dalam format JSON
    # return jsonify(data)

# API data testing
@app.route('/data-testing')
def dataTesting():
    TWEET_DATA = pd.read_csv("data_tweet.csv", usecols=['rawContent', 'status'])
    filtered_data1 = TWEET_DATA.reset_index()
    filtered_data1['status'] = filtered_data1['status'].replace(0.0, 0).astype(int)
    sorted_data1 = pd.concat([
        filtered_data1[filtered_data1['status'] == 0].sort_values(by='status', ascending=False),
        filtered_data1[filtered_data1['status'] != 0]
    ])
    data1 = sorted_data1.to_dict(orient='records')

    PELABELAN_DATA = pd.read_csv("pelabelan.csv", usecols=['sentimen', 'status','aktual'])
    filtered_data2 = PELABELAN_DATA.reset_index()
    filtered_data2['status'] = filtered_data2['status'].replace(0.0, 0).astype(int)
    sorted_data2 = pd.concat([
        filtered_data2[filtered_data2['status'] == 0].sort_values(by='status', ascending=False),
        filtered_data2[filtered_data2['status'] != 0]
    ])
    data2 = sorted_data2.to_dict(orient='records')

    return jsonify({"data_tweet": data1, "data_pelabelan": data2})







# API Count data
@app.route('/data-chart')
def dataChart():
    
    TWEET_DATA = pd.read_csv("pelabelan.csv", usecols=['sentimen'])
    counts = TWEET_DATA['sentimen'].value_counts()
    data = {"positif": int(counts.get('positif', 0)), "negatif": int(counts.get('negatif', 0))}
    
    return jsonify(data)

# # API preprocessing
# @app.route('/preprocessing')
# def preprocessing():
    
#     TWEET_DATA = pd.read_csv("data_tweet.csv")
    
#     # Convert text to lowercase
#     TWEET_DATA['rawContent'] = TWEET_DATA['rawContent'].str.lower()

#     # Tokenizing
#     def remove_tweet_special(text):
#         text = text.replace('\\t'," ").replace('\\n'," ").replace('\\u'," ").replace('\\',"")
#         text = text.encode('ascii', 'replace').decode('ascii')
#         text = ' '.join(re.sub("([@#][A-Za-z0-9]+)|(\w+:\/\/\S+)"," ", text).split())
#         return text.replace("http://", " ").replace("https://", " ")

#     TWEET_DATA['rawContent'] = TWEET_DATA['rawContent'].apply(remove_tweet_special)

#     # Remove numbers
#     def remove_number(text):
#         return  re.sub(r"\d+", "", text)

#     TWEET_DATA['rawContent'] = TWEET_DATA['rawContent'].apply(remove_number)

#     # Remove punctuation
#     def remove_punctuation(text):
#         return text.translate(str.maketrans("","",string.punctuation))

#     TWEET_DATA['rawContent'] = TWEET_DATA['rawContent'].apply(remove_punctuation)

#     # Remove leading and trailing whitespaces
#     def remove_whitespace_LT(text):
#         return text.strip()

#     TWEET_DATA['rawContent'] = TWEET_DATA['rawContent'].apply(remove_whitespace_LT)

#     # Remove multiple whitespaces
#     def remove_whitespace_multiple(text):
#         return re.sub('\s+',' ',text)

#     TWEET_DATA['rawContent'] = TWEET_DATA['rawContent'].apply(remove_whitespace_multiple)

#     # Remove single characters
#     def remove_single_char(text):
#         return re.sub(r"\b[a-zA-Z]\b", "", text)

#     TWEET_DATA['rawContent'] = TWEET_DATA['rawContent'].apply(remove_single_char)

#     # NLTK word tokenization
#     def word_tokenize_wrapper(text):
#         return word_tokenize(text)

#     TWEET_DATA['tweet_tokens'] = TWEET_DATA['rawContent'].apply(word_tokenize_wrapper)
    
#     # NLTK calculate frequency distribution
#     def freqDist_wrapper(text):
#         return FreqDist(text)

#     TWEET_DATA['tweet_tokens_fdist'] = TWEET_DATA['tweet_tokens'].apply(freqDist_wrapper)
#     TWEET_DATA['tweet_tokens_fdist'].head().apply(lambda x : x.most_common())
    
#     # Additional stopwords
#     list_stopwords = stopwords.words('indonesian')
#     list_stopwords.extend(["yg", "dg", "rt", "dgn", "ny", "d", 'klo', 
#                            'kalo', 'amp', 'biar', 'bikin', 'bilang', 
#                            'gak', 'ga', 'krn', 'nya', 'nih', 'sih', 
#                            'si', 'tau', 'tdk', 'tuh', 'utk', 'ya', 
#                            'jd', 'jgn', 'sdh', 'aja', 'n', 't', 
#                            'nyg', 'hehe', 'pen', 'u', 'nan', 'loh', 'rt',
#                            '&amp', 'yah', 'game', 'bagus', 'jaring', 'gk'])

#     # Read additional stopwords from file
#     txt_stopword = pd.read_csv("stopwords.txt", names= ["stopwords"], header = None)
#     list_stopwords.extend(txt_stopword["stopwords"][0].split(' '))
#     list_stopwords = set(list_stopwords)

#     # Remove stopwords from token list
#     def stopwords_removal(words):
#         return [word for word in words if word not in list_stopwords]

#     TWEET_DATA['tweet_tokens_WSW'] = TWEET_DATA['tweet_tokens'].apply(stopwords_removal) 

#     # Stemming
#     factory = StemmerFactory()
#     stemmer = factory.create_stemmer()

#     term_dict = {}

#     for document in TWEET_DATA['tweet_tokens_WSW']:
#         for term in document:
#             if term not in term_dict:
#                 term_dict[term] = ' '

#     for term in term_dict:
#         term_dict[term] = stemmer.stem(term)

#     # Apply stemmed terms to dataframe
#     def get_stemmed_term(document):
#         return [term_dict[term] for term in document]

#     TWEET_DATA['tweet_tokens_stemmed'] = TWEET_DATA['tweet_tokens_WSW'].swifter.apply(get_stemmed_term)
    
#     # Save the preprocessed data to a CSV file
#     TWEET_DATA.to_csv("Text_Preprocessing.csv")
    
#     return "Preprocessing successful"

# API preprocessing
@app.route('/preprocessing')
def preprocessing():
    
    TWEET_DATA = pd.read_csv("data_tweet.csv")
    
     #---------- Case Folding -------
    # mengubah text menjadi lowercase
    TWEET_DATA['caseFolding'] = TWEET_DATA['rawContent'].str.lower()

    # ------ Tokenizing ---------

    def remove_tweet_special(text):
        # remove tab, new line, ans back slice
        text = text.replace('\\t'," ").replace('\\n'," ").replace('\\u'," ").replace('\\',"")
        # remove non ASCII (emoticon, chinese word, .etc)
        text = text.encode('ascii', 'replace').decode('ascii')
        # remove mention, link, hashtag
        text = ' '.join(re.sub("([@#][A-Za-z0-9]+)|(\w+:\/\/\S+)"," ", text).split())
        # remove incomplete URL
        return text.replace("http://", " ").replace("https://", " ")

    TWEET_DATA['cleaning'] = TWEET_DATA['caseFolding'].apply(remove_tweet_special)

    #remove number
    def remove_number(text):
        return  re.sub(r"\d+", "", text)

    TWEET_DATA['cleaning'] = TWEET_DATA['cleaning'].apply(remove_number)

    #remove punctuation
    def remove_punctuation(text):
        return text.translate(str.maketrans("","",string.punctuation))

    TWEET_DATA['cleaning'] = TWEET_DATA['cleaning'].apply(remove_punctuation)

    #remove whitespace leading & trailing
    def remove_whitespace_LT(text):
        return text.strip()

    TWEET_DATA['cleaning'] = TWEET_DATA['cleaning'].apply(remove_whitespace_LT)

    #remove multiple whitespace into single whitespace
    def remove_whitespace_multiple(text):
        return re.sub('\s+',' ',text)

    TWEET_DATA['cleaning'] = TWEET_DATA['cleaning'].apply(remove_whitespace_multiple)

    # remove single char
    def remove_singl_char(text):
        return re.sub(r"\b[a-zA-Z]\b", "", text)

    TWEET_DATA['cleaning'] = TWEET_DATA['cleaning'].apply(remove_singl_char)


# -----Tokenizing-----

    # NLTK word rokenize 
    def word_tokenize_wrapper(text):
        return word_tokenize(text)

    TWEET_DATA['tweet_tokens'] = TWEET_DATA['cleaning'].apply(word_tokenize_wrapper)
    
    # NLTK calc frequency distribution
    def freqDist_wrapper(text):
        return FreqDist(text)

    TWEET_DATA['tweet_tokens_fdist'] = TWEET_DATA['tweet_tokens'].apply(freqDist_wrapper)
    TWEET_DATA['tweet_tokens_fdist'].head().apply(lambda x : x.most_common())
    

    # ----------------------- get stopword from NLTK stopword -------------------------------
    # get stopword indonesia
    list_stopwords = stopwords.words('indonesian')


    # ---------------------------- manualy add stopword  ------------------------------------
    # append additional stopword
    list_stopwords.extend(["yg", "dg", "rt", "dgn", "ny", "d", 'klo', 
                           'kalo', 'amp', 'biar', 'bikin', 'bilang', 
                           'gak', 'ga', 'krn', 'nya', 'nih', 'sih', 
                           'si', 'tau', 'tdk', 'tuh', 'utk', 'ya', 
                           'jd', 'jgn', 'sdh', 'aja', 'n', 't', 
                           'nyg', 'hehe', 'pen', 'u', 'nan', 'loh', 'rt',
                           '&amp', 'yah', 'game', 'bagus', 'jaring', 'gk', 'tolong', 'terima', 'kasih', 'moonton', 'mohon', 'baik', 'mobile', 'legends', 'legend', 'mobilelegend', 'mobilelegends', 'kasi', 'bintang', 'udh'])

    # ----------------------- add stopword from txt file ------------------------------------
    # read txt stopword using pandas
    txt_stopword = pd.read_csv("stopwords.txt", names= ["stopwords"], header = None)

    # convert stopword string to list & append additional stopword
    list_stopwords.extend(txt_stopword["stopwords"][0].split(' '))

    # ---------------------------------------------------------------------------------------

    # convert list to dictionary
    list_stopwords = set(list_stopwords)


    #remove stopword pada list token
    def stopwords_removal(words):
        return [word for word in words if word not in list_stopwords]

# -----STOPWORD------
    TWEET_DATA['tweet_tokens_WSW'] = TWEET_DATA['tweet_tokens'].apply(stopwords_removal) 

    # create stemmer
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

    # stemmed
    def stemmed_wrapper(term):
        return stemmer.stem(term)

    term_dict = {}

    for document in TWEET_DATA['tweet_tokens_WSW']:
        for term in document:
            if term not in term_dict:
                term_dict[term] = ' '

    print(len(term_dict))
    print("------------------------")

    for term in term_dict:
        term_dict[term] = stemmed_wrapper(term)
        print(term,":" ,term_dict[term])

    print(term_dict)
    print("------------------------")


    # apply stemmed term to dataframe
    def get_stemmed_term(document):
        return [term_dict[term] for term in document]

    # -----STEMMING-------
    TWEET_DATA['tweet_tokens_stemmed'] = TWEET_DATA['tweet_tokens_WSW'].swifter.apply(get_stemmed_term)
    
    TWEET_DATA.to_csv("Text_Preprocessing.csv")
    
    return "preprocessing berhasil"

# API TF-IDF
@app.route('/tf-idf')
def tfidf():
    # Load tweet data from CSV
    TWEET_DATA = pd.read_csv("Text_Preprocessing.csv", usecols=["tweet_tokens_stemmed"])
    TWEET_DATA.columns = ["tweet"]

    # Convert tweet tokens from string to list
    def convert_text_list(texts):
        texts = ast.literal_eval(texts)
        return [text for text in texts]

    TWEET_DATA["tweet_list"] = TWEET_DATA["tweet"].apply(convert_text_list)
    
    # Calculate TF for each document
    def calc_TF(document):
        TF_dict = Counter(document)
        for term in TF_dict:
            TF_dict[term] = TF_dict[term] / len(document)
        return TF_dict

    TWEET_DATA["TF_dict"] = TWEET_DATA['tweet_list'].apply(calc_TF)
    
    # Calculate DF
    def calc_DF(tfDict):
        count_DF = Counter()
        for document in tfDict:
            count_DF.update(document)
        return count_DF

    DF = calc_DF(TWEET_DATA["TF_dict"])
    
    n_document = len(TWEET_DATA)

    # Calculate IDF
    def calc_IDF(__n_document, __DF):
        IDF_Dict = {}
        for term in __DF:
            IDF_Dict[term] = np.log(__n_document / (__DF[term] + 1))  # Avoid division by zero
        return IDF_Dict

    IDF = calc_IDF(n_document, DF)
    
    # Calculate TF-IDF for each document
    def calc_TF_IDF(TF, IDF):
        TF_IDF_Dict = {}
        for key in IDF:
            TF_IDF_Dict[key] = TF.get(key, 0) * IDF[key]  # Menggunakan nilai default 0 jika term tidak ada dalam TF
        return TF_IDF_Dict
    
    TWEET_DATA["TF-IDF_dict"] = TWEET_DATA.apply(lambda row: calc_TF_IDF(row["TF_dict"], IDF), axis=1)

    # Export TF-IDF vectors to CSV
    TWEET_DATA["TF_IDF_Vec"] = TWEET_DATA["TF-IDF_dict"].apply(lambda x: [x[term] for term in sorted(DF.keys())])
    TWEET_DATA["TF_IDF_Vec"].to_csv("tfidf_vectors.csv", index=False)
    
    return "TF-IDF computation completed successfully"

# API TF-IDF
@app.route('/tf-idf-2')
def tfidf2():
    TWEET_DATA = pd.read_csv("Text_Preprocessing.csv", usecols=["tweet_tokens_stemmed"])
    TWEET_DATA.columns = ["tweet"]

    def convert_text_list(texts):
        texts = ast.literal_eval(texts)
        return [text for text in texts]

    TWEET_DATA["tweet_list"] = TWEET_DATA["tweet"].apply(convert_text_list)
    
    def calc_TF(document):
        # Counts the number of times the word appears in review
        TF_dict = {}
        for term in document:
            if term in TF_dict:
                TF_dict[term] += 1
            else:
                TF_dict[term] = 1
        # Computes tf for each word
        for term in TF_dict:
            TF_dict[term] = TF_dict[term] / len(document)
        return TF_dict

    TWEET_DATA["TF_dict"] = TWEET_DATA['tweet_list'].apply(calc_TF)
    
    # Check TF result
    #index = 99
    index = TWEET_DATA.shape[0]
    
    print('%20s' % "term", "\t", "TF\n")
    for key in TWEET_DATA["TF_dict"][index-1]:
        print('%20s' % key, "\t", TWEET_DATA["TF_dict"][index-1][key])
    
    def calc_DF(tfDict):
        count_DF = {}
        # Run through each document's tf dictionary and increment countDict's (term, doc) pair
        for document in tfDict:
            for term in document:
                if term in count_DF:
                    count_DF[term] += 1
                else:
                    count_DF[term] = 1
        return count_DF

    DF = calc_DF(TWEET_DATA["TF_dict"])
    
    n_document = len(TWEET_DATA)

    def calc_IDF(__n_document, __DF):
        IDF_Dict = {}
        for term in __DF:
            IDF_Dict[term] = np.log(__n_document / (__DF[term] + 1))
        return IDF_Dict

    #Stores the idf dictionary
    IDF = calc_IDF(n_document, DF)
    
    #calc TF-IDF
    def calc_TF_IDF(TF):
        TF_IDF_Dict = {}
        #For each word in the review, we multiply its tf and its idf.
        for key in TF:
            TF_IDF_Dict[key] = TF[key] * IDF[key]
        return TF_IDF_Dict

    #Stores the TF-IDF Series
    TWEET_DATA["TF-IDF_dict"] = TWEET_DATA["TF_dict"].apply(calc_TF_IDF)
    
    # Check TF-IDF result
    #index = 99
    index = TWEET_DATA.shape[0]

    print('%20s' % "term", "\t", '%10s' % "TF", "\t", '%20s' % "TF-IDF\n")
    for key in TWEET_DATA["TF-IDF_dict"][index-1]:
        print('%20s' % key, "\t", TWEET_DATA["TF_dict"][index-1][key] ,"\t" , TWEET_DATA["TF-IDF_dict"][index-1][key])
        
    
    # sort descending by value for DF dictionary 
    sorted_DF = sorted(DF.items(), key=lambda kv: kv[1], reverse=True)[:100]

    # Create a list of unique words from sorted dictionay `sorted_DF`
    unique_term = [item[0] for item in sorted_DF]

    def calc_TF_IDF_Vec(__TF_IDF_Dict):
        TF_IDF_vector = [0.0] * len(unique_term)

        # For each unique word, if it is in the review, store its TF-IDF value.
        for i, term in enumerate(unique_term):
            if term in __TF_IDF_Dict:
                TF_IDF_vector[i] = __TF_IDF_Dict[term]
        return TF_IDF_vector

    TWEET_DATA["TF_IDF_Vec"] = TWEET_DATA["TF-IDF_dict"].apply(calc_TF_IDF_Vec)
    
    
    # Convert Series to List
    TF_IDF_Vec_List = np.array(TWEET_DATA["TF_IDF_Vec"].to_list())

    # Sum element vector in axis=0 
    sums = TF_IDF_Vec_List.sum(axis=0)

    data = []

    for col, term in enumerate(unique_term):
        data.append((term, sums[col]))

    ranking = pd.DataFrame(data, columns=['term', 'rank'])
    ranking.sort_values('rank', ascending=False)
    
    # join list of token as single document string

    def join_text_list(texts):
        texts = ast.literal_eval(texts)
        return ' '.join([text for text in texts])
    
    TWEET_DATA["tweet_join"] = TWEET_DATA["tweet"].apply(join_text_list)
    
#     from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
#     from sklearn.preprocessing import normalize

    max_features = 100

    # calc TF vector
    cvect = CountVectorizer(max_features=max_features)
    TF_vector = cvect.fit_transform(TWEET_DATA["tweet_join"])

    # normalize TF vector
    normalized_TF_vector = normalize(TF_vector, norm='l1', axis=1)

    # calc IDF
    tfidf = TfidfVectorizer(max_features=max_features, smooth_idf=False)
    tfs = tfidf.fit_transform(TWEET_DATA["tweet_join"])
    IDF_vector = tfidf.idf_

    # hitung TF x IDF sehingga dihasilkan TFIDF matrix / vector
    tfidf_mat = normalized_TF_vector.multiply(IDF_vector).toarray()

    max_features = 100

    # ngram_range (1, 3) to use unigram, bigram, trigram
    cvect = CountVectorizer(max_features=max_features, ngram_range=(1,3))
    counts = cvect.fit_transform(TWEET_DATA["tweet_join"])

    normalized_counts = normalize(counts, norm='l1', axis=1)

    tfidf = TfidfVectorizer(max_features=max_features, ngram_range=(1,3), smooth_idf=False)
    tfs = tfidf.fit_transform(TWEET_DATA["tweet_join"])

    # tfidf_mat = normalized_counts.multiply(tfidf.idf_).toarray()
    
    type(counts)
    counts.shape
    
    print(tfidf.vocabulary_)
    
    print(tfidf.get_feature_names_out())
    a=tfidf.get_feature_names_out()
    
    print(tfs.toarray())
    b=tfs.toarray()
    
    tfidf_mat = normalized_counts.multiply(IDF_vector).toarray()
    dfbtf = pd.DataFrame(data=tfidf_mat,columns=[a])
    dfbtf
    
    dfbtf.to_csv("hasil_vector_matrix.csv")
    
    return "tf-idf sukses"

# API Pelabelan sentimen
@app.route('/sentimen-pelabelan')
def pelabelan():

    # Baca file excel
    data = pd.read_csv('hasil_vector_matrix.csv')
    data1 = pd.read_csv('data_tweet.csv')
    # Mengambil indeks baris dengan status = 0
    index_zero_status = data1.loc[data1['status'] == 0].index.tolist()
    
    threshold = 0.5
    # mengambil nilai pada kolom terakhir pada setiap baris data, kecuali kolom terakhir yang merupakan label sentimen
    last_row = data.iloc[:, -1]

    # list untuk menyimpan label sentimen
    sentiments = []

    # loop pada setiap nilai pada kolom terakhir pada setiap baris data
    for i in range(len(last_row)):
        if last_row[i] >= threshold:
            # jika nilai lebih besar atau sama dengan threshold, sentimen positif
            sentiments.append('positif')
        else:
            # jika nilai kurang dari threshold, sentimen negatif
            sentiments.append('negatif')

    # menambahkan kolom "sentimen" ke dalam dataframe
    data.loc[data.index.isin(index_zero_status), 'status'] = 0
    data.loc[~data.index.isin(index_zero_status), 'status'] = 1
    
    # data['aktual'] = sentiments
    data['aktual'] = data1['status'].apply(lambda x: 'negatif' if x in [1, 2] else 'positif')
    data['sentimen'] = sentiments
    data.loc[data['status'] == 0, 'aktual'] = '-'
    
    data.to_csv("pelabelan.csv", index=False)

    # Tampilkan dataframe hasil pelabelan
    print(data)
    last_row = data.tail(1)
    data = last_row.to_dict(orient='records')
    return jsonify(data)

# approve status
@app.route('/approve-status', methods=['POST'])
def approve_status():
    # Mendapatkan indeks yang dikirim melalui body request
    index = request.json.get('index')

    # Update status pada file data_tweet.csv
    tweet_data = pd.read_csv('data_tweet.csv')
    if int(index) in tweet_data.index:
        tweet_data.at[int(index), 'status'] = 1
        tweet_data.to_csv('data_tweet.csv', index=False)
    else:
        return jsonify({'message': 'Invalid index for data_tweet.csv.'})

    # Update status pada file pelabelan.csv
    pelabelan_data = pd.read_csv('pelabelan.csv')
    if int(index) in pelabelan_data.index:
        pelabelan_data.at[int(index), 'status'] = 1
        pelabelan_data.at[int(index), 'aktual'] = pelabelan_data.at[int(index), 'sentimen']  # Menambahkan pembaruan nilai 'aktual' dengan nilai 'sentimen'
        pelabelan_data.to_csv('pelabelan.csv', index=False)
    else:
        return jsonify({'message': 'Invalid index for pelabelan.csv.'})

    return jsonify({'message': 'Status updated successfully.'})


# decline status
@app.route('/decline-status', methods=['POST'])
def decline_status():
    # Mendapatkan indeks yang dikirim melalui body request
    index = request.json.get('index')

    # Update status pada file data_tweet.csv
    tweet_data = pd.read_csv('data_tweet.csv')
    if int(index) in tweet_data.index:
        tweet_data.at[int(index), 'status'] = 2
        tweet_data.to_csv('data_tweet.csv', index=False)
    else:
        return jsonify({'message': 'Invalid index for data_tweet.csv.'})

   # Update status pada file pelabelan.csv
    pelabelan_data = pd.read_csv('pelabelan.csv')
    if int(index) in pelabelan_data.index:
        pelabelan_data.at[int(index), 'status'] = 2

        sentimen = pelabelan_data.at[int(index), 'sentimen']
        aktual = 'negatif' if sentimen == 'positif' else 'positif'
        pelabelan_data.at[int(index), 'aktual'] = aktual

        pelabelan_data.to_csv('pelabelan.csv', index=False)
    else:
        return jsonify({'message': 'Invalid index for pelabelan.csv.'})

    return jsonify({'message': 'Status updated successfully.'})

    
if __name__ == '__main__':
    app.run()


# In[ ]:




# In[ ]:


