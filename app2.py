#!/usr/bin/env python
# coding: utf-8

# In[3]:

from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import pandas as pd
import numpy as np
import string 
import re #regex library
import csv
import math
# import word_tokenize & FreqDist from NLTK
from nltk.tokenize import word_tokenize 
from nltk.probability import FreqDist
from nltk.corpus import stopwords
# !pip3 install swifter
# !pip3 install PySastrawi

from collections import Counter

import ast
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# logging.basicConfig(level=logging.DEBUG)
model = None  # Global model
scaler = None

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def initialize_model():
    # Initialize with polynomial kernel
    return SVC(kernel='poly', degree=3, coef0=1, C=1, gamma='scale')
    
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# @app.route('/sequential_svm/train', methods=['POST'])
# def train_sequential_svm():
#     global model, scaler
#     # Membaca dataset
#     df = pd.read_csv("pelabelan.csv", header=0)
#     # df = pd.read_csv("hasil_vector_matrix.csv", header=0)
    
#     # # Memisahkan fitur dan label
#     # X = df.drop(columns=['Unnamed: 0', 'aktual', 'sentimen']).values
#     # y = df['sentimen'].values
    
#     # X = df.drop(columns=['Unnamed: 0', 'aktual']).values
#     # y = df['aktual'].values
    
#     X = df.drop(columns=['status','label']).values
#     y = df['label'].values

#     # Menstandarisasi fitur
#     scaler = StandardScaler()
#     X = scaler.fit_transform(X)

#     # Menginisialisasi model
#     model = initialize_model()

#     # Melatih model
#     model.fit(X, y)

#     return "Pelatihan Sequential SVM berhasil"

@app.route('/sequential_svm/train', methods=['POST'])
def train_sequential_svm():
    global model, scaler, tfidf_vectorizer
    
    # Membaca dataset
    df = pd.read_csv("data_tweet.csv", header=0)
    # df = pd.read_csv("pelabelan.csv", header=0)
    
    # X = df['status'].astype(str).values  # Pastikan kolom 'status' berisi teks
    X = df['rawContent'].values  # Pastikan kolom 'status' berisi teks
    y = df['label'].values

    # Mengubah teks menjadi format yang sesuai untuk model menggunakan TF-IDF
    tfidf_vectorizer = TfidfVectorizer()
    X_tfidf = tfidf_vectorizer.fit_transform(X)
    
    # Menstandarisasi fitur
    scaler = StandardScaler(with_mean=False)
    X_scaled = scaler.fit_transform(X_tfidf)

    # Menginisialisasi model
    model = initialize_model()

    # Melatih model
    model.fit(X_scaled, y)
    
    # Menyimpan model, scaler, dan TF-IDF vectorizer
    with open('model.pkl', 'wb') as model_file, open('scaler.pkl', 'wb') as scaler_file, open('tfidf_vectorizer.pkl', 'wb') as tfidf_file:
        pickle.dump(model, model_file)
        pickle.dump(scaler, scaler_file)
        pickle.dump(tfidf_vectorizer, tfidf_file)

    return "Pelatihan Sequential SVM berhasil"
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix

# Inisialisasi logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

# Muat model, scaler, dan TF-IDF vectorizer
try:
    with open('model.pkl', 'rb') as model_file, open('scaler.pkl', 'rb') as scaler_file, open('tfidf_vectorizer.pkl', 'rb') as tfidf_file:
        model = pickle.load(model_file)
        scaler = pickle.load(scaler_file)
        tfidf_vectorizer = pickle.load(tfidf_file)
except FileNotFoundError as e:
    print(f"Error loading files: {e}")
    model, scaler, tfidf_vectorizer = None, None, None


# @app.route('/predict', methods=['POST'])
# def predict():
#     global model, scaler, tfidf_vectorizer 
    

#     # global model, scaler  # Pastikan scaler juga diakses secara global
#     if model is None:
#         print("Model belum dilatih.")
#         return "Model belum dilatih.", 400

#     # Menerima data dari request
#     data = request.get_json()
#     if not data or 'text' not in data:
#         print("Data tidak valid: Tidak ada teks dalam data.")
#         return "Data tidak valid", 400

#     try:
#         # Preprocessing teks input
#         processed_text = preprocess_text(data['text'])
#         print(f"Teks diproses: {processed_text}")

#         # Mengubah teks menjadi format yang sesuai untuk model menggunakan TF-IDF
#         # vectorized_text = tfidf_predict(processed_text)
#         # print(f"Teks tervektorisasi: {vectorized_text}")
   
#         vectorized_text = tfidf_vectorizer.transform([processed_text])
#         vectorized_text = vectorized_text.toarray()  # Konversi sparse matrix ke dense array
#         print(f"Teks tervektorisasi: {vectorized_text}")
        
#         # vectorized_text = tfidf_vectorizer.transform([processed_text])
#         # print(f"Teks tervektorisasi: {vectorized_text.toarray()}")
        
#         # if len(vectorized_text) < scaler.mean_.shape[0]:
#         #     vectorized_text = np.pad(vectorized_text, (0, scaler.mean_.shape[0] - len(vectorized_text)), 'constant')
#         #     print(f"Vektor fitur dipanjangkan menjadi {len(vectorized_text)} fitur.")
        
#         # elif len(vectorized_text) > scaler.mean_.shape[0]:
#         #     vectorized_text = vectorized_text[:scaler.mean_.shape[0]]
#         #     print(f"Vektor fitur dipotong menjadi {len(vectorized_text)} fitur.")

#         # Standarisasi fitur sebelum prediksi
#         # vectorized_text = scaler.transform([vectorized_text])  # Pastikan menggunakan transform bukan fit_transform
#         # print(f"Teks terstandarisasi: {vectorized_text}.")

#         vectorized_text = scaler.transform(vectorized_text)
#         print(f"Teks terstandarisasi: {vectorized_text}")

#         # Melakukan prediksi menggunakan model yang sudah dilatih
#         prediction = model.predict(vectorized_text)
#         print(f"Prediksi: {prediction}")

#         # Menghitung kernel dari data uji
#         # kernel_matrix = model.decision_function(vectorized_text)
#         # print(f"Matriks kernel data uji: {kernel_matrix}")
#         kernel_matrix = model.decision_function(vectorized_text).tolist()  # Konversi ke list
#         print(f"Matriks kernel data uji: {kernel_matrix}")

#         # Mendapatkan nilai alpha (ai) dan support vectors (yi)
#         # dual_coef = model.dual_coef_
#         # support_vectors = model.support_vectors_
#         # print(f"Nilai alpha (ai): {dual_coef}")
#         # print(f"Support vectors (yi): {support_vectors}")
        
#         dual_coef = model.dual_coef_.tolist()  # Konversi ke list
#         support_vectors = model.support_vectors_.toarray().tolist()  # Konversi ke dense array dan kemudian ke list
#         print(f"Nilai alpha (ai): {dual_coef}")
#         print(f"Support vectors (yi): {support_vectors}")

#         # Mengonversi prediksi numerik ke label sentimen
#         sentiment = 'Negatif' if kernel_matrix[0] < 0 else 'Positif'
#         print(f"Sentimen yang diprediksi: {sentiment}")
        
#         return jsonify({"sentimen": prediction, "kernel_matrix": kernel_matrix, "alpha_values": dual_coef, "support_vectors": support_vectors})
#         # return jsonify({"sentimen": prediction, "kernel_matrix": kernel_matrix.tolist(), "alpha_values": dual_coef.tolist(), "support_vectors": support_vectors.tolist()})
#     except Exception as e:
#         print(f"Kesalahan dalam pemrosesan prediksi: {str(e)}")
#         if isinstance(e, TypeError) and "float() argument" in str(e):
#             print("Tipe data yang diberikan tidak sesuai, pastikan data yang diberikan adalah numerik.")
#             return jsonify({"error": "Kesalahan dalam pemrosesan prediksi", "message": "Tipe data yang diberikan tidak sesuai, pastikan data yang diberikan adalah numerik."}), 500
#         return jsonify({"error": "Kesalahan dalam pemrosesan prediksi", "message": str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    global model, scaler, tfidf_vectorizer

    if model is None:
        print("Model belum dilatih.")
        return "Model belum dilatih.", 400

    data = request.get_json()
    if not data or 'text' not in data:
        print("Data tidak valid: Tidak ada teks dalam data.")
        return "Data tidak valid", 400

    try:
        processed_text = preprocess_text(data['text'])
        print(f"Teks diproses: {processed_text}")

        vectorized_text = tfidf_vectorizer.transform([processed_text])
        vectorized_text = vectorized_text.toarray()
        print(f"Teks tervektorisasi: {vectorized_text}")

        vectorized_text = scaler.transform(vectorized_text)
        print(f"Teks terstandarisasi: {vectorized_text}")

        prediction = model.predict(vectorized_text)
        prediction = prediction.tolist()
        print(f"Prediksi: {prediction}")

        kernel_matrix = model.decision_function(vectorized_text).tolist()
        print(f"Matriks kernel data uji: {kernel_matrix}")

      
        return jsonify({
            "sentimen": prediction,
            "kernel_matrix": kernel_matrix,
            "preprocess_text": processed_text
        })
    except Exception as e:
        print(f"Kesalahan dalam pemrosesan prediksi: {str(e)}")
        if isinstance(e, TypeError) and "float() argument" in str(e):
            print("Tipe data yang diberikan tidak sesuai, pastikan data yang diberikan adalah numerik.")
            return jsonify({
                "error": "Kesalahan dalam pemrosesan prediksi",
                "message": "Tipe data yang diberikan tidak sesuai, pastikan data yang diberikan adalah numerik."
            }), 500
        return jsonify({
            "error": "Kesalahan dalam pemrosesan prediksi",
            "message": str(e)
        }), 500


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

def tfidf_predict(processed_text):
    # Konversi teks yang diproses menjadi list token
    input_tokens = word_tokenize(processed_text)
    print(f"Input tokens: {input_tokens}")

    # Load data untuk mendapatkan DF dan IDF yang sudah ada
    TWEET_DATA = pd.read_csv("Text_Preprocessing.csv", usecols=["tweet_tokens_stemmed"])
    TWEET_DATA["tweet_list"] = TWEET_DATA["tweet_tokens_stemmed"].apply(ast.literal_eval)
    print(f"Loaded tweet data: {TWEET_DATA.head()}")

    # Menghitung DF dari data yang ada
    def calc_DF(tfDict):
        count_DF = Counter()
        for document in tfDict:
            count_DF.update(document)
        return count_DF

    DF = calc_DF(TWEET_DATA["tweet_list"].apply(Counter))
    print(f"Document Frequency (DF): {DF}")

    n_document = len(TWEET_DATA) + 1  # Termasuk dokumen input
    max_proses = 1000  # Batas maksimum proses
    n_document = min(n_document, max_proses)  # Batasi jumlah dokumen yang diproses
    print(f"Number of documents: {n_document}")

    # Menghitung IDF
    def calc_IDF(__n_document, __DF):
        IDF_Dict = {}
        for term in __DF:
            IDF_Dict[term] = np.log(__n_document / (__DF[term] + 1))
        return IDF_Dict

    IDF = calc_IDF(n_document, DF)
    print(f"Inverse Document Frequency (IDF): {IDF}")

    # Menghitung TF untuk input
    TF_dict = Counter(input_tokens)
    for term in TF_dict:
        TF_dict[term] = TF_dict[term] / len(input_tokens)
    print(f"Term Frequency (TF) for input: {TF_dict}")

    # Menghitung TF-IDF untuk input
    def calc_TF_IDF(TF, IDF):
        TF_IDF_Dict = {}
        for term in IDF:
            TF_IDF_Dict[term] = TF.get(term, 0) * IDF[term]
        return TF_IDF_Dict

    TF_IDF_dict = calc_TF_IDF(TF_dict, IDF)
    print(f"TF-IDF dictionary for input: {TF_IDF_dict}")

    # Mengonversi TF-IDF dict ke vektor berdasarkan urutan DF
    TF_IDF_vector = np.array([TF_IDF_dict.get(term, 0) for term in sorted(DF.keys())])
    print(f"TF-IDF vector: {TF_IDF_vector}")

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
        
        total_data = df.shape[0]
        print(f"Total Data: {total_data}")

        label_encoder = LabelEncoder()
        df['label'] = label_encoder.fit_transform(df['label'])

        # Pisahkan fitur dan label
        X = df.drop(columns=['status', 'label'])
        y = df['label']
        
        XNum = X.astype(np.float64)
        yNum = y.astype(np.float64)

        # Memeriksa apakah terdapat lebih dari satu kelas dalam data
        if len(np.unique(yNum)) <= 1:
            return "Error: Jumlah kelas harus lebih dari satu; hanya mendapatkan 1 kelas"

        # Get test_size parameter from request
        test_size = request.args.get('test_size', default=0.2, type=float)
        # Split data menjadi data latih dan data uji
        # X_train, X_test, y_train, y_test = train_test_split(XNum, yNum, test_size=0.4, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(XNum, yNum, test_size=test_size, random_state=42)

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
        
        total_data_y = total_data * test_size

        return jsonify(accuracy=accuracy, confusion_matrix=cm.tolist(),
                    tn=tn, fp=fp, fn=fn, tp=tp,
                    classification_report=classification_rep,
                    precision=precision, f1_score=f1, recall=recall, total_data=total_data_y)
    except Exception as e:
        app.logger.error(f"Kesalahan dalam memproses permintaan SVM: {str(e)}")
        return jsonify({"error": f"Kesalahan dalam memproses permintaan SVM: {str(e)}"}), 500


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

    
@app.route('/data-training')
def dataTraining():
    TWEET_DATA = pd.read_csv("data_tweet.csv", usecols=['rawContent', 'status'])
    # TWEET_DATA['status'] = TWEET_DATA['status'].astype(int)  
    data = TWEET_DATA.to_dict(orient='records')

    negatif_count = 0
    positif_count = 0
    unlabel_count = 0

    for entry in data:
        if entry['status'] in [1, 2]:
            entry['status'] = 'Negatif'
            negatif_count += 1
        elif entry['status'] in [4, 5]:
            entry['status'] = 'Positif'
            positif_count += 1
        else:
            entry['status'] = 'Unlabel'
            unlabel_count += 1

    response = {
        'data': data,
        'positif_count': positif_count,
        'negatif_count': negatif_count
    }

    return jsonify(response)
  # Memeriksa ekstensi file
 
    # TWEET_DATA = pd.read_excel("data_tweet.csv", usecols=['rawContent', 'status'])
    # # Mengonversi data ke dalam format dictionary
    # data = TWEET_DATA.to_dict(orient='records')
    
    # Mengembalikan data dalam format JSON
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
# @app.route('/preprocessing')
# def preprocessing():
    
#     TWEET_DATA = pd.read_csv("data_tweet.csv")
    
#     # Menghapus kolom selain 'rawContent'
#     raw_content_df = TWEET_DATA.drop(columns=[col for col in TWEET_DATA.columns if col != 'rawContent'])
    
#      #---------- Case Folding -------
#     # mengubah text menjadi lowercase
#     raw_content_df['caseFolding'] = raw_content_df['rawContent'].str.lower()

#     # ------ Tokenizing ---------

#     def remove_tweet_special(text):
#         # remove tab, new line, ans back slice
#         text = text.replace('\\t'," ").replace('\\n'," ").replace('\\u'," ").replace('\\',"")
#         # remove non ASCII (emoticon, chinese word, .etc)
#         text = text.encode('ascii', 'replace').decode('ascii')
#         # remove mention, link, hashtag
#         text = ' '.join(re.sub("([@#][A-Za-z0-9]+)|(\w+:\/\/\S+)"," ", text).split())
#         # remove incomplete URL
#         return text.replace("http://", " ").replace("https://", " ")

#     raw_content_df['cleaning'] = raw_content_df['caseFolding'].apply(remove_tweet_special)

#     #remove number
#     def remove_number(text):
#         return  re.sub(r"\d+", "", text)

#     raw_content_df['cleaning'] = raw_content_df['cleaning'].apply(remove_number)

#     #remove punctuation
#     def remove_punctuation(text):
#         return text.translate(str.maketrans("","",string.punctuation))

#     raw_content_df['cleaning'] = raw_content_df['cleaning'].apply(remove_punctuation)

#     #remove whitespace leading & trailing
#     def remove_whitespace_LT(text):
#         return text.strip()

#     raw_content_df['cleaning'] = raw_content_df['cleaning'].apply(remove_whitespace_LT)

#     #remove multiple whitespace into single whitespace
#     def remove_whitespace_multiple(text):
#         return re.sub('\s+',' ',text)

#     raw_content_df['cleaning'] = raw_content_df['cleaning'].apply(remove_whitespace_multiple)

#     # remove single char
#     def remove_singl_char(text):
#         return re.sub(r"\b[a-zA-Z]\b", "", text)

#     raw_content_df['cleaning'] = raw_content_df['cleaning'].apply(remove_singl_char)


# # -----Tokenizing-----

#     # NLTK word rokenize 
#     def word_tokenize_wrapper(text):
#         return word_tokenize(text)

#     raw_content_df['tweet_tokens'] = raw_content_df['cleaning'].apply(word_tokenize_wrapper)
    
#     # NLTK calc frequency distribution
#     def freqDist_wrapper(text):
#         return FreqDist(text)

#     raw_content_df['tweet_tokens_fdist'] = raw_content_df['tweet_tokens'].apply(freqDist_wrapper)
#     raw_content_df['tweet_tokens_fdist'].head().apply(lambda x : x.most_common())
    

#     # ----------------------- get stopword from NLTK stopword -------------------------------
#     # get stopword indonesia
#     list_stopwords = stopwords.words('indonesian')

#     # ---------------------------- manualy add stopword  ------------------------------------
#     # append additional stopword
#     list_stopwords.extend(["yg", "dg", "rt", "dgn", "ny", "d", 'klo', 
#                            'kalo', 'amp', 'biar', 'bikin', 'bilang', 
#                            'gak', 'ga', 'krn', 'nya', 'nih', 'sih', 
#                            'si', 'tau', 'tdk', 'tuh', 'utk', 'ya', 
#                            'jd', 'jgn', 'sdh', 'aja', 'n', 't', 
#                            'nyg', 'hehe', 'pen', 'u', 'nan', 'loh', 'rt',
#                            '&amp', 'yah', 'game', 'bagus', 'jaring', 'gk', 'tolong', 'terima', 'kasih', 'moonton', 'mohon', 'baik', 'mobile', 'legends', 'legend', 'mobilelegend', 'mobilelegends', 'kasi', 'bintang', 'udh'])

#     # ----------------------- add stopword from txt file ------------------------------------
#     # read txt stopword using pandas
#     # txt_stopword = pd.read_csv("stopwords.txt", names= ["stopwords"], header = None)

#     # convert stopword string to list & append additional stopword
#     # list_stopwords.extend(txt_stopword["stopwords"][0].split(' '))
    
#     # Read the stopwords from the txt file
#     txt_stopword = pd.read_csv("stopwords.txt", names=["stopwords"], header=None)

#     # Convert the stopwords column to a list
#     stopword_list = txt_stopword["stopwords"].tolist()
#     # ---------------------------------------------------------------------------------------

#     # Extend the list_stopwords with the stopwords from the file   
#     list_stopwords.extend(stopword_list)
#     # convert list to dictionary
#     list_stopwords = set(list_stopwords)


#     #remove stopword pada list token
#     def stopwords_removal(words):
#         return [word for word in words if word not in list_stopwords]

# # -----STOPWORD------
#     raw_content_df['tweet_tokens_WSW'] = raw_content_df['tweet_tokens'].apply(stopwords_removal) 


#  # ------------------- Normalization -----------------------
#     # normalization_dict = {
#     #     "gak": "tidak", "ga": "tidak", "kalo": "kalau", "klo": "kalau",
#     #     "nya": "dia", "nih": "ini", "sih": "saja", "tau": "tahu",
#     #     "tdk": "tidak", "tuh": "itu", "utk": "untuk", "ya": "iya", "jd": "jadi",
#     #     "jgn": "jangan", "aja": "saja", "dgn": "dengan", "dg": "dengan", "yg": "yang"
#     # }
#     normalization_dict = {}
#     # with open('normalisasi.txt', 'r') as file:
#     #     for line in file:
#     #         slang, normal = line.strip().split(':')
#     #         normalization_dict[slang] = normal
#     # with open('normalisasi.txt', 'r') as file:
#     #     for line in file:
#     #         if ':' in line:
#     #             slang, normal = line.strip().replace('"', '').split(': ')
#     #             normalization_dict[slang.strip()] = normal.strip()
#     with open('normalisasi.txt', 'r') as file:
#         for line in file:
#             if ':' in line:
#                 parts = line.strip().replace('"', '').split(': ')
#                 if len(parts) == 2:
#                     slang, normal = parts
#                     normalization_dict[slang.strip()] = normal.strip()
#     def normalize_text(text):
#         return [normalization_dict.get(word, word) for word in text]

#     # def normalize_text(text):
#     #     return [normalization_dict[word] if word in normalization_dict else word for word in text]

#     raw_content_df['tweet_tokens_normalized'] = raw_content_df['tweet_tokens_WSW'].apply(normalize_text)


#     # create stemmer
#     factory = StemmerFactory()
#     stemmer = factory.create_stemmer()

#     # stemmed
#     def stemmed_wrapper(term):
#         return stemmer.stem(term)

#     term_dict = {}

#     # for document in TWEET_DATA['tweet_tokens_WSW']:
#     #     for term in document:
#     #         if term not in term_dict:
#     #             term_dict[term] = ' '
#     for document in raw_content_df['tweet_tokens_normalized']:
#         for term in document:
#             if term not in term_dict:
#                 term_dict[term] = stemmed_wrapper(term)

#     print(len(term_dict))
#     print("------------------------")

#     for term in term_dict:
#         term_dict[term] = stemmed_wrapper(term)
#         print(term,":" ,term_dict[term])

#     print(term_dict)
#     print("------------------------")


#     # apply stemmed term to dataframe
#     def get_stemmed_term(document):
#         return [term_dict[term] for term in document]

#     # -----STEMMING-------
#     raw_content_df['tweet_tokens_stemmed'] = raw_content_df['tweet_tokens_normalized'].swifter.apply(get_stemmed_term)
    
#     raw_content_df.to_csv("Text_Preprocessing.csv")
    
#     return "preprocessing berhasil"


class Preprocessor:
    def __init__(self):
        # Load stopwords from file
        self.stoplist = set(pd.read_csv("stopwords.txt", names=["stopwords"], header=None)['stopwords'])
        
        # Load normalization dictionary from file
        self.normalization_dict = {}
        with open('normalisasi.txt', 'r') as file:
            for line in file:
                if ':' in line:
                    parts = line.strip().replace('"', '').split(': ')
                    if len(parts) == 2:
                        slang, normal = parts
                        self.normalization_dict[slang.strip()] = normal.strip()
    
    def preprocess(self, df):
        if 'rawContent' not in df.columns:
            raise ValueError("The DataFrame does not contain the 'rawContent' column.")
        
        # Case Folding
        df['rawContent'] = df['rawContent'].str.lower()
        
        # Data Cleaning
        df['rawContent'] = df['rawContent'].apply(lambda x: re.sub(r'(\A|\s)@(\w+)|(^https?:\/\/.*[\r\n]*)|(\A|\s)[0-9](\w+)|(\d)', " ", x))
        df['rawContent'] = df['rawContent'].apply(lambda x: re.sub(r'\d', " ", x))
        df['rawContent'] = df['rawContent'].apply(lambda x: re.sub(r'[.?%&^*!()/,`~;:< >#]', " ", x))
        
        # Normalisasi
        def normalize_text(text):
            words = text.split()
            normalized_words = [self.normalization_dict.get(word, word) for word in words]
            return ' '.join(normalized_words)
        
        df['rawContent'] = df['rawContent'].apply(normalize_text)
        
        # Stopword Removal
        df['rawContent'] = df['rawContent'].apply(lambda x: " ".join(word for word in x.split() if word not in self.stoplist))
        
        # Tokenization
        df['rawContent'] = df['rawContent'].apply(lambda x: x.split())
        
        return df
    
    # def preprocess2(self, text):
    #     # Case Folding
    #     text = text.lower()
        
    #     # Data Cleaning
    #     text = re.sub(r'(\A|\s)@(\w+)|(^https?:\/\/.*[\r\n]*)|(\A|\s)[0-9](\w+)|(\d)', " ", text)
    #     text = re.sub(r'\d', " ", text)
    #     text = re.sub(r'[.?%&^*!()/,`~;:< >#]', " ", text)
        
    #     # Normalization
    #     def normalize_text(text):
    #         words = text.split()
    #         normalized_words = [self.normalization_dict.get(word, word) for word in words]
    #         return ' '.join(normalized_words)
        
    #     text = normalize_text(text)
        
    #     # Stopword Removal
    #     text = " ".join(word for word in text.split() if word not in self.stoplist)
        
    #     # Tokenization
    #     text = text.split()
        
    #     return text
    
    def preprocess2(self, text):
        # Proses preprocessing untuk satu data teks
        processed_text = self.preprocess_single(text)
        return processed_text

    def preprocess_single(self, text):
        # Implementasi lengkap dari preprocessing untuk satu data teks
        # Lakukan case folding, data cleaning, normalisasi, stopword removal, dan tokenization
        processed_text = text.lower()
        processed_text = re.sub(r'(\A|\s)@(\w+)|(^https?:\/\/.*[\r\n]*)|(\A|\s)[0-9](\w+)|(\d)', " ", processed_text)
        processed_text = re.sub(r'\d', " ", processed_text)
        processed_text = re.sub(r'[.?%&^*!()/,`~;:< >#]', " ", processed_text)
        processed_text = self.normalize_text(processed_text)
        processed_text = " ".join(word for word in processed_text.split() if word not in self.stoplist)
        processed_text = processed_text.split()
        
        return processed_text

    def normalize_text(self, text):
        # Normalisasi teks
        words = text.split()
        normalized_words = [self.normalization_dict.get(word, word) for word in words]
        return ' '.join(normalized_words)
    
    def fitur_kata(self, hasil_preprocessing):
        fitur = []
        for i in range(len(hasil_preprocessing)):
            for word in hasil_preprocessing[i][0]:
                if word not in fitur:
                    fitur.append(word)
        return fitur

    def tf(self, hasil_preprocessing, fitur):
        tf = []
        for i in range(len(fitur)):
            temp= []
            for j in range(len(hasil_preprocessing)):
                temp.append(hasil_preprocessing[j][0].count(fitur[i]))
            tf.append(temp)
        return tf

    def wtf(self, tf):
        wtf = []
        for i in range(len(tf)):
            temp= []
            for j in range(len(tf[i])):
                hasil = 0
                if tf[i][j] > 0:
                    hasil = 1 + math.log10(tf[i][j])
                temp.append(hasil)
            wtf.append(temp)
        return wtf

    # def dfidf(self, tf):
    #     df = []
    #     idf = []
    #     for i in range(len(tf)):
    #         hasil_df = 0
    #         for j in range(len(tf[i])):
    #             if tf[i][j] > 0:
    #                 hasil_df = tf[i][j] + hasil_df
    #         hasil_idf = math.log10(float((len(tf[i])))/float(hasil_df))
    #         df.append(hasil_df)
    #         idf.append(hasil_idf)
    #     return df, idf
    def dfidf(self, tf):
        idf = []
        for i in range(len(tf)):
            hasil_df = 0
            for j in range(len(tf[i])):
                if tf[i][j] > 0:
                    hasil_df = tf[i][j] + hasil_df
            hasil_idf = math.log10(float((len(tf[i])))/float(hasil_df))
            idf.append(hasil_idf)
        return idf

    def tfidf(self, wtf, idf):
        tfidf = []
        for i in range(len(idf)):
            temp = []
            for j in range(len(wtf[i])):
                if wtf[i][j] > 0:
                    hasil = wtf[i][j] * idf[i]
                    temp.append(hasil)
                else:
                    temp.append(0)
            tfidf.append(temp)
        return tfidf


    # def tfidf(self, wtf, idf):
    #     tfidf = []
    #     for i in range(len(idf)):
    #         temp = []
    #         for j in range(len(wtf[i])):
    #             if wtf[i][j] > 0:
    #                 hasil = wtf[i][j] * idf[i]
    #                 temp.append(hasil)
    #             else:
    #                 temp.append(0)
    #         tfidf.append(temp)
    #     return tfidf
    def transpose(self, tfidf):
        transpose = []
        for i in range(len(tfidf[0])):
            temp = []
            for j in range(len(tfidf)):
                temp.append(tfidf[j][i])
            transpose.append(temp)
        return transpose
    
    def kernel(self, mat, trans, d):
        c = 1
        mat_kernel = []
        print(f"Shape of trans: {len(trans)}x{len(trans[0])} | Shape of mat: {len(mat)}x{len(mat[0])}")
        for i in range(len(trans)):
            temp = []
            for j in range(len(mat[0])):  # Adjust the iteration to match mat's columns
                total = 0
                for x in range(len(trans[0])):
                    try:
                        hasil = trans[i][x] * mat[x][j]
                    except IndexError as e:
                        print(f"IndexError at trans[{i}][{x}] and mat[{x}][{j}]: {e}")
                        raise
                    total += hasil
                tambah_c = total + c
                hasil_pangkat = math.pow(tambah_c, d)
                kernel = hasil_pangkat
                temp.append(kernel)
            mat_kernel.append(temp)
        return mat_kernel
    
    # def kernel(self, trans, mat, d):
    #     # Print ukuran matriks untuk debugging
    #     print(f"Ukuran trans: {len(trans)}x{len(trans[0])} (if not empty)")
    #     print(f"Ukuran mat: {len(mat)}x{len(mat[0])} (if not empty)")

    #     mat_kernel = []
    #     try:
    #         for i in range(len(trans)):
    #             row_result = []
    #             for j in range(len(mat[0])):  # mat[0] to get the number of columns in mat
    #                 total = 0
    #                 for x in range(len(trans[i])):  # Length of a row in trans
    #                     total += trans[i][x] * mat[x][j]
    #                 row_result.append(total)
    #             mat_kernel.append(row_result)
    #     except IndexError as e:
    #         print(f"IndexError: {e}")
    #         print(f"Error at trans[{i}][{x}] and mat[{x}][{j}]")
    #         raise
        
    #     return mat_kernel


    # def kernel(self, trans, mat, d):
    #     # Pastikan ukuran list sesuai
    #     print(f"Size of trans: {len(trans)}")
    #     if len(trans) > 0:
    #         print(f"Size of trans[0]: {len(trans[0])}")
        
    #     print(f"Size of mat: {len(mat)}")
    #     if len(mat) > 0:
    #         print(f"Size of mat[0]: {len(mat[0])}")

    #     hasil = 0
    #     for i in range(len(trans)):
    #         for j in range(len(mat[0])):
    #             for x in range(len(trans[0])):
    #                 # Tambahkan validasi indeks
    #                 if x < len(mat) and j < len(mat[x]):
    #                     hasil += trans[i][x] * mat[x][j]
    #                 else:
    #                     print(f"Index out of range: trans[{i}][{x}], mat[{x}][{j}]")

    #     return hasil


    # def hessian(kelas, kernel):
    #     lamda = 0.5
    #     hessian = []
    #     for i in range(len(kernel)):
    #         temp = []
    #         for j in range(len(kernel[0])):
    #             hess = kelas[i] * kelas[j] * kernel[i][j] + (math.pow(lamda, 2))
    #             temp.append(hess)
    #         hessian.append(temp)
    #     return hessian

    def hessian(self, kelas, kernel):
        lamda = 0.5
        hessian = []
        if isinstance(kernel, list) and all(isinstance(row, list) for row in kernel):
            for i in range(len(kernel)):
                temp = []
                for j in range(len(kernel[0])):
                    hess = kelas[i] * kelas[j] * kernel[i][j] + (math.pow(lamda, 2))
                    temp.append(hess)
                hessian.append(temp)
        else:
            raise TypeError("Expected kernel to be a list of lists (matrix)")
        return hessian



    def sequential(self, hessian, itermax, lr):
        epsilon = 0.0001
        c = 1
        alfa = [[0] * len(hessian)]
        E = []
        delta_a = []
        for _ in range(itermax):
            temp_e = []
            temp_d = []
            temp_a = []
            for j in range(len(hessian)):
                total = 0
                for x in range(len(hessian[0])):
                    total += hessian[j][x] * alfa[-1][x]
                temp_e.append(total)
            E.append(temp_e)
            for x in range(len(E[-1])):
                hasil = min(max(lr * (1 - E[-1][x]), -alfa[-1][x]), c - alfa[-1][x])
                temp_d.append(hasil)
            delta_a.append(temp_d)
            for x in range(len(temp_d)):
                hasil = alfa[-1][x] + delta_a[-1][x]
                temp_a.append(hasil)
            alfa.append(temp_a)
        return E[-1], delta_a[-1], alfa[-1]

    # def bias(self, matrix, sv, kelas, d):
    #     c = 1
    #     xpos = xneg = 0
    #     for i in range(len(kelas)):
    #         if kelas[i] == -1:
    #             xneg = max(xneg, sv[i])
    #         elif kelas[i] == 1:
    #             xpos = max(xpos, sv[i])
    #     alpos = sv.index(xpos)
    #     alneg = sv.index(xneg)
    #     sigmapos = sigmaneg = 0
    #     for i in range(len(kelas)):
    #         totpos = totneg = 0
    #         for j in range(len(matrix[i])):
    #             totpos += matrix[i][j] * matrix[alpos][j]
    #             totneg += matrix[i][j] * matrix[alneg][j]
    #         kernelpos = math.pow((totpos + c), d)
    #         kernelneg = math.pow((totneg + c), d)
    #         h_pos = sv[i] * kelas[i] * kernelpos
    #         h_neg = sv[i] * kelas[i] * kernelneg
    #         sigmapos += h_pos
    #         sigmaneg += h_neg
    #     bias_value = -0.5 * (sigmapos + sigmaneg)
    #     return bias_value
    
    # def bias(self, matrix, sv, kelas, d):
    #     c = 1
    #     xpos = None
    #     xneg = None
        
    #     # Find maximum sv for class 1
    #     for i in range(len(kelas)):
    #         if kelas[i] == 1:
    #             if xpos is None or sv[i] > sv[xpos]:
    #                 xpos = i
        
    #     # Find maximum sv for class -1
    #     for i in range(len(kelas)):
    #         if kelas[i] == -1:
    #             if xneg is None or sv[i] > sv[xneg]:
    #                 xneg = i
        
    #     if xpos is None or xneg is None:
    #         raise ValueError("No suitable support vectors found for classes 1 or -1.")
        
    #     alpos = xpos
    #     alneg = xneg
        
    #     sigmapos = sigmaneg = 0
        
    #     for i in range(len(kelas)):
    #         totpos = totneg = 0
            
    #         for j in range(len(matrix[i])):
    #             totpos += matrix[i][j] * matrix[alpos][j]
    #             totneg += matrix[i][j] * matrix[alneg][j]
            
    #         kernelpos = math.pow((totpos + c), d)
    #         kernelneg = math.pow((totneg + c), d)
            
    #         h_pos = sv[i] * kelas[i] * kernelpos
    #         h_neg = sv[i] * kelas[i] * kernelneg
            
    #         sigmapos += h_pos
    #         sigmaneg += h_neg
        
    #     bias_value = -0.5 * (sigmapos + sigmaneg)
        
    #     return bias_value

    # def bias(self, matrix, sv, kelas, d):
    #     c = 1
    #     xpos = xneg = None
        
    #     # Find maximum sv for class 1
    #     for i in range(len(kelas)):
    #         if kelas[i] == 1:
    #             if xpos is None or sv[i] > sv[xpos]:
    #                 xpos = i
        
    #     # Find maximum sv for class -1
    #     for i in range(len(kelas)):
    #         if kelas[i] == -1:
    #             if xneg is None or sv[i] > sv[xneg]:
    #                 xneg = i
        
    #     if xpos is None or xneg is None:
    #         raise ValueError("No suitable support vectors found for classes 1 or -1. Please check your SVM model input.")
        
    #     alpos = xpos
    #     alneg = xneg
        
    #     sigmapos = sigmaneg = 0
        
    #     for i in range(len(kelas)):
    #         totpos = totneg = 0
            
    #         if alpos < len(matrix):
    #             for j in range(len(matrix[i])):
    #                 totpos += matrix[i][j] * matrix[alpos][j]
            
    #         if alneg < len(matrix):
    #             for j in range(len(matrix[i])):
    #                 totneg += matrix[i][j] * matrix[alneg][j]
            
    #         kernelpos = math.pow((totpos + c), d)
    #         kernelneg = math.pow((totneg + c), d)
            
    #         h_pos = sv[i] * kelas[i] * kernelpos
    #         h_neg = sv[i] * kelas[i] * kernelneg
            
    #         sigmapos += h_pos
    #         sigmaneg += h_neg
        
    #     bias_value = -0.5 * (sigmapos + sigmaneg)
        
    #     return bias_value

    def bias(self, matrix, sv, kelas, d):
        c = 1
        xpos = xneg = None

        # Debugging information
        print(f"Length of matrix: {len(matrix)}")
        print(f"Length of kelas: {len(kelas)}")

        # Check if the lengths of matrix and kelas match
        if len(matrix) != len(kelas):
            raise ValueError(f"Length of matrix ({len(matrix)}) does not match length of kelas ({len(kelas)})")

        # Find maximum sv for class 1
        for i in range(len(matrix)):
            if kelas[i] == 1:
                if xpos is None or sv[i] > sv[xpos]:
                    xpos = i

        # Find maximum sv for class -1
        for i in range(len(matrix)):
            if kelas[i] == -1:
                if xneg is None or sv[i] > sv[xneg]:
                    xneg = i

        if xpos is None or xneg is None:
            raise ValueError("No suitable support vectors found for classes 1 or -1. Please check your SVM model input.")

        alpos = xpos
        alneg = xneg

        sigmapos = sigmaneg = 0

        for i in range(len(matrix)):
            totpos = totneg = 0

            # Debugging information
            print(f"Processing row {i}, alpos: {alpos}, alneg: {alneg}")
            if alpos >= len(matrix) or alneg >= len(matrix):
                print(f"Index out of range: alpos={alpos}, alneg={alneg}, len(matrix)={len(matrix)}")
                raise IndexError("alpos or alneg index out of range.")

            if i >= len(matrix):
                print(f"Index out of range: i={i}, len(matrix)={len(matrix)}")
                raise IndexError("i index out of range.")
            
            print(f"Row {i} length: {len(matrix[i])}")

            if alpos < len(matrix):
                for j in range(len(matrix[i])):
                    totpos += matrix[i][j] * matrix[alpos][j]

            if alneg < len(matrix):
                for j in range(len(matrix[i])):
                    totneg += matrix[i][j] * matrix[alneg][j]

            kernelpos = math.pow((totpos + c), d)
            kernelneg = math.pow((totneg + c), d)

            h_pos = sv[i] * kelas[i] * kernelpos
            h_neg = sv[i] * kelas[i] * kernelneg

            sigmapos += h_pos
            sigmaneg += h_neg

        bias_value = -0.5 * (sigmapos + sigmaneg)

        return bias_value


    # def datatesting(self, kelas, matrix, trans, d, alfa, bias):
    #     c = 1
    #     hasiltes = []
    #     hasilkelas = []
    #     for i in range(len(matrix[0])):
    #         temp = []
    #         for j in range(len(kelas)):
    #             total = 0
    #             for x in range(len(matrix[i])):
    #                 total += trans[i][x] * matrix[x][j]
    #             kernel_value = math.pow((total + c), d)
    #             hasil = alfa[j] * kelas[j] * kernel_value
    #             temp.append(hasil)
    #         jumlah = sum(temp) + bias
    #         hasiltes.append(jumlah)
    #         klas = 1 if jumlah > 0 else -1
    #         hasilkelas.append(klas)
    #     return hasiltes, hasilkelas
    
    def datatesting(self, kelas, matrix, trans, d, alfa, bias):
        c = 1
        hasiltes = []
        hasilkelas = []
        
        # Periksa panjang trans
        if len(trans) <= 0 or len(trans[0]) <= 0 or len(matrix) <= 0 or len(matrix[0]) <= 0:
            raise ValueError("Matrix dimensions are not valid.")
        
        for i in range(len(matrix[0])):
            temp = []
            
            # Pastikan i dalam rentang yang valid
            if i < len(trans):
                for j in range(len(kelas)):
                    total = 0
                    for x in range(len(trans[i])):
                        total += trans[i][x] * matrix[x][j]
                    
                    kernel_value = math.pow((total + c), d)
                    hasil = alfa[j] * kelas[j] * kernel_value
                    temp.append(hasil)
                
                jumlah = sum(temp) + bias
                hasiltes.append(jumlah)
                klas = 1 if jumlah > 0 else -1
                hasilkelas.append(klas)
            else:
                # Handle jika i melebihi panjang trans
                print(f"Index out of range: trans[{i}]")
        
        return hasiltes, hasilkelas




@app.route('/preprocessingbaru')
def preprocessingbaru():
    # Load data from CSV
    TWEET_DATA = pd.read_csv("data_tweet.csv")
    
    # Drop columns except 'rawContent'
    raw_content_df = TWEET_DATA.drop(columns=[col for col in TWEET_DATA.columns if col != 'rawContent'])
    
    # Create an instance of the Preprocessor class
    preprocessor = Preprocessor()
    
    # Process the DataFrame
    processed_df = preprocessor.preprocess(raw_content_df)
    
    # Adjust the index to start from 1
    processed_df.index = processed_df.index + 1
    
    # Save the preprocessed data to a CSV file
    processed_df.to_csv("Text_Preprocessing.csv", index=True)
    
    # Convert DataFrame to dictionary for JSON response
    processed_data = processed_df.to_dict(orient='records')
    
    return jsonify(processed_data)

@app.route('/combined', methods=['POST'])
def combined_route():
    data = request.json
    data_testing = data.get('rawContent')
    d = data.get('d')
    itermax = data.get('itermax')
    lr = data.get('lr')

    print(data_testing)

    # Load preprocessed data from CSV
    processed_df = pd.read_csv("Text_Preprocessing.csv")
    hasil_preprocessing = processed_df[['rawContent']].values.tolist()
    
    # Create an instance of the Preprocessor class
    preprocessor = Preprocessor()
    
    # Calculate features and various metrics
    fitur = preprocessor.fitur_kata(hasil_preprocessing)
    tf = preprocessor.tf(hasil_preprocessing, fitur)
    wtf = preprocessor.wtf(tf)
    idf = preprocessor.dfidf(tf)
    tfidf = preprocessor.tfidf(wtf, idf)
    
    # Load training data from CSV
    training_data_df = pd.read_csv("data_tweet.csv")
    kelas = training_data_df['kelas'].tolist()
    kelas = [1 if k == 'Positif' else -1 for k in kelas]

    # Transpose TF-IDF matrix
    transposed_tfidf = preprocessor.transpose(tfidf)
    
    # Calculate kernel, hessian, sequential
    kernel_matrix = preprocessor.kernel(tfidf, transposed_tfidf, d)
    hessian_matrix = preprocessor.hessian(kelas, kernel_matrix)
    E, delta_a, alfa = preprocessor.sequential(hessian_matrix, itermax, lr)
    
    # Calculate bias
    bias_value = preprocessor.bias(tfidf, alfa, kelas, d)

    # Preprocess test data
    preprocessed_test_df = preprocessor.preprocess2(data_testing)
    print(preprocessed_test_df)
    
    # Calculate features and various metrics for test data
    test_fitur = preprocessor.fitur_kata(preprocessed_test_df)
    test_tf = preprocessor.tf(preprocessed_test_df, test_fitur)
    test_wtf = preprocessor.wtf(test_tf)
    test_idf = preprocessor.dfidf(test_tf)
    test_tfidf = preprocessor.tfidf(test_wtf, test_idf)
    
    # Transpose TF-IDF matrix for test data
    transposed_test_tfidf = preprocessor.transpose(test_tfidf)
    
    # Perform data testing
    hasiltes, hasilkelas = preprocessor.datatesting(kelas, tfidf, transposed_test_tfidf, d, alfa, bias_value)
    
    response = {
        'E': E,
        'delta_a': delta_a,
        'alfa': alfa,
        'bias': bias_value,
        'hasiltes': hasiltes,
        'hasilkelas': hasilkelas
    }
    
    return jsonify(response)

@app.route('/perhitungan')
def calculation_route():
    # Load preprocessed data from CSV
    processed_df = pd.read_csv("Text_Preprocessing.csv")
    
    # Extract the list of preprocessed tokens
    # hasil_preprocessing = processed_df[['tweet_tokens_stemmed']].values.tolist()
    hasil_preprocessing = processed_df[['rawContent']].values.tolist()
    
    # Create an instance of the Preprocessor class
    preprocessor = Preprocessor()
    
    # Calculate features, tf, wtf, df, idf, tfidf, lexicon weights, and sentiment scores
    fitur = preprocessor.fitur_kata(hasil_preprocessing)
    tf = preprocessor.tf(hasil_preprocessing, fitur)
    wtf = preprocessor.wtf(tf)
    idf = preprocessor.dfidf(tf)
    tfidf = preprocessor.tfidf(wtf, idf)
    
    # Prepare response in JSON format
    response_data = {
        "fitur_kata": fitur,
        "tf": tf,
        "wtf": wtf,
        # "df": df,
        "idf": idf,
        "tfidf": tfidf
    }
    
    return jsonify(response_data)

     # Save the processed data to a new CSV file
    # df_fitur = pd.DataFrame(fitur, columns=['fitur_kata'])
    # df_tf = pd.DataFrame(tf).transpose()
    # df_wtf = pd.DataFrame(wtf).transpose()
    # df_tfidf = pd.DataFrame(tfidf).transpose()
    
    # df_combined = pd.concat([df_fitur, df_tf, df_wtf, df_tfidf], axis=1)
    # df_combined.columns = ['fitur_kata'] + [f'tf_{i}' for i in range(len(df_tf.columns))] + \
    #                       [f'wtf_{i}' for i in range(len(df_wtf.columns))] + \
    #                       [f'tfidf_{i}' for i in range(len(df_tfidf.columns))]
    # df_combined.to_csv("Processed_Features.csv", index=False)

@app.route('/testdatanew', methods=['POST'])
def testdatanew():
    data = request.json
    data_testing = data['data_testing']
    
    processed_df = pd.read_csv("Text_Preprocessing.csv")
    
    # Extract the list of preprocessed tokens
    hasil_preprocessing = processed_df[['rawContent']].values.tolist()
    # hasil_preprocessing = processed_df[['tweet_tokens_stemmed']].values.tolist()
    
    # Create an instance of the Preprocessor class
    preprocessor = Preprocessor()
    
    # Calculate features, tf, wtf, df, idf, tfidf
    fitur = preprocessor.fitur_kata(hasil_preprocessing)
    tf = preprocessor.tf(hasil_preprocessing, fitur)
    wtf = preprocessor.wtf(tf)
    idf = preprocessor.dfidf(tf)
    tfidf = preprocessor.tfidf(wtf, idf)

     # # Load training data from CSV
    training_data_df = pd.read_csv("data_tweet.csv")
    
    # # Extract training class
    kelas = training_data_df['kelas'].tolist()
    # Convert 'Negatif' and 'Positif' to numeric values
    kelas = [1 if k == 'Positif' else -1 for k in kelas]
    
    # Degree, max iterations, and learning rate
    d = data['d']
    itermax = data['itermax']
    lr = data['lr']
    
    # Initialize the preprocessor and preprocess the testing data
    preprocessor = Preprocessor()
    fitur = preprocessor.fitur_kata([data_testing])
    
    # Debugging step: Check fitur
    print("Fitur: ", fitur)
    
    tf_matrix = preprocessor.tf([data_testing], fitur)
    
    # Debugging step: Check tf_matrix
    print("TF Matrix: ", tf_matrix)
    
    wtf_matrix = preprocessor.wtf(tf_matrix)
    
    # Debugging step: Check wtf_matrix
    print("WTF Matrix: ", wtf_matrix)
    
    idf = preprocessor.dfidf(tf_matrix)
    
    # Debugging step: Check df and idf
    # print("DF: ", df)
    print("IDF: ", idf)
    
    tfidf_testing = preprocessor.tfidf(wtf_matrix, idf)
    
    # Debugging step: Check tfidf_testing
    print("TFIDF Testing: ", tfidf_testing)
    
    transposed_tfidf = preprocessor.transpose(tfidf_testing)

    # Debugging shapes
    print("Shape of tfidf: ", tfidf.shape)
    print("Shape of tfidf_testing: ", len(tfidf_testing), len(tfidf_testing[0]) if len(tfidf_testing) > 0 else 0)
    print("Shape of transposed_tfidf: ", len(transposed_tfidf), len(transposed_tfidf[0]) if len(transposed_tfidf) > 0 else 0)
    
    # Ensure shapes are compatible for kernel calculation
    if tfidf.shape[1] != len(transposed_tfidf):
        raise ValueError("Incompatible shapes for tfidf and transposed_tfidf")

    # Reshape tfidf_testing to match the number of columns in tfidf
    tfidf_testing_resized = tfidf_testing + [[0.0]] * (tfidf.shape[1] - len(tfidf_testing))
    
    # Transpose the resized tfidf_testing
    transposed_tfidf_resized = preprocessor.transpose(tfidf_testing_resized)

    # Debugging shapes after resizing
    print("Shape of tfidf_testing_resized: ", len(tfidf_testing_resized), len(tfidf_testing_resized[0]) if len(tfidf_testing_resized) > 0 else 0)
    print("Shape of transposed_tfidf_resized: ", len(transposed_tfidf_resized), len(transposed_tfidf_resized[0]) if len(transposed_tfidf_resized) > 0 else 0)

    mat_kernel = preprocessor.kernel(tfidf.values.tolist(), transposed_tfidf_resized, d)
    
    hessian_matrix = preprocessor.hessian(kelas, mat_kernel)
    E, delta_a, alfa = preprocessor.sequential(hessian_matrix, itermax, lr)
    bias_value = preprocessor.bias(tfidf, alfa, kelas, d)
    hasiltes, hasilkelas = preprocessor.datatesting(kelas, tfidf, transposed_tfidf_resized, d, alfa, bias_value)
    
    return jsonify({
        'hasiltes': hasiltes,
        'hasilkelas': hasilkelas
    })
    
    # # Read and process the tfidf matrix
    # tfidf = pd.read_csv("hasil_vector_matrix.csv", header=None)
    # tfidf = tfidf.apply(pd.to_numeric, errors='coerce').fillna(0)
    
@app.route('/preprocessing')
def preprocessing():
    logger.info("Starting preprocessing")
    
    TWEET_DATA = pd.read_csv("data_tweet.csv")
    logger.info("Data read from CSV")

    # Menghapus kolom selain 'rawContent'
    raw_content_df = TWEET_DATA.drop(columns=[col for col in TWEET_DATA.columns if col != 'rawContent'])
    logger.info("Columns other than 'rawContent' dropped")

    # Case Folding
    raw_content_df['caseFolding'] = raw_content_df['rawContent'].str.lower()
    logger.info("Case folding applied")

    # Cleaning text
    def remove_tweet_special(text):
        text = text.replace('\\t'," ").replace('\\n'," ").replace('\\u'," ").replace('\\',"")
        text = text.encode('ascii', 'replace').decode('ascii')
        text = ' '.join(re.sub("([@#][A-Za-z0-9]+)|(\w+:\/\/\S+)"," ", text).split())
        text = text.replace("http://", " ").replace("https://", " ")
        text = re.sub(r'[.!&,/@#$%^*\-]{1,5}', ' ', text)  # Replace one, two, or three occurrences of . ! & -
         # Menghapus angka
        text = re.sub(r"\d+", "", text)

        # Menghapus tanda baca
        text = text.translate(str.maketrans("", "", string.punctuation))

        # Menghapus spasi berlebih
        text = re.sub(r'\s+', ' ', text).strip()

        # Menghapus karakter tunggal
        text = re.sub(r"\b[a-zA-Z]\b", "", text)
        return text

    raw_content_df['cleaning'] = raw_content_df['caseFolding'].apply(remove_tweet_special)
    logger.info("Special characters removed")

    # Remove number
    def remove_number(text):
        return  re.sub(r"\d+", "", text)

    raw_content_df['cleaning'] = raw_content_df['cleaning'].apply(remove_number)
    logger.info("Numbers removed")

    # Remove punctuation
    def remove_punctuation(text):
        return text.translate(str.maketrans("","",string.punctuation))

    raw_content_df['cleaning'] = raw_content_df['cleaning'].apply(remove_punctuation)
    logger.info("Punctuation removed")

    # Remove whitespace leading & trailing
    def remove_whitespace_LT(text):
        return text.strip()

    raw_content_df['cleaning'] = raw_content_df['cleaning'].apply(remove_whitespace_LT)
    logger.info("Leading and trailing whitespace removed")

    # Remove multiple whitespace into single whitespace
    def remove_whitespace_multiple(text):
        return re.sub('\s+',' ',text)

    raw_content_df['cleaning'] = raw_content_df['cleaning'].apply(remove_whitespace_multiple)
    logger.info("Multiple whitespace reduced to single")

    # Remove single char
    def remove_singl_char(text):
        return re.sub(r"\b[a-zA-Z]\b", "", text)

    raw_content_df['cleaning'] = raw_content_df['cleaning'].apply(remove_singl_char)
    logger.info("Single characters removed")

    # Tokenizing
    def word_tokenize_wrapper(text):
        return word_tokenize(text)

    raw_content_df['tweet_tokens'] = raw_content_df['cleaning'].apply(word_tokenize_wrapper)
    logger.info("Tokenization applied")

    def freqDist_wrapper(text):
        return FreqDist(text)

    raw_content_df['tweet_tokens_fdist'] = raw_content_df['tweet_tokens'].apply(freqDist_wrapper)
    raw_content_df['tweet_tokens_fdist'].head().apply(lambda x : x.most_common())
    logger.info("Frequency distribution calculated")

    # Stopword Removal
    list_stopwords = stopwords.words('indonesian')
    list_stopwords.extend([...])  # Add additional stopwords

    txt_stopword = pd.read_csv("stopwords.txt", names=["stopwords"], header=None)
    stopword_list = txt_stopword["stopwords"].tolist()
    list_stopwords.extend(stopword_list)
    list_stopwords = set(list_stopwords)

    def stopwords_removal(words):
        return [word for word in words if word not in list_stopwords]

    raw_content_df['tweet_tokens_WSW'] = raw_content_df['tweet_tokens'].apply(stopwords_removal)
    logger.info("Stopwords removed")

    # Normalization
    normalization_dict = {}
    with open('normalisasi.txt', 'r') as file:
        for line in file:
            if ':' in line:
                parts = line.strip().replace('"', '').split(': ')
                if len(parts) == 2:
                    slang, normal = parts
                    normalization_dict[slang.strip()] = normal.strip()

    def normalize_text(text):
        return [normalization_dict.get(word, word) for word in text]

    raw_content_df['tweet_tokens_normalized'] = raw_content_df['tweet_tokens_WSW'].apply(normalize_text)
    logger.info("Text normalization applied")
    

    # Stemming
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

    def stemmed_wrapper(term):
        return stemmer.stem(term)

    term_dict = {}
    for document in raw_content_df['tweet_tokens_normalized']:
        for term in document:
            if term not in term_dict:
                term_dict[term] = stemmed_wrapper(term)

    for term in term_dict:
        term_dict[term] = stemmed_wrapper(term)
        
    print(len(term_dict))
    print("------------------------")

    for term in term_dict:
        term_dict[term] = stemmed_wrapper(term)
        print(term,":" ,term_dict[term])

    print(term_dict)
    print("------------------------")

    def get_stemmed_term(document):
        return [term_dict[term] for term in document]

    raw_content_df['tweet_tokens_stemmed'] = raw_content_df['tweet_tokens_normalized'].swifter.apply(get_stemmed_term)
    logger.info("Stemming applied")

    # Ubah indeks agar dimulai dari 1
    raw_content_df.index = raw_content_df.index + 1
    # Save the preprocessed data
    raw_content_df.to_csv("Text_Preprocessing.csv")
    logger.info("Preprocessed data saved to CSV")

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

# API TF-IDF
@app.route('/tf-idf-3')
def tfidf3():
    TWEET_DATA = pd.read_csv("Text_Preprocessing.csv", usecols=["tweet_tokens_stemmed"])
    TWEET_DATA.columns = ["tweet"]


    def convert_text_list(texts):
        texts = ast.literal_eval(texts)
        return [text for text in texts]

    TWEET_DATA["tweet_list"] = TWEET_DATA["tweet"].apply(convert_text_list)

    
    def calc_TF(document):
        TF_dict = {}
        for term in document:
            TF_dict[term] = TF_dict.get(term, 0) + 1
        for term in TF_dict:
            TF_dict[term] /= len(document)
        return TF_dict

    TWEET_DATA["TF_dict"] = TWEET_DATA['tweet_list'].apply(calc_TF)

    def calc_DF(tfDict):
        count_DF = {}
        for document in tfDict:
            for term in document:
                count_DF[term] = count_DF.get(term, 0) + 1
        return count_DF

    DF = calc_DF(TWEET_DATA["TF_dict"])

    def calc_IDF(n_document, DF):
        IDF_Dict = {}
        for term in DF:
            IDF_Dict[term] = np.log(n_document / (DF[term] + 1))
        return IDF_Dict

    n_document = len(TWEET_DATA)
    IDF = calc_IDF(n_document, DF)
    
    def calc_TF_IDF(TF):
        TF_IDF_Dict = {}
        for term in TF:
            TF_IDF_Dict[term] = TF[term] * IDF.get(term, 0)
        return TF_IDF_Dict

    TWEET_DATA["TF-IDF_dict"] = TWEET_DATA["TF_dict"].apply(calc_TF_IDF)

    sorted_DF = sorted(DF.items(), key=lambda kv: kv[1], reverse=True)[:100]
    unique_term = [item[0] for item in sorted_DF]

    def calc_TF_IDF_Vec(TF_IDF_Dict):
        TF_IDF_vector = [0.0] * len(unique_term)
        for i, term in enumerate(unique_term):
            if term in TF_IDF_Dict:
                TF_IDF_vector[i] = TF_IDF_Dict[term]
        return TF_IDF_vector

    TWEET_DATA["TF_IDF_Vec"] = TWEET_DATA["TF-IDF_dict"].apply(calc_TF_IDF_Vec)

    TF_IDF_Vec_List = np.array(TWEET_DATA["TF_IDF_Vec"].to_list())
    sums = TF_IDF_Vec_List.sum(axis=0)

    data = [(term, sums[i]) for i, term in enumerate(unique_term)]
    ranking = pd.DataFrame(data, columns=['term', 'rank']).sort_values('rank', ascending=False)

    def join_text_list(texts):
        texts = ast.literal_eval(texts)
        return ' '.join([text for text in texts])

    TWEET_DATA["tweet_join"] = TWEET_DATA["tweet"].apply(join_text_list)

    max_features = 100

    cvect = CountVectorizer(max_features=max_features)
    TF_vector = cvect.fit_transform(TWEET_DATA["tweet_join"])
    normalized_TF_vector = normalize(TF_vector, norm='l1', axis=1)

    tfidf = TfidfVectorizer(max_features=max_features, smooth_idf=False)
    tfs = tfidf.fit_transform(TWEET_DATA["tweet_join"])
    IDF_vector = tfidf.idf_

    tfidf_mat = normalized_TF_vector.multiply(IDF_vector).toarray()

    cvect = CountVectorizer(max_features=max_features, ngram_range=(1, 3))
    counts = cvect.fit_transform(TWEET_DATA["tweet_join"])
    normalized_counts = normalize(counts, norm='l1', axis=1)

    tfidf = TfidfVectorizer(max_features=max_features, ngram_range=(1, 3), smooth_idf=False)
    tfs = tfidf.fit_transform(TWEET_DATA["tweet_join"])

    tfidf_mat = normalized_counts.multiply(IDF_vector).toarray()
    dfbtf = pd.DataFrame(data=tfidf_mat, columns=tfidf.get_feature_names_out())

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
    # data.loc[data.index.isin(index_zero_status), 'status'] = 0
    # data.loc[~data.index.isin(index_zero_status), 'status'] = 1
    
    # data['aktual'] = sentiments
    # data['aktual'] = data1['status'].apply(lambda x: 'negatif' if x in [1, 2] else 'positif')
    data['status'] = data1['status']
    data['label'] = data1['status'].apply(lambda x: 'negatif' if x in [1, 2] else 'positif')
    # data['sentimen'] = sentiments
    # data.loc[data['status'] == 0, 'aktual'] = '-'
    
    data.to_csv("pelabelan.csv", index=False)

    # Tampilkan dataframe hasil pelabelan
    print(data)
    last_row = data.tail(1)
    data = last_row.to_dict(orient='records')
    return jsonify(data)

# @app.route('/tfidf-and-sentiment-labeling', methods=['GET'])
# def tfidf_and_sentiment_labeling():
#     # Baca file teks yang telah dipreprocessing
#     TWEET_DATA = pd.read_csv("Text_Preprocessing.csv", usecols=["tweet_tokens_stemmed"])
#     TWEET_DATA.columns = ["tweet"]

#     # Fungsi untuk mengubah teks dalam format list
#     def convert_text_list(texts):
#         texts = ast.literal_eval(texts)
#         return [text for text in texts]

#     # Aplikasikan fungsi pada kolom tweet
#     TWEET_DATA["tweet_list"] = TWEET_DATA["tweet"].apply(convert_text_list)

#     # Fungsi untuk menghitung Term Frequency (TF)
#     def calc_TF(document):
#         TF_dict = {}
#         for term in document:
#             TF_dict[term] = TF_dict.get(term, 0) + 1
#         for term in TF_dict:
#             TF_dict[term] /= len(document)
#         return TF_dict

#     # Hitung TF untuk setiap dokumen
#     TWEET_DATA["TF_dict"] = TWEET_DATA['tweet_list'].apply(calc_TF)

#     # Fungsi untuk menghitung Document Frequency (DF)
#     def calc_DF(tfDict):
#         count_DF = {}
#         for document in tfDict:
#             for term in document:
#                 count_DF[term] = count_DF.get(term, 0) + 1
#         return count_DF

#     # Hitung DF dari TF_dict
#     DF = calc_DF(TWEET_DATA["TF_dict"])

#     # Fungsi untuk menghitung Inverse Document Frequency (IDF)
#     def calc_IDF(n_document, DF):
#         IDF_Dict = {}
#         for term in DF:
#             IDF_Dict[term] = np.log(n_document / (DF[term] + 1))
#         return IDF_Dict

#     # Hitung IDF dari semua dokumen
#     n_document = len(TWEET_DATA)
#     IDF = calc_IDF(n_document, DF)

#     # Fungsi untuk menghitung TF-IDF
#     def calc_TF_IDF(TF):
#         TF_IDF_Dict = {}
#         for term in TF:
#             TF_IDF_Dict[term] = TF[term] * IDF.get(term, 0)
#         return TF_IDF_Dict

#     # Hitung TF-IDF untuk setiap dokumen
#     TWEET_DATA["TF-IDF_dict"] = TWEET_DATA["TF_dict"].apply(calc_TF_IDF)

#     # Sort DF untuk mendapatkan kata-kata dengan frekuensi tinggi
#     sorted_DF = sorted(DF.items(), key=lambda kv: kv[1], reverse=True)[:100]
#     unique_term = [item[0] for item in sorted_DF]

#     # Fungsi untuk menghasilkan vektor TF-IDF dari TF-IDF dict
#     def calc_TF_IDF_Vec(TF_IDF_Dict):
#         TF_IDF_vector = [0.0] * len(unique_term)
#         for i, term in enumerate(unique_term):
#             if term in TF_IDF_Dict:
#                 TF_IDF_vector[i] = TF_IDF_Dict[term]
#         return TF_IDF_vector

#     # Hitung vektor TF-IDF untuk setiap dokumen
#     TWEET_DATA["TF_IDF_Vec"] = TWEET_DATA["TF-IDF_dict"].apply(calc_TF_IDF_Vec)

#     # Normalisasi vektor TF untuk mendapatkan matriks TF-IDF
#     TF_IDF_Vec_List = np.array(TWEET_DATA["TF_IDF_Vec"].to_list())
#     sums = TF_IDF_Vec_List.sum(axis=0)
#     data = [(term, sums[i]) for i, term in enumerate(unique_term)]
#     ranking = pd.DataFrame(data, columns=['term', 'rank']).sort_values('rank', ascending=False)

#     # Fungsi untuk menggabungkan teks dari list
#     def join_text_list(texts):
#         texts = ast.literal_eval(texts)
#         return ' '.join([text for text in texts])

#     # Gabungkan teks dari list
#     TWEET_DATA["tweet_join"] = TWEET_DATA["tweet"].apply(join_text_list)

#     # Inisialisasi CountVectorizer dan hitung vektor TF
#     max_features = 100
#     cvect = CountVectorizer(max_features=max_features)
#     TF_vector = cvect.fit_transform(TWEET_DATA["tweet_join"])
#     normalized_TF_vector = normalize(TF_vector, norm='l1', axis=1)

#     # Inisialisasi TfidfVectorizer dan hitung matriks TF-IDF
#     tfidf = TfidfVectorizer(max_features=max_features, smooth_idf=False)
#     tfs = tfidf.fit_transform(TWEET_DATA["tweet_join"])
#     IDF_vector = tfidf.idf_

#     # Normalisasi dan hitung matriks TF-IDF dengan n-grams
#     cvect_ngrams = CountVectorizer(max_features=max_features, ngram_range=(1, 3))
#     counts_ngrams = cvect_ngrams.fit_transform(TWEET_DATA["tweet_join"])
#     normalized_counts_ngrams = normalize(counts_ngrams, norm='l1', axis=1)

#     print("normalized_counts_ngrams shape:", normalized_counts_ngrams.shape)
#     print("IDF_vector shape:", IDF_vector.shape)
    
#     # Ubah bentuk IDF_vector jika perlu
#     if IDF_vector.shape[0] != normalized_counts_ngrams.shape[1]:
#         IDF_vector = np.reshape(IDF_vector, (1, -1))

#     print("IDF_vector reshaped shape:", IDF_vector.shape)
    
#      # Hitung tfidf_mat_ngrams
#     tfidf_mat_ngrams = normalized_counts_ngrams.multiply(IDF_vector).toarray()


#     tfidf_ngrams = TfidfVectorizer(max_features=max_features, ngram_range=(1, 3), smooth_idf=False)
#     tfs_ngrams = tfidf_ngrams.fit_transform(TWEET_DATA["tweet_join"])
#     tfidf_mat_ngrams = normalized_counts_ngrams.multiply(IDF_vector)

#     # Simpan matriks TF-IDF ke dalam file
#     # dfbtf = pd.DataFrame(data=tfidf_mat_ngrams.toarray(), columns=tfidf_ngrams.get_feature_names_out())
#     dfbtf = pd.DataFrame(data=tfidf_mat_ngrams, columns=tfidf_ngrams.get_feature_names_out())
#     dfbtf.to_csv("hasil_vector_matrix.csv", index=False)

#     # Gabungkan hasil labeling dengan data tweet asli
#     data_tweet = pd.read_csv('data_tweet.csv')

#     dfbtf['status'] = data_tweet['status']
#     dfbtf['label'] = data_tweet['status'].apply(lambda x: 'negatif' if x in [1, 2] else 'positif')
#     # Simpan hasil pelabelan ke dalam file
#     dfbtf.to_csv("pelabelan.csv", index=False)

#     # Ambil data terakhir untuk respons API
#     last_row_data = dfbtf.tail(1).to_dict(orient='records')

#     return jsonify(last_row_data)


# @app.route('/tfidf-and-sentiment-labeling', methods=['GET'])
# def tfidf_and_sentiment_labeling():
#     try:
#         # Baca file teks yang telah dipreprocessing
#         TWEET_DATA = pd.read_csv("Text_Preprocessing.csv", usecols=["tweet_tokens_stemmed"])
#         # TWEET_DATA = pd.read_csv("Text_Preprocessing.csv", usecols=["rawContent"])
        
#         TWEET_DATA.columns = ["tweet"]

#         logging.debug("Loaded TWEET_DATA: %s", TWEET_DATA.head())

#         # Fungsi untuk mengubah teks dalam format list
#         def convert_text_list(texts):
#             return ast.literal_eval(texts)

#         # Aplikasikan fungsi pada kolom tweet
#         TWEET_DATA["tweet_list"] = TWEET_DATA["tweet"].apply(convert_text_list)

#         logging.debug("Converted text to list: %s", TWEET_DATA["tweet_list"].head())

#         # Fungsi untuk menghitung Term Frequency (TF)
#         def calc_TF(document):
#             TF_dict = {}
#             for term in document:
#                 TF_dict[term] = TF_dict.get(term, 0) + 1
#             for term in TF_dict:
#                 TF_dict[term] /= len(document)
#             return TF_dict

#         # Hitung TF untuk setiap dokumen
#         TWEET_DATA["TF_dict"] = TWEET_DATA['tweet_list'].apply(calc_TF)

#         logging.debug("Calculated TF_dict: %s", TWEET_DATA["TF_dict"].head())

#         # Fungsi untuk menghitung Document Frequency (DF)
#         def calc_DF(tfDict):
#             count_DF = {}
#             for document in tfDict:
#                 for term in document:
#                     count_DF[term] = count_DF.get(term, 0) + 1
#             return count_DF

#         # Hitung DF dari TF_dict
#         DF = calc_DF(TWEET_DATA["TF_dict"])

#         logging.debug("Calculated DF: %s", DF)

#         # Fungsi untuk menghitung Inverse Document Frequency (IDF)
#         def calc_IDF(n_document, DF):
#             IDF_Dict = {}
#             for term in DF:
#                 IDF_Dict[term] = np.log(n_document / (DF[term] + 1))
#             return IDF_Dict

#         # Hitung IDF dari semua dokumen
#         n_document = len(TWEET_DATA)
#         IDF = calc_IDF(n_document, DF)

#         logging.debug("Calculated IDF: %s", IDF)

#         # Fungsi untuk menghitung TF-IDF
#         def calc_TF_IDF(TF):
#             TF_IDF_Dict = {}
#             for term in TF:
#                 TF_IDF_Dict[term] = TF[term] * IDF.get(term, 0)
#             return TF_IDF_Dict

#         # Hitung TF-IDF untuk setiap dokumen
#         TWEET_DATA["TF-IDF_dict"] = TWEET_DATA["TF_dict"].apply(calc_TF_IDF)

#         logging.debug("Calculated TF-IDF_dict: %s", TWEET_DATA["TF-IDF_dict"].head())
        
#         index = TWEET_DATA.shape[0]

#         print('%20s' % "term", "\t", '%10s' % "TF", "\t", '%20s' % "TF-IDF\n")
#         for key in TWEET_DATA["TF-IDF_dict"][index-1]:
#             print('%20s' % key, "\t", TWEET_DATA["TF_dict"][index-1][key] ,"\t" , TWEET_DATA["TF-IDF_dict"][index-1][key])

#         # Sort DF untuk mendapatkan kata-kata dengan frekuensi tinggi
#         sorted_DF = sorted(DF.items(), key=lambda kv: kv[1], reverse=True)[:100]
#         unique_term = [item[0] for item in sorted_DF]

#         logging.debug("Sorted DF and unique terms: %s", unique_term)

#         # Fungsi untuk menghasilkan vektor TF-IDF dari TF-IDF dict
#         def calc_TF_IDF_Vec(TF_IDF_Dict):
#             TF_IDF_vector = [0.0] * len(unique_term)
#             for i, term in enumerate(unique_term):
#                 if term in TF_IDF_Dict:
#                     TF_IDF_vector[i] = TF_IDF_Dict[term]
#             return TF_IDF_vector

#         # Hitung vektor TF-IDF untuk setiap dokumen
#         TWEET_DATA["TF_IDF_Vec"] = TWEET_DATA["TF-IDF_dict"].apply(calc_TF_IDF_Vec)

#         logging.debug("Calculated TF_IDF_Vec: %s", TWEET_DATA["TF_IDF_Vec"].head())

#         # Normalisasi vektor TF untuk mendapatkan matriks TF-IDF
#         TF_IDF_Vec_List = np.array(TWEET_DATA["TF_IDF_Vec"].to_list())
#         sums = TF_IDF_Vec_List.sum(axis=0)
#         data = [(term, sums[i]) for i, term in enumerate(unique_term)]
#         ranking = pd.DataFrame(data, columns=['term', 'rank']).sort_values('rank', ascending=False)

#         logging.debug("Normalized TF vectors and ranking: %s", ranking.head())

#         # Fungsi untuk menggabungkan teks dari list
#         def join_text_list(texts):
#             return ' '.join(ast.literal_eval(texts))

#         # Gabungkan teks dari list
#         TWEET_DATA["tweet_join"] = TWEET_DATA["tweet"].apply(join_text_list)

#         logging.debug("Joined text lists: %s", TWEET_DATA["tweet_join"].head())

#         # max_features = request.args.get('max_features', default=100, type=int)
#         # Inisialisasi CountVectorizer dan hitung vektor TF
#         # max_features = 50
#         max_features = 100
#         cvect = CountVectorizer(max_features=max_features)
#         TF_vector = cvect.fit_transform(TWEET_DATA["tweet_join"])
#         normalized_TF_vector = normalize(TF_vector, norm='l1', axis=1)

#         logging.debug("Calculated normalized TF vector shape: %s", normalized_TF_vector.shape)

#         # Inisialisasi TfidfVectorizer dan hitung matriks TF-IDF
#         tfidf = TfidfVectorizer(max_features=max_features, smooth_idf=False)
#         tfs = tfidf.fit_transform(TWEET_DATA["tweet_join"])
#         IDF_vector = tfidf.idf_

#         logging.debug("Calculated IDF vector shape: %s", IDF_vector.shape)

#         # Normalisasi dan hitung matriks TF-IDF dengan n-grams
#         cvect_ngrams = CountVectorizer(max_features=max_features, ngram_range=(1, 3))
#         counts_ngrams = cvect_ngrams.fit_transform(TWEET_DATA["tweet_join"])
#         normalized_counts_ngrams = normalize(counts_ngrams, norm='l1', axis=1)

#         logging.debug("Calculated normalized counts n-grams shape: %s", normalized_counts_ngrams.shape)
        
#         # Ubah bentuk IDF_vector jika perlu
#         # if IDF_vector.shape[0] != normalized_counts_ngrams.shape[1]:
#         IDF_vector = np.reshape(IDF_vector, (1, -1))

#         logging.debug("IDF_vector reshaped shape: %s", IDF_vector.shape)
        
#         # Hitung tfidf_mat_ngrams
#         tfidf_mat_ngrams = normalized_counts_ngrams.multiply(IDF_vector).toarray()

#         tfidf_ngrams = TfidfVectorizer(max_features=max_features, ngram_range=(1, 3), smooth_idf=False)
#         tfs_ngrams = tfidf_ngrams.fit_transform(TWEET_DATA["tweet_join"])
#         tfidf_mat_ngrams = normalized_counts_ngrams.multiply(IDF_vector)
        
#         type(counts_ngrams)
#         counts_ngrams.shape
        
#         print(tfidf.vocabulary_)
        
#         print(tfidf.get_feature_names_out())
#         a=tfidf.get_feature_names_out()
        
#         print(tfs.toarray())
#         b=tfs.toarray()

#         logging.debug("Calculated tfidf_mat_ngrams shape: %s", tfidf_mat_ngrams.shape)

#         # Simpan matriks TF-IDF ke dalam file
#         dfbtf = pd.DataFrame(data=tfidf_mat_ngrams.toarray(), columns=tfidf_ngrams.get_feature_names_out())
#         dfbtf.to_csv("hasil_vector_matrix.csv", index=False)

#         logging.debug("Saved hasil_vector_matrix.csv")

#         # Gabungkan hasil labeling dengan data tweet asli
#         data_tweet = pd.read_csv('data_tweet.csv')

#         dfbtf['status'] = data_tweet['status']
#         dfbtf['label'] = data_tweet['status'].apply(lambda x: 'negatif' if x in [1, 2] else 'positif')
#         # Simpan hasil pelabelan ke dalam file
#         dfbtf.to_csv("pelabelan.csv", index=False)

#         logging.debug("Saved pelabelan.csv")

#         # Ambil data terakhir untuk respons API
#         last_row_data = dfbtf.tail(1).to_dict(orient='records')

#         return jsonify(last_row_data)
    
#     except Exception as e:
#         logging.error("Error occurred: %s", e)
#         return jsonify({"error": str(e)}), 500

@app.route('/tfidf-and-sentiment-labeling', methods=['GET'])
def tfidf_and_sentiment_labeling():
    try:
        # Baca file teks yang telah dipreprocessing
        TWEET_DATA = pd.read_csv("Text_Preprocessing.csv", usecols=["tweet_tokens_stemmed"])
        TWEET_DATA.columns = ["tweet_tokens_stemmed"]

        # Join token lists into single strings
        TWEET_DATA["tweet_join"] = TWEET_DATA["tweet_tokens_stemmed"].apply(lambda x: ' '.join(eval(x)))

        # Initialize CountVectorizer and fit-transform to get term frequencies
        cvect = CountVectorizer(max_features=100, lowercase=False)
        TF_vector = cvect.fit_transform(TWEET_DATA["tweet_join"])

        # Initialize TfidfTransformer and fit-transform to get TF-IDF values
        tfidf_transformer = TfidfTransformer(smooth_idf=False)
        tfidf_matrix = tfidf_transformer.fit_transform(TF_vector)

        # Normalize TF-IDF matrix
        normalized_tfidf_matrix = normalize(tfidf_matrix, norm='l1', axis=1)

        # Create DataFrame to store TF-IDF results
        dfbtf = pd.DataFrame(normalized_tfidf_matrix.toarray(), columns=cvect.get_feature_names_out())

        # Save TF-IDF matrix to CSV
        dfbtf.to_csv("hasil_vector_matrix.csv", index=False)
        logging.debug("Saved hasil_vector_matrix.csv")

        # Gabungkan hasil labeling dengan data tweet asli
        data_tweet = pd.read_csv('data_tweet.csv')
        dfbtf['status'] = data_tweet['status']
        dfbtf['label'] = data_tweet['status'].apply(lambda x: 'negatif' if x in [1, 2] else 'positif')

        # Simpan hasil pelabelan ke dalam file
        dfbtf.to_csv("pelabelan.csv", index=False)
        logging.debug("Saved pelabelan.csv")

        # Ambil data terakhir untuk respons API
        last_row_data = dfbtf.tail(1).to_dict(orient='records')
        return jsonify(last_row_data)
    
    except Exception as e:
        logging.error("Error occurred: %s", e)
        return jsonify({"error": str(e)}), 500


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


