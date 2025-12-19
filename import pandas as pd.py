import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import re
import nltk
from nltk.stem import WordNetLemmatizer
import joblib

# =========================================================
# 1. KONFIGURASI DAN PEMUATAN DATA
# =========================================================
FILE_NAME = 'spam.csv'
ENCODING_TYPE = 'latin-1'

print("=========================================")
print("Memuat Dataset Spam...")
print("=========================================")

try:
    df = pd.read_csv(FILE_NAME, encoding=ENCODING_TYPE)

    # Ambil dua kolom utama
    df = df.iloc[:, [0, 1]]
    df.columns = ['Label', 'Pesan']

    # Konversi label
    df['Label'] = df['Label'].map({'ham': 0, 'spam': 1})

    print(f"✅ Data berhasil dimuat. Total data: {len(df)}")
    print(f"Jumlah Spam: {len(df[df['Label'] == 1])}")

except FileNotFoundError:
    print(f"❌ File '{FILE_NAME}' tidak ditemukan.")
    exit()

# =========================================================
# 2. PREPROCESSING TEKS (Orang 1)
# =========================================================
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = " ".join([lemmatizer.lemmatize(word) for word in text.split()])
    return text

df['Pesan'] = df['Pesan'].apply(clean_text)
