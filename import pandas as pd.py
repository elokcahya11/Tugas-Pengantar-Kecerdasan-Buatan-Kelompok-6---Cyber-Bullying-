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

# =========================================================
# 3. FEATURE EXTRACTION & SPLIT DATA
# =========================================================
vectorizer = CountVectorizer(
    stop_words='english',
    lowercase=True,
    ngram_range=(1, 2)
)

X_train, X_test, y_train, y_test = train_test_split(
    df['Pesan'], df['Label'],
    test_size=0.2,
    random_state=42
)

X_train_features = vectorizer.fit_transform(X_train)
X_test_features = vectorizer.transform(X_test)

# =========================================================
# 4. PELATIHAN & OPTIMASI MODEL (Orang 2)
# =========================================================
print("\n=========================================")
print("Pelatihan Model Naïve Bayes...")
print("=========================================")

params = {"alpha": [0.1, 0.3, 0.5, 1.0]}
grid = GridSearchCV(
    MultinomialNB(),
    params,
    cv=5,
    scoring="accuracy"
)

grid.fit(X_train_features, y_train)
model = grid.best_estimator_

print("Model terbaik:", grid.best_params_)

# =========================================================
# 5. EVALUASI MODEL
# =========================================================
y_pred = model.predict(X_test_features)

accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"\nAkurasi Model: {accuracy:.4f} ({accuracy*100:.2f}%)")
print("\nLaporan Klasifikasi:")
print(classification_report(y_test, y_pred, target_names=['AMAN', 'SPAM']))

print("Confusion Matrix:")
print(cm)

# =========================================================
# 6. SIMPAN MODEL (Orang 3)
# =========================================================
joblib.dump(model, "spam_model.pkl")
joblib.dump(vectorizer, "spam_vectorizer.pkl")
print("\nModel dan vectorizer berhasil disimpan.")

# =========================================================
# 7. FUNGSI PREDIKSI PESAN
# =========================================================
def predict_message(message_list):
    """
    Memprediksi list pesan:
    0 = AMAN
    1 = SPAM
    """
    features = vectorizer.transform(message_list)
    predictions = model.predict(features)

    results = []
    for msg, pred in zip(message_list, predictions):
        label = "SPAM" if pred == 1 else "AMAN"
        results.append(f"'{msg[:45]}...' -> {label}")
    return results
