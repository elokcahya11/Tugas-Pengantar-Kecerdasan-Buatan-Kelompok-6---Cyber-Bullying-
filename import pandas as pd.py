import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import re
import nltk

# --- 1. Konfigurasi dan Pemuatan Data ---
FILE_NAME = 'spam.csv'
ENCODING_TYPE = 'latin-1' # Tipe encoding umum untuk dataset ini

print("=========================================")
print("Memuat Dataset Phishing/Spam...")
print("=========================================")

try:
    # Memuat data. Menggunakan 'latin-1' karena ada karakter non-ASCII
    df = pd.read_csv(FILE_NAME, encoding=ENCODING_TYPE)
    
    # Dataset ini seringkali tidak memiliki header standar; kita ambil dua kolom pertama
    df = df.iloc[:, [0, 1]] 
    df.columns = ['Label', 'Pesan'] # Ubah nama kolom
    
    # Konversi label: 'ham' -> 0 (Aman), 'spam' -> 1 (Berbahaya)
    df['Label'] = df['Label'].map({'ham': 0, 'spam': 1})
    
    print(f"✅ Data berhasil dimuat. Total baris: {len(df)}")
    print(f"Jumlah Pesan Berbahaya (1/Spam): {len(df[df['Label'] == 1])}")
    
except FileNotFoundError:
    print(f"❌ ERROR KRITIS: File '{FILE_NAME}' tidak ditemukan.")
    print("Pastikan Anda sudah mengunduh data dan menempatkannya di folder proyek.")
    exit()
except Exception as e:
    print(f"❌ ERROR saat memproses data: {e}")
    exit()

# --- PREPROCESSING TAMBAHAN (Orang 1) ---
import re
import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = " ".join([lemmatizer.lemmatize(word) for word in text.split()])
    return text

df['Pesan'] = df['Pesan'].apply(clean_text)

# --- 2. Pemrosesan Teks (NLP Preprocessing) ---
# CountVectorizer akan mengubah teks menjadi vektor angka (Bag of Words)
# stop_words='english' menghapus kata umum yang tidak membantu klasifikasi (seperti 'the', 'a', 'is').
# ngram_range=(1, 2) menggunakan kata tunggal dan pasangan kata (bi-grams) sebagai fitur.
vectorizer = CountVectorizer(stop_words='english', lowercase=True, ngram_range=(1, 2))

# Memisahkan data menjadi training (80%) dan testing (20%)
X_train, X_test, y_train, y_test = train_test_split(
    df['Pesan'], df['Label'], test_size=0.2, random_state=42
)

# Mengubah data teks training menjadi fitur numerik dan MENGUBAH model vectorizer
X_train_features = vectorizer.fit_transform(X_train)
# Hanya MENGGUNAKAN vectorizer yang sudah dilatih pada data testing
X_test_features = vectorizer.transform(X_test)


# --- 3. Pelatihan Model (Multinomial Naïve Bayes) ---
print("\n=========================================")
print("Memulai Pelatihan Model Naïve Bayes...")
print("=========================================")

# MultinomialNB sangat cocok untuk masalah klasifikasi teks
# --- MODEL OPTIMIZATION (Orang 2) ---
from sklearn.model_selection import GridSearchCV

params = {"alpha": [0.1, 0.3, 0.5, 1.0]}
grid = GridSearchCV(MultinomialNB(), params, cv=5, scoring="accuracy")
grid.fit(X_train_features, y_train)

model = grid.best_estimator_
print("Model terbaik:", grid.best_params_)

# --- 4. Evaluasi Model ---
y_pred = model.predict(X_test_features)
accuracy = accuracy_score(y_test, y_pred)

print(f"✅ Pelatihan Selesai!")
print(f"Akurasi Model pada data uji: {accuracy:.4f} ({accuracy*100:.2f}%)")
print("\nLaporan Klasifikasi:")
print(classification_report(y_test, y_pred, target_names=['Aman (0)', 'Berbahaya (1)']))
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# --- 5. Fungsi Prediksi (Demo) ---
def predict_message(message_list):
    # --- SAVE MODEL DAN CLI (Orang 3) ---
import joblib

joblib.dump(model, "spam_model.pkl")
joblib.dump(vectorizer, "spam_vectorizer.pkl")

def run_cli():
    print("\n=== SISTEM DETEKSI SPAM CLI ===")
    while True:
        msg = input("Masukkan pesan ('exit' untuk keluar): ")
        if msg.lower() == "exit":
            break
        vec = vectorizer.transform([msg])
        pred = model.predict(vec)[0]
        print("Hasil:", "SPAM" if pred == 1 else "AMAN")
    """Mengambil list pesan string dan memprediksi apakah itu aman (0) atau phishing (1)."""
    # Mengubah pesan baru menjadi fitur numerik
    new_features = vectorizer.transform(message_list)
    predictions = model.predict(new_features)
    
    results = []
    for msg, pred in zip(message_list, predictions):
        label = "BERBAHAYA (PHISHING)" if pred == 1 else "AMAN (LEGIT)"
        results.append(f"'{msg[:45]}...' -> KLASIFIKASI: {label}")
    return results

print("\n=========================================")
print("DEMO PENGUJIAN MODEL")
print("=========================================")

test_messages = [
    "Selamat! Anda memenangkan uang 100 juta. Segera klik link ini untuk klaim hadiah.", # Phishing
    "Hi Budi, apakah laporan bulanan sudah siap untuk rapat tim besok pagi?", # Aman
    "URGENT! Your Apple ID has been compromised. Verify your account now at http://suspicious-link.com" # Phishing
]

predictions = predict_message(test_messages)
for result in predictions:

    print(result)


