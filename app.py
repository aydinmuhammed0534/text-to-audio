from flask import Flask, request, render_template, send_file, flash, url_for
import edge_tts
import asyncio
import os
import pdfplumber
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
import pyttsx3
import platform
import time
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import re
from langdetect import detect
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.probability import FreqDist

app = Flask(__name__)
app.secret_key = '0A9M7H13R5'  # Buraya güçlü bir anahtar koyun

# Ses dosyaları için klasör oluştur
UPLOAD_FOLDER = 'audio_files'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# İhtiyaç duyulan NLTK paketlerini indir
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

def get_stop_words(lang):
    """Dile göre stop words listesi döndürür"""
    if lang == 'tr':
        stop_words = set(stopwords.words('turkish'))
        stop_words.update([
            've', 'veya', 'bir', 'bu', 'şu', 'o', 'da', 'de', 'ki', 'ile', 'için',
            'gibi', 'kadar', 'sonra', 'önce', 'daha', 'artık', 'ancak', 'yine'
        ])
    else:  # English
        stop_words = set(stopwords.words('english'))
        stop_words.update([
            'would', 'could', 'should', 'might', 'must', 'need', 'want', 'seem',
            'like', 'also', 'however', 'therefore', 'thus', 'hence'
        ])
    return stop_words

def get_important_words(lang):
    """Dile göre önemli kelimeleri döndürür"""
    if lang == 'tr':
        return {
            'önemli', 'dikkat', 'sonuç', 'örneğin', 'özetle', 'kısacası',
            'böylece', 'dolayısıyla', 'bu nedenle', 'kanıtlamak', 'göstermek'
        }
    else:  # English
        return {
            'important', 'significant', 'result', 'conclusion', 'summary',
            'therefore', 'consequently', 'thus', 'prove', 'demonstrate'
        }

def preprocess_text(text):
    # Gereksiz boşlukları temizle
    text = re.sub(r'\s+', ' ', text)
    # Özel karakterleri temizle
    text = re.sub(r'[^\w\s.]', '', text)
    return text.strip()

def summarize_text(text):
    # Dil tespiti
    try:
        lang = detect(text)
        if lang not in ['tr', 'en']:
            lang = 'en'  # Varsayılan olarak İngilizce
    except:
        lang = 'en'

    # Metni cümlelere ayır
    sentences = sent_tokenize(text)
    if len(sentences) < 5:
        return text, lang

    # Stop words ve önemli kelimeleri al
    stop_words = get_stop_words(lang)
    important_words = get_important_words(lang)

    # Kelime frekanslarını hesapla
    words = word_tokenize(text.lower())
    words = [word for word in words if word not in stop_words]
    freq_dist = FreqDist(words)

    # Cümle puanlarını hesapla
    sentence_scores = {}
    for i, sentence in enumerate(sentences):
        words = word_tokenize(sentence.lower())
        words = [word for word in words if word not in stop_words]

        if not words:
            continue

        # Temel puan hesaplama
        score = sum(freq_dist[word] for word in words)

        # Önemli kelime bonusu
        important_word_count = sum(1 for word in words if word in important_words)
        score += important_word_count * 2

        # Pozisyon bazlı ağırlıklandırma
        if i == 0:  # İlk cümle
            score *= 1.5
        elif i == len(sentences) - 1:  # Son cümle
            score *= 1.3
        elif i < len(sentences) * 0.2:  # İlk %20
            score *= 1.2
        elif i > len(sentences) * 0.8:  # Son %20
            score *= 1.1

        # Uzunluk normalizasyonu
        score = score / (len(words) + 1)
        sentence_scores[sentence] = score

    # Özet uzunluğunu belirle
    text_length = len(text.split())
    if text_length > 3000:
        summary_ratio = 0.25
    elif text_length > 2000:
        summary_ratio = 0.30
    elif text_length > 1000:
        summary_ratio = 0.35
    elif text_length > 500:
        summary_ratio = 0.40
    else:
        summary_ratio = 0.50

    # Cümle sayısını belirle
    target_sentences = min(max(int(len(sentences) * summary_ratio), 10), 50)

    # Özeti oluştur
    summary_sentences = sorted(sentence_scores.items(), 
                             key=lambda x: x[1], 
                             reverse=True)[:target_sentences]
    
    summary = ' '.join(sentence for sentence, score in sorted(summary_sentences, 
                      key=lambda x: sentences.index(x[0])))

    return summary, lang

@app.route('/')
def index():
    # Ana sayfada varsayılan dil olarak Türkçe ayarla
    return render_template('index.html', detected_lang='tr')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash("Dosya yüklenmedi. Lütfen bir dosya seçin.")
        return render_template('index.html')

    file = request.files['file']
    voice = request.form.get('voice', 'tr-TR-AhmetNeural')

    try:
        # Dosya okuma
        if file.filename.endswith('.pdf'):
            text = ''
            with pdfplumber.open(file) as pdf:
                for page in pdf.pages:
                    try:
                        text += page.extract_text() + "\n"
                    except Exception as e:
                        continue
        elif file.filename.endswith('.txt'):
            text = file.read().decode('utf-8')
        else:
            flash("Desteklenmeyen dosya türü. Lütfen .txt veya .pdf dosyası yükleyin.")
            return render_template('index.html')

        # Özet oluştur ve dili tespit et
        summary_text, detected_lang = summarize_text(text)

        # Dile göre ses asistanını seç
        if detected_lang == 'tr':
            voice = request.form.get('voice', 'tr-TR-AhmetNeural')
        else:
            voice = request.form.get('voice', 'en-US-JennyNeural')

        # Ses dosyası oluştur
        async def generate_audio():
            communicate = edge_tts.Communicate(summary_text, voice)
            audio_filename = f"audio_{int(time.time())}.mp3"
            audio_path = os.path.join(UPLOAD_FOLDER, audio_filename)
            await communicate.save(audio_path)
            return audio_filename

        audio_filename = asyncio.run(generate_audio())
        audio_url = url_for('get_audio', filename=audio_filename)

        return render_template('index.html', 
                             summary=summary_text, 
                             audio_file=audio_url,
                             detected_lang=detected_lang)

    except Exception as e:
        print(f"Hata: {str(e)}")
        flash(f"Bir hata oluştu: {str(e)}")
        return render_template('index.html', summary=None, audio_file="")

# Ses dosyalarını serve etmek için route'u güncelle
@app.route('/audio/<filename>')
def get_audio(filename):
    try:
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        if os.path.exists(file_path):
            return send_file(file_path, 
                           mimetype='audio/mpeg',
                           as_attachment=False)
        else:
            flash("Ses dosyası bulunamadı.")
            return render_template('index.html')
    except Exception as e:
        print(f"Ses dosyası gönderme hatası: {str(e)}")
        flash(f"Ses dosyası yüklenirken hata oluştu: {str(e)}")
        return render_template('index.html')

# Eski ses dosyalarını temizle
def cleanup_old_files():
    try:
        for file in os.listdir(UPLOAD_FOLDER):
            file_path = os.path.join(UPLOAD_FOLDER, file)
            # 1 saatten eski dosyaları sil
            if os.path.getmtime(file_path) < time.time() - 3600:
                os.remove(file_path)
    except Exception as e:
        print(f"Temizleme hatası: {str(e)}")  # Debug için

# Uygulama başlangıcında UPLOAD_FOLDER'ı oluştur
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
    print(f"Upload klasörü oluşturuldu: {UPLOAD_FOLDER}")

if __name__ == '__main__':
    app.run(debug=True)