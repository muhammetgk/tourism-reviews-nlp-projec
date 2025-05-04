import os
import re
import json
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# NLTK gerekli dosyaları indirme
try:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    print("NLTK kaynakları başarıyla indirildi.")
except Exception as e:
    print(f"NLTK yüklemesinde hata: {e}")


class TextProcessor:
    """
    Metinleri işleyen sınıf - Lemmatization, Stemming ve diğer ön işleme adımlarını gerçekleştirir
    """
    def __init__(self, language='english'):
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words(language))
    
    def preprocess_text(self, text):
        """
        Metni temizleme ve normalleştirme
        - Küçük harfe çevirme
        - Alfanumerik olmayan karakterleri temizleme
        - Stopword'leri çıkarma
        - Lemmatization ve Stemming uygulanması
        """
        if not isinstance(text, str):
            text = str(text)
            
        # Küçük harfe çevirme
        text = text.lower()
        
        # Alfanumerik olmayan karakterleri kaldırma
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        
        # Fazla boşlukları temizleme
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Cümlelere ayırma
        try:
            sentences = sent_tokenize(text)
        except:
            # Hata durumunda basit nokta ile ayırma
            sentences = text.split('.')
            sentences = [s.strip() for s in sentences if s.strip()]
        
        # Her cümle için tokenizasyon, lemmatizasyon ve stemming
        lemmatized_sentences = []
        stemmed_sentences = []
        
        for sentence in sentences:
            try:
                tokens = word_tokenize(sentence)
            except:
                # Hata durumunda basit boşluk ile ayırma
                tokens = sentence.split()
            
            # Sadece harf olan kelimeleri al ve stopword'leri çıkar
            filtered_tokens = [token for token in tokens if token.isalpha() and token.lower() not in self.stop_words]
            
            # Lemmatization
            lemmatized_tokens = [self.lemmatizer.lemmatize(token) for token in filtered_tokens]
            lemmatized_sentences.append(lemmatized_tokens)
            
            # Stemming
            stemmed_tokens = [self.stemmer.stem(token) for token in filtered_tokens]
            stemmed_sentences.append(stemmed_tokens)
        
        # Düz metin olarak birleştirme
        lemmatized_text = ' '.join([' '.join(tokens) for tokens in lemmatized_sentences])
        stemmed_text = ' '.join([' '.join(tokens) for tokens in stemmed_sentences])
        
        return {
            'lemmatized_text': lemmatized_text,
            'stemmed_text': stemmed_text,
            'tokenized_lemmatized': lemmatized_sentences,
            'tokenized_stemmed': stemmed_sentences
        }


class CategoryClassifier:
    """Metin içeriğine göre kategori belirleme sınıfı"""
    def __init__(self):
        # Kategori için anahtar kelimeler
        self.category_keywords = {
            'hotel': ["hotel", "room", "stay", "accommodation", "lobby", "reception", "check-in", 
                     "checkout", "bed", "suite", "motel", "inn", "breakfast", "concierge", "staff"],
            
            'restaurant': ["restaurant", "food", "meal", "dish", "dinner", "lunch", "menu", "chef", 
                          "taste", "delicious", "waiter", "cafe", "bistro", "dine", "cuisine", "eat"],
            
            'museum': ["museum", "exhibit", "gallery", "art", "collection", "history", "artifact", 
                      "display", "tour guide", "exhibition", "painting", "sculpture", "heritage"],
            
            'attraction': ["attraction", "visit", "tour", "experience", "sight", "landmark", "tourist", 
                          "monument", "historic", "architecture", "castle", "palace", "cathedral"],
            
            'park': ["park", "garden", "nature", "walk", "trail", "outdoor", "tree", "lake", "green", 
                    "forest", "hiking", "picnic", "botanical", "wildlife", "scenic"]
        }
    
    def classify_text(self, text):
        """Metin içeriğindeki anahtar kelimelere göre kategori belirleme"""
        if not isinstance(text, str):
            text = str(text)
            
        text = text.lower()
        
        # Her kategorinin anahtar kelimelerinin ne kadar geçtiğini sayma
        category_scores = {}
        
        for category, keywords in self.category_keywords.items():
            score = sum(1 for word in keywords if word in text)
            category_scores[category] = score
        
        # En yüksek skora sahip kategoriyi seçme
        top_category = max(category_scores.items(), key=lambda x: x[1])
        
        # Eğer hiç anahtar kelime eşleşmezse "other" olarak işaretle
        if top_category[1] == 0:
            return "other"
        else:
            return top_category[0]
    
    def get_all_category_scores(self, text):
        """Tüm kategoriler için skor hesaplama"""
        if not isinstance(text, str):
            text = str(text)
            
        text = text.lower()
        
        # Her kategorinin anahtar kelimelerinin ne kadar geçtiğini sayma
        category_scores = {}
        
        for category, keywords in self.category_keywords.items():
            score = sum(1 for word in keywords if word in text)
            category_scores[category] = score
        
        return category_scores


def zipf_analizi_ciz(metin, baslik, output_dir="zipf_graphs"):
    """
    Verilen metin için Zipf yasası analizini yaparak log-log grafiğini çizer
    
    Parameters:
    -----------
    metin: str
        Analiz edilecek metin
    baslik: str
        Grafiğin başlığı
    output_dir: str, default="zipf_graphs"
        Grafiklerin kaydedileceği dizin
        
    Returns:
    --------
    tuple
        (kelimeler, sıralar, frekanslar) içeren tuple
    """
    # Çıktı dizinini oluşturma
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Kelime frekanslarını say
    kelime_sayimlari = Counter(metin.split())
    
    # Sıraları ve frekansları al
    kelimeler = sorted(kelime_sayimlari.items(), key=lambda x: x[1], reverse=True)
    siralar = np.arange(1, len(kelimeler) + 1)
    frekanslar = [sayi for kelime, sayi in kelimeler]
    
    # Log-log grafiği çiz
    plt.figure(figsize=(10, 6))
    plt.loglog(siralar, frekanslar, marker='.', linestyle='none', alpha=0.5)
    
    # Regresyon çizgisi ekle
    plt.loglog(siralar, [frekanslar[0]/r for r in siralar], linestyle='-', color='r', 
              label='Zipf Yasası (1/rank)')
    
    # Etiket ve başlık ekle
    plt.xlabel('Sıra (log ölçeği)')
    plt.ylabel('Frekans (log ölçeği)')
    plt.title(f'Zipf Dağılımı: {baslik}')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Grafiği kaydet
    plt.savefig(os.path.join(output_dir, f'zipf_{baslik.lower().replace(" ", "_")}.png'))
    plt.close()
    
    print(f"{baslik} için Zipf analizi grafiği kaydedildi.")
    
    return kelimeler, siralar, frekanslar


class DataProcessor:
    """Veri işleme sınıfı - Yelp veri setini okuma, temizleme ve kaydetme"""
    def __init__(self, input_file_path=None, output_dir="processed_data", max_file_size_mb=50, max_records=None):
        self.input_file_path = input_file_path or "C:\\Users\\gokmu\\Desktop\\ddisleme\\yelp_academic_dataset_review.json"
        self.output_dir = output_dir
        self.max_file_size_mb = max_file_size_mb
        self.max_records = max_records
        
        # Çıktı dosya yollarını belirle
        self.output_file_path = os.path.join(output_dir, "processed_reviews.csv")
        self.lemmatized_csv = os.path.join(output_dir, "processed_reviews_lemmatized.csv")
        self.stemmed_csv = os.path.join(output_dir, "processed_reviews_stemmed.csv")
        
        # Zipf grafikleri için dizin
        self.zipf_dir = "zipf_graphs"
        if not os.path.exists(self.zipf_dir):
            os.makedirs(self.zipf_dir)
        
        # Text processor ve category classifier
        self.text_processor = TextProcessor()
        self.category_classifier = CategoryClassifier()
        
        # Çıktı dizinini oluştur
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def process_data(self):
        """Veri setini işleme"""
        # Dosyanın var olup olmadığını kontrol et
        if not os.path.exists(self.input_file_path):
            print(f"Hata: '{self.input_file_path}' dosyası bulunamadı.")
            return False
        
        print(f"Yelp veri seti işleniyor... Maksimum dosya boyutu: {self.max_file_size_mb} MB")
        
        # İşlenmiş veriyi saklayacak liste
        processed_data = []
        record_count = 0
        file_size_bytes = 0
        
        # Yaklaşık dosya boyutunu kontrol etmek için
        bytes_per_mb = 1024 * 1024
        max_file_size_bytes = self.max_file_size_mb * bytes_per_mb
        
        # CSV dosyası için sütunlar
        columns = ['title', 'review_text', 'category', 'lemmatized_text', 'stemmed_text', 'tokenized_lemmatized', 'tokenized_stemmed']
        
        # Ham veri için zipf analizi
        raw_text_sample = ""
        
        # CSV dosyasını açıp başlık satırını yaz
        with open(self.output_file_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = pd.DataFrame(columns=columns).to_csv(csvfile, index=False)
            
            # JSON dosyasını satır satır oku
            with open(self.input_file_path, 'r', encoding='utf-8') as jsonfile:
                for line in tqdm(jsonfile, desc="Satırlar işleniyor"):
                    try:
                        # Maksimum kayıt kontrolü
                        if self.max_records is not None and record_count >= self.max_records:
                            break
                        
                        # Maksimum dosya boyutu kontrolü
                        if file_size_bytes >= max_file_size_bytes:
                            print(f"Maksimum dosya boyutuna ulaşıldı: {self.max_file_size_mb} MB")
                            break
                        
                        # JSON satırını parse et
                        review = json.loads(line.strip())
                        
                        # Temel kontroller
                        if 'text' not in review:
                            continue
                        
                        review_text = review['text']
                        
                        # Ham veri örneği için biraz metin al (ilk 50 yorum)
                        if record_count < 50:
                            raw_text_sample += review_text + " "
                        
                        # Çok kısa yorumları atla
                        if len(review_text.split()) < 10:
                            continue
                        
                        # Kategoriyi belirle
                        category = self.category_classifier.classify_text(review_text)
                        
                        # "other" kategorisine düşen yorumları atla
                        if category == "other":
                            continue
                        
                        # Metni işle
                        processed = self.text_processor.preprocess_text(review_text)
                        
                        # Başlık oluştur (yoksa)
                        title = review.get('business_id', f"{category.capitalize()} Review {record_count}")
                        
                        # Yeni satır oluştur
                        row = {
                            'title': title,
                            'review_text': review_text,
                            'category': category,
                            'lemmatized_text': processed['lemmatized_text'],
                            'stemmed_text': processed['stemmed_text'],
                            'tokenized_lemmatized': str(processed['tokenized_lemmatized']),
                            'tokenized_stemmed': str(processed['tokenized_stemmed'])
                        }
                        
                        # Listeye ekle
                        processed_data.append(row)
                        
                        # Satır sayısını ve dosya boyutunu güncelle
                        record_count += 1
                        # Tahmini boyut hesaplama (kabaca)
                        row_size = len(str(row).encode('utf-8'))
                        file_size_bytes += row_size
                        
                        # Her 1000 satırda bir durum bilgisi ver
                        if record_count % 1000 == 0:
                            print(f"{record_count} satır işlendi, yaklaşık dosya boyutu: {file_size_bytes / bytes_per_mb:.2f} MB")
                        
                    except json.JSONDecodeError:
                        continue
                    except Exception as e:
                        print(f"Satır işlenirken hata: {e}")
                        continue
        
        # İşlenmiş veriyi DataFrame'e dönüştür
        df = pd.DataFrame(processed_data)
        
        # CSV olarak kaydet
        df.to_csv(self.output_file_path, index=False)
        
        # Zipf Analizi - Ham Veri
        if raw_text_sample:
            zipf_analizi_ciz(raw_text_sample, "Ham Veri", self.zipf_dir)
        
        # Zipf Analizi - İşlenmiş Veri
        all_lemmatized = ' '.join(df['lemmatized_text'])
        all_stemmed = ' '.join(df['stemmed_text'])
        
        zipf_analizi_ciz(all_lemmatized, "Lemmatize Edilmiş Veri", self.zipf_dir)
        zipf_analizi_ciz(all_stemmed, "Stem Edilmiş Veri", self.zipf_dir)
        
        # Zipf karşılaştırma grafiği
        self._create_zipf_comparison_plot(raw_text_sample, all_lemmatized, all_stemmed)
        
        # Kategori dağılımını hesapla
        category_counts = df['category'].value_counts()
        
        print("\nKategorilere göre dağılım:")
        for category, count in category_counts.items():
            print(f"  - {category}: {count} kayıt")
        
        # Kategori dağılımını görselleştir
        self._visualize_category_distribution(df)
        
        print(f"\nToplam {record_count} kayıt işlendi ve '{self.output_file_path}' dosyasına kaydedildi.")
        print(f"Dosya boyutu: {os.path.getsize(self.output_file_path) / bytes_per_mb:.2f} MB")
        
        # Lemmatize ve Stem edilmiş cümleleri ayrı CSV'lere kaydet
        self._save_processed_sentences(df)
        
        # TF-IDF için işlenmiş verileri ayrı CSV'lere kaydet
        df['lemmatized_text'].to_csv(os.path.join(self.output_dir, "tfidf_lemmatized.csv"), index=False, header=True)
        df['stemmed_text'].to_csv(os.path.join(self.output_dir, "tfidf_stemmed.csv"), index=False, header=True)
        
        print(f"TF-IDF için metinler 'tfidf_lemmatized.csv' ve 'tfidf_stemmed.csv' dosyalarına kaydedildi.")
        
        return True
    
    def _save_processed_sentences(self, df):
        """Lemmatize ve Stem edilmiş cümleleri ayrı CSV'lere kaydet"""
        # Lemmatize edilmiş cümleleri kaydet
        with open(self.lemmatized_csv, 'w', newline='', encoding='utf-8') as file:
            writer = pd.DataFrame(columns=['sentence']).to_csv(file, index=False)
            
            for idx, row in df.iterrows():
                try:
                    # String formatındaki listeler gerçek listelere dönüştürülüyor
                    lemmatized_sentences = eval(row['tokenized_lemmatized'])
                    
                    # Her cümleyi ayrı bir satır olarak kaydet
                    for sentence in lemmatized_sentences:
                        if isinstance(sentence, list) and sentence:
                            sentence_text = ' '.join(sentence)
                            pd.DataFrame([{'sentence': sentence_text}]).to_csv(file, index=False, header=False)
                except:
                    continue
        
        # Stem edilmiş cümleleri kaydet
        with open(self.stemmed_csv, 'w', newline='', encoding='utf-8') as file:
            writer = pd.DataFrame(columns=['sentence']).to_csv(file, index=False)
            
            for idx, row in df.iterrows():
                try:
                    # String formatındaki listeler gerçek listelere dönüştürülüyor
                    stemmed_sentences = eval(row['tokenized_stemmed'])
                    
                    # Her cümleyi ayrı bir satır olarak kaydet
                    for sentence in stemmed_sentences:
                        if isinstance(sentence, list) and sentence:
                            sentence_text = ' '.join(sentence)
                            pd.DataFrame([{'sentence': sentence_text}]).to_csv(file, index=False, header=False)
                except:
                    continue
        
        print(f"Lemmatize edilmiş cümleler '{self.lemmatized_csv}' dosyasına kaydedildi.")
        print(f"Stem edilmiş cümleler '{self.stemmed_csv}' dosyasına kaydedildi.")
    
    def _create_zipf_comparison_plot(self, raw_text, lemmatized_text, stemmed_text):
        """Ham, Lemmatize ve Stem edilmiş metinler için Zipf karşılaştırması"""
        plt.figure(figsize=(12, 8))
        
        # Ham veri için Zipf
        if raw_text:
            raw_words = Counter(raw_text.split())
            raw_ranks = np.arange(1, len(raw_words) + 1)
            raw_freqs = [count for word, count in sorted(raw_words.items(), key=lambda x: x[1], reverse=True)]
            
            plt.loglog(raw_ranks, raw_freqs, 'b.', alpha=0.5, label='Ham Veri')
        
        # Lemmatized veri için Zipf
        lemma_words = Counter(lemmatized_text.split())
        lemma_ranks = np.arange(1, len(lemma_words) + 1)
        lemma_freqs = [count for word, count in sorted(lemma_words.items(), key=lambda x: x[1], reverse=True)]
        
        plt.loglog(lemma_ranks, lemma_freqs, 'g.', alpha=0.5, label='Lemmatize Edilmiş')
        
        # Stemmed veri için Zipf
        stem_words = Counter(stemmed_text.split())
        stem_ranks = np.arange(1, len(stem_words) + 1)
        stem_freqs = [count for word, count in sorted(stem_words.items(), key=lambda x: x[1], reverse=True)]
        
        plt.loglog(stem_ranks, stem_freqs, 'r.', alpha=0.5, label='Stem Edilmiş')
        
        # Teorik Zipf eğrisi
        first_rank_freq = max(lemma_freqs[0] if lemma_freqs else 0, 
                              stem_freqs[0] if stem_freqs else 0,
                              raw_freqs[0] if raw_text and raw_freqs else 0)
        
        plt.loglog(np.arange(1, 1000), [first_rank_freq/r for r in np.arange(1, 1000)], 'k-', 
                 label='Zipf Yasası (1/rank)')
        
        plt.xlabel('Sıra (log ölçeği)')
        plt.ylabel('Frekans (log ölçeği)')
        plt.title('Zipf Yasası Karşılaştırması: Ham vs Lemmatize vs Stem')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(os.path.join(self.zipf_dir, "zipf_comparison.png"))
        plt.close()
        
        print("Zipf yasası karşılaştırma grafiği kaydedildi.")
    
    def _visualize_category_distribution(self, df):
        """Kategori dağılımını görselleştirme"""
        plt.figure(figsize=(12, 6))
        
        # Kategori sayılarını al
        category_counts = df['category'].value_counts()
        
        # Çubuk grafiği
        sns_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        bars = plt.bar(category_counts.index, category_counts.values, color=sns_colors[:len(category_counts)])
        
        # Değerleri çubukların üzerine ekle
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{height}', ha='center', va='bottom')
        
        plt.title('Kategori Dağılımı', fontsize=16)
        plt.xlabel('Kategori', fontsize=14)
        plt.ylabel('Yorum Sayısı', fontsize=14)
        plt.xticks(rotation=0)
        plt.tight_layout()
        
        # Grafiği kaydet
        plt.savefig(os.path.join(self.output_dir, "category_distribution.png"))
        plt.close()
        
        print("Kategori dağılımı grafiği kaydedildi.")


def main():
    print("Turizm Yorumları Veri Seti Temizleme ve İşleme")
    print("=" * 70)
    
    # Varsayılan değerler
    input_file = "C:\\Users\\gokmu\\Desktop\\ddisleme\\yelp_academic_dataset_review.json"
    output_dir = "processed_data"
    max_file_size = 50
    max_records = None
    
    # Kullanıcı girişi
    input_file_user = input(f"Veri seti dosya yolu (varsayılan: {input_file}): ")
    if input_file_user.strip():
        input_file = input_file_user
    
    output_dir_user = input(f"Çıktı dizini (varsayılan: {output_dir}): ")
    if output_dir_user.strip():
        output_dir = output_dir_user
    
    max_file_size_user = input(f"Maksimum dosya boyutu MB (varsayılan: {max_file_size}): ")
    if max_file_size_user.strip():
        try:
            max_file_size = int(max_file_size_user)
        except:
            print(f"Geçersiz değer. Varsayılan değer kullanılacak: {max_file_size} MB")
    
    max_records_user = input("İşlenecek maksimum kayıt sayısı (sınırlama olmayacaksa boş bırakın): ")
    if max_records_user.strip():
        try:
            max_records = int(max_records_user)
        except:
            print("Geçersiz değer. Tüm kayıtlar işlenecek.")
    
    print(f"\nVeri seti dosyası: {input_file}")
    print(f"Çıktı dizini: {output_dir}")
    print(f"Maksimum dosya boyutu: {max_file_size} MB")
    if max_records:
        print(f"Maksimum kayıt sayısı: {max_records}")
    
    # Veri işleme
    processor = DataProcessor(
        input_file_path=input_file,
        output_dir=output_dir,
        max_file_size_mb=max_file_size,
        max_records=max_records
    )
    
    success = processor.process_data()
    
    if success:
        print("\nVeri işleme tamamlandı! İşlenen veri şu dosyalarda bulunabilir:")
        print(f"1. Ana veri seti: {os.path.join(output_dir, 'processed_reviews.csv')}")
        print(f"2. Lemmatize edilmiş cümleler: {os.path.join(output_dir, 'processed_reviews_lemmatized.csv')}")
        print(f"3. Stem edilmiş cümleler: {os.path.join(output_dir, 'processed_reviews_stemmed.csv')}")
        print(f"4. TF-IDF için metinler: {os.path.join(output_dir, 'tfidf_lemmatized.csv')} ve {os.path.join(output_dir, 'tfidf_stemmed.csv')}")
        print(f"5. Zipf analizi grafikleri: zipf_graphs/ dizininde")
        print("\nBu dosyaları model oluşturmak için kullanabilirsiniz (model_trainer.py).")
    else:
        print("\nVeri işleme tamamlanamadı. Lütfen girdi dosyanızı kontrol edin.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Program çalıştırılırken hata oluştu: {e}")
        import traceback
        traceback.print_exc()