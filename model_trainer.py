import os
import re
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time
import gensim
from gensim.models import Word2Vec
import joblib
import warnings
from collections import Counter
warnings.filterwarnings('ignore')

# NLTK gerekli dosyaları indirme
try:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    print("NLTK kaynakları başarıyla indirildi.")
except Exception as e:
    print(f"NLTK yüklemesinde hata: {e}")


class TFIDFAnalyzer:
    """TF-IDF analizi için sınıf"""
    def __init__(self, texts=None, max_features=5000, min_df=2, max_df=0.85, models_folder="models"):
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.texts = texts
        self.vectorizer = None
        self.tfidf_matrix = None
        self.feature_names = None
        self.similarity_matrix = None
        self.models_folder = models_folder
        
        # Models klasörünü oluştur
        if not os.path.exists(models_folder):
            os.makedirs(models_folder)
        
        if texts is not None:
            self.create_tfidf_vectors(texts)
    
    def create_tfidf_vectors(self, texts):
        """TF-IDF vektörleri oluşturma"""
        print("TF-IDF vektörleri oluşturuluyor...")
        
        # Boş metinleri kontrol et
        valid_texts = [text for text in texts if isinstance(text, str) and text.strip()]
        
        if not valid_texts:
            raise ValueError("TF-IDF için geçerli metin bulunamadı!")
        
        # TF-IDF vektörleştiriciyi oluştur
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            min_df=self.min_df,
            max_df=self.max_df
        )
        
        # Vektörleri hesapla
        self.tfidf_matrix = self.vectorizer.fit_transform(valid_texts)
        self.feature_names = self.vectorizer.get_feature_names_out()
        
        print(f"TF-IDF matrisi oluşturuldu. Şekil: {self.tfidf_matrix.shape}")
        
        # Vektörleştiriciyi kaydet
        joblib.dump(self.vectorizer, os.path.join(self.models_folder, f"tfidf_vectorizer_max{self.max_features}_min{self.min_df}.pkl"))
        
        return self.tfidf_matrix
    
    def save_tfidf_matrix_to_csv(self, output_path, sample_size=1000):
        """TF-IDF matrisini CSV olarak kaydet (büyük matrisler için örnek seçimi)"""
        if self.tfidf_matrix is None or self.feature_names is None:
            raise ValueError("Önce TF-IDF vektörleri oluşturulmalıdır.")
        
        # Matrisi dense formata çevir
        dense_matrix = self.tfidf_matrix.toarray()
        
        # Büyük matrisler için örnek seçimi
        if dense_matrix.shape[0] > sample_size or dense_matrix.shape[1] > 1000:
            row_sample = min(sample_size, dense_matrix.shape[0])
            col_sample = min(1000, dense_matrix.shape[1])
            dense_sample = dense_matrix[:row_sample, :col_sample]
            feature_sample = self.feature_names[:col_sample]
            
            print(f"Not: Matris çok büyük, ilk {row_sample} satır ve {col_sample} sütun kaydediliyor.")
            df = pd.DataFrame(dense_sample, columns=feature_sample)
        else:
            df = pd.DataFrame(dense_matrix, columns=self.feature_names)
        
        # CSV olarak kaydet
        df.to_csv(output_path, index=False)
        print(f"TF-IDF matrisi '{output_path}' dosyasına kaydedildi.")
    
    def calculate_similarity_matrix(self):
        """Benzerlik matrisi hesaplama"""
        if self.tfidf_matrix is None:
            raise ValueError("Önce TF-IDF vektörleri oluşturulmalıdır.")
        
        print("Benzerlik matrisi hesaplanıyor...")
        self.similarity_matrix = cosine_similarity(self.tfidf_matrix)
        print(f"Benzerlik matrisi oluşturuldu. Şekil: {self.similarity_matrix.shape}")
        
        # Benzerlik matrisini kaydet
        joblib.dump(self.similarity_matrix, os.path.join(self.models_folder, "similarity_matrix.pkl"))
        
        return self.similarity_matrix
    
    def find_similar_words(self, word, top_n=5):
        """Bir kelimeye benzer kelimeleri bulma"""
        if self.tfidf_matrix is None or self.feature_names is None:
            raise ValueError("Önce TF-IDF vektörleri oluşturulmalıdır.")
        
        try:
            # Kelimenin indeksini bulma
            word_idx = np.where(self.feature_names == word.lower())[0][0]
            
            # Kelimenin vektörünü alma
            word_vector = self.tfidf_matrix[:, word_idx].toarray().T
            
            # Tüm kelimeler için benzerlik hesaplama
            word_similarities = []
            
            for i, feature in enumerate(self.feature_names):
                if i != word_idx:  # Kendisini hariç tut
                    # Diğer kelime vektörü
                    feature_vector = self.tfidf_matrix[:, i].toarray().T
                    
                    # Kosinüs benzerliği hesapla
                    similarity = cosine_similarity(word_vector, feature_vector)[0][0]
                    word_similarities.append((feature, similarity))
            
            # Benzerliğe göre sıralama
            word_similarities.sort(key=lambda x: x[1], reverse=True)
            
            # En benzer kelimeleri döndür
            return word_similarities[:top_n]
            
        except (IndexError, ValueError):
            print(f"'{word}' kelimesi TF-IDF sözlüğünde bulunamadı.")
            return []
    
    def visualize_word_similarities(self, word, similar_words):
        """Bir kelimeye benzer kelimeleri görselleştirme"""
        if not similar_words:
            print(f"'{word}' için benzer kelime bulunamadı.")
            return
        
        plt.figure(figsize=(10, 6))
        
        # Verileri hazırla
        words = [w for w, _ in similar_words]
        scores = [s for _, s in similar_words]
        
        # Çubuk grafiği
        bars = plt.barh(words, scores, color='skyblue')
        
        # Değerleri çubukların sağına ekle
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{scores[i]:.4f}', ha='left', va='center')
        
        plt.title(f"'{word}' Kelimesine Benzer Kelimeler (TF-IDF)")
        plt.xlabel('Benzerlik Skoru')
        plt.ylabel('Kelimeler')
        plt.xlim(0, max(scores) * 1.1)  # Skor değerlerinin görünmesi için boşluk bırak
        plt.tight_layout()
        
        # Kaydet
        plt.savefig(os.path.join(self.models_folder, f"word_similarities_{word}.png"))
        plt.close()


class Word2VecTrainer:
    """Word2Vec model eğitimi için sınıf"""
    def __init__(self, models_folder="models"):
        self.models_folder = models_folder
        
        # Models klasörünü oluştur
        if not os.path.exists(models_folder):
            os.makedirs(models_folder)
        
        # Parametreler - ödevde verilen parametre seti
        self.parameters = [
            {'model_type': 'cbow', 'window': 2, 'vector_size': 100},
            {'model_type': 'skipgram', 'window': 2, 'vector_size': 100},
            {'model_type': 'cbow', 'window': 4, 'vector_size': 100},
            {'model_type': 'skipgram', 'window': 4, 'vector_size': 100},
            {'model_type': 'cbow', 'window': 2, 'vector_size': 300},
            {'model_type': 'skipgram', 'window': 2, 'vector_size': 300},
            {'model_type': 'cbow', 'window': 4, 'vector_size': 300},
            {'model_type': 'skipgram', 'window': 4, 'vector_size': 300}
        ]
        
        self.models = {}
        self.training_stats = {}
    
    def train_and_save_model(self, corpus, params, model_name):
        """Word2Vec modelini eğitme ve kaydetme"""
        try:
            # Eğitim süresini ölçme
            start_time = time.time()
            
            # Word2Vec modelini eğit
            model = Word2Vec(
                corpus, 
                vector_size=params['vector_size'],
                window=params['window'], 
                min_count=1, 
                sg=1 if params['model_type'] == 'skipgram' else 0,
                workers=4
            )
            
            # Eğitim süresini hesapla
            training_time = time.time() - start_time
            
            # Modeli kaydet
            model_path = os.path.join(self.models_folder, f"{model_name}_{params['model_type']}_window{params['window']}_dim{params['vector_size']}.model")
            model.save(model_path)
            
            # Model boyutunu hesapla
            model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
            
            print(f"{model_name}_{params['model_type']}_window{params['window']}_dim{params['vector_size']} model saved!")
            
            # İstatistikleri kaydet
            stats = {
                'model_type': params['model_type'],
                'window': params['window'],
                'vector_size': params['vector_size'],
                'training_time': training_time,
                'vocabulary_size': len(model.wv.index_to_key),
                'model_size_mb': model_size_mb
            }
            
            full_model_name = f"{model_name}_{params['model_type']}_window{params['window']}_dim{params['vector_size']}"
            self.training_stats[full_model_name] = stats
            self.models[full_model_name] = model
            
            return model
        except Exception as e:
            print(f"Model eğitimi sırasında hata: {e}")
            return None
    
    def train_all_models(self, lemmatized_corpus, stemmed_corpus):
        """Lemmatize ve stem edilmiş corpus'lar için tüm modelleri eğitme"""
        models = {}
        
        # Lemmatize edilmiş corpus ile modelleri eğitme
        print("Lemmatize edilmiş metin üzerinde modeller eğitiliyor...")
        for param in self.parameters:
            model = self.train_and_save_model(lemmatized_corpus, param, "lemmatized_model")
            if model:
                models[f"lemmatized_{param['model_type']}_w{param['window']}_d{param['vector_size']}"] = model
        
        # Stemlenmiş corpus ile modelleri eğitme
        print("Stem edilmiş metin üzerinde modeller eğitiliyor...")
        for param in self.parameters:
            model = self.train_and_save_model(stemmed_corpus, param, "stemmed_model")
            if model:
                models[f"stemmed_{param['model_type']}_w{param['window']}_d{param['vector_size']}"] = model
        
        # Eğitim istatistiklerini görselleştirme
        self.visualize_training_stats()
        
        return models
    
    def visualize_training_stats(self):
        """Eğitim istatistiklerini görselleştirme"""
        if not self.training_stats:
            print("Henüz model eğitim istatistikleri mevcut değil.")
            return
        
        # İstatistikleri DataFrame'e dönüştürme
        stats_df = pd.DataFrame.from_dict(self.training_stats, orient='index')
        
        # İstatistikleri kaydet
        stats_df.to_csv(os.path.join(self.models_folder, "w2v_model_stats.csv"))
        
        # Eğitim Süreleri Grafiği
        plt.figure(figsize=(14, 7))
        sns.barplot(x=stats_df.index, y='training_time', data=stats_df)
        plt.title('Word2Vec Modelleri Eğitim Süreleri')
        plt.xlabel('Model')
        plt.ylabel('Eğitim Süresi (saniye)')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(os.path.join(self.models_folder, "w2v_training_times.png"))
        plt.close()
        
        # Model Boyutları Grafiği
        plt.figure(figsize=(14, 7))
        sns.barplot(x=stats_df.index, y='model_size_mb', data=stats_df)
        plt.title('Word2Vec Modelleri Boyutları')
        plt.xlabel('Model')
        plt.ylabel('Model Boyutu (MB)')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(os.path.join(self.models_folder, "w2v_model_sizes.png"))
        plt.close()
        
        # Vokabüler Boyutları Grafiği
        plt.figure(figsize=(14, 7))
        sns.barplot(x=stats_df.index, y='vocabulary_size', data=stats_df)
        plt.title('Word2Vec Modelleri Vokabüler Boyutları')
        plt.xlabel('Model')
        plt.ylabel('Vokabüler Boyutu')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(os.path.join(self.models_folder, "w2v_vocabulary_sizes.png"))
        plt.close()
        
        # Model türüne göre karşılaştırma (CBOW vs SkipGram)
        plt.figure(figsize=(12, 6))
        model_type_stats = stats_df.groupby('model_type')[['vocabulary_size', 'model_size_mb', 'training_time']].mean()
        model_type_stats.plot(kind='bar', figsize=(12, 6))
        plt.title('CBOW vs SkipGram Karşılaştırması')
        plt.ylabel('Ortalama Değer')
        plt.tight_layout()
        plt.savefig(os.path.join(self.models_folder, "model_type_comparison.png"))
        plt.close()
        
        # Pencere boyutuna göre karşılaştırma
        plt.figure(figsize=(12, 6))
        window_stats = stats_df.groupby('window')[['vocabulary_size', 'model_size_mb', 'training_time']].mean()
        window_stats.plot(kind='bar', figsize=(12, 6))
        plt.title('Pencere Boyutu Karşılaştırması')
        plt.ylabel('Ortalama Değer')
        plt.tight_layout()
        plt.savefig(os.path.join(self.models_folder, "window_size_comparison.png"))
        plt.close()
        
        # Vektör boyutuna göre karşılaştırma
        plt.figure(figsize=(12, 6))
        vector_stats = stats_df.groupby('vector_size')[['vocabulary_size', 'model_size_mb', 'training_time']].mean()
        vector_stats.plot(kind='bar', figsize=(12, 6))
        plt.title('Vektör Boyutu Karşılaştırması')
        plt.ylabel('Ortalama Değer')
        plt.tight_layout()
        plt.savefig(os.path.join(self.models_folder, "vector_size_comparison.png"))
        plt.close()
        
        print("Eğitim istatistikleri görselleştirildi ve w2v_model_stats.csv dosyasına kaydedildi.")
        
        return stats_df
    
    def get_similar_words(self, model, word, top_n=5):
        """Bir kelimeye en benzer kelimeleri bulmak"""
        try:
            similar_words = model.wv.most_similar(word, topn=top_n)
            return similar_words
        except KeyError:
            print(f"'{word}' kelimesi model vokabülerinde bulunamadı.")
            return []
        except Exception as e:
            print(f"Benzer kelimeler bulunurken hata: {e}")
            return []


class ModelComparator:
    """Word2Vec modellerini karşılaştırmak için sınıf"""
    def __init__(self, models_dir="models"):
        self.models_dir = models_dir
        self.models = {}
        self.model_stats = {}
        self.test_words = ["hotel", "restaurant", "food", "museum", "service"]
        self.similarity_results = {}
    
    def load_models(self):
        """Modelleri yükleme"""
        print("Word2Vec modelleri yükleniyor...")
        
        model_files = [f for f in os.listdir(self.models_dir) if f.endswith('.model')]
        
        if not model_files:
            print("Yüklenecek model bulunamadı!")
            return False
        
        # Modelleri yükle
        for model_file in model_files:
            model_path = os.path.join(self.models_dir, model_file)
            try:
                model = Word2Vec.load(model_path)
                self.models[model_file] = model
                
                # Model adından parametreleri çıkarma
                if "lemmatized" in model_file:
                    preprocess = "lemmatized"
                else:
                    preprocess = "stemmed"
                
                if "cbow" in model_file:
                    model_type = "cbow"
                else:
                    model_type = "skipgram"
                
                window = int(re.search(r'window(\d+)', model_file).group(1))
                dim = int(re.search(r'dim(\d+)', model_file).group(1))
                
                # Model istatistiklerini kaydet
                self.model_stats[model_file] = {
                    'preprocess': preprocess,
                    'model_type': model_type,
                    'window': window,
                    'vector_size': dim,
                    'vocabulary_size': len(model.wv.index_to_key),
                    'model_size_mb': os.path.getsize(model_path) / (1024 * 1024)
                }
                
                print(f"Yüklendi: {model_file}")
            except Exception as e:
                print(f"Hata: {model_file} yüklenemedi - {e}")
        
        print(f"Toplam {len(self.models)} model yüklendi.")
        return True
    
    def compare_similar_words(self, test_words=None, top_n=5):
        """Test kelimeleri için benzer kelimeleri karşılaştırma"""
        if not test_words:
            test_words = self.test_words
        
        print(f"Test kelimeleri: {test_words}")
        print("Kelime benzerlikleri karşılaştırılıyor...")
        
        # Her model için kelime benzerliklerini hesapla
        for model_name, model in self.models.items():
            print(f"\nModel: {model_name}")
            model_results = {}
            
            for word in test_words:
                try:
                    similar_words = model.wv.most_similar(word, topn=top_n)
                    model_results[word] = similar_words
                    
                    print(f"  '{word}' için en benzer {top_n} kelime:")
                    for w, score in similar_words:
                        print(f"    {w}: {score:.4f}")
                except KeyError:
                    print(f"  '{word}' kelimesi model vokabülerinde bulunamadı.")
                    model_results[word] = []
            
            self.similarity_results[model_name] = model_results
        
        # Her kelime için model karşılaştırma grafiği
        for word in test_words:
            self._visualize_word_across_models(word, top_n)
        
        return self.similarity_results
    
    def _visualize_word_across_models(self, word, top_n=5):
        """Bir kelimeyi farklı modellerde karşılaştırmalı görselleştirme"""
        # Kelime için sonuçları olan modelleri bul
        valid_models = {}
        for model_name, results in self.similarity_results.items():
            if word in results and results[word]:
                valid_models[model_name] = results[word]
        
        if not valid_models:
            print(f"'{word}' kelimesi hiçbir modelde bulunamadı veya benzer kelimesi yok.")
            return
        
        # Görselleştirme
        plt.figure(figsize=(15, 10))
        
        # Kaç model var?
        n_models = len(valid_models)
        cols = 2
        rows = (n_models + 1) // cols
        
        # Her model için bir alt-grafik oluştur
        for i, (model_name, similar_words) in enumerate(valid_models.items(), 1):
            plt.subplot(rows, cols, i)
            
            words = [w for w, _ in similar_words]
            scores = [s for _, s in similar_words]
            
            # Çubuk grafik
            bars = plt.barh(words, scores, color='skyblue')
            
            # Değerleri çubukların üzerine ekle
            for j, bar in enumerate(bars):
                width = bar.get_width()
                plt.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                        f'{scores[j]:.2f}', ha='left', va='center')
            
            plt.title(model_name, fontsize=10)
            plt.xlim(0, 1)
        
        plt.suptitle(f"'{word}' Kelimesi için Model Karşılaştırması", fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        # Kaydet
        plt.savefig(os.path.join(self.models_dir, f"word_similarities_comparison_{word}.png"))
        plt.close()
        
        print(f"'{word}' kelimesi için model karşılaştırması görselleştirildi.")
    
    def analyze_model_parameters(self):
        """Model parametrelerinin etkisini analiz etme"""
        if not self.model_stats:
            print("Karşılaştırma için yeterli veri bulunmuyor. Önce modelleri yükleyin.")
            return
        
        print("\nModel parametrelerinin etkisi analiz ediliyor...")
        
        # İstatistikleri DataFrame'e dönüştür
        stats_df = pd.DataFrame.from_dict(self.model_stats, orient='index')
        
        # 1. Model türü etkisi (CBOW vs SkipGram)
        print("\n1. Model Türü Etkisi (CBOW vs SkipGram):")
        model_type_stats = stats_df.groupby('model_type')[['vocabulary_size', 'model_size_mb']].mean()
        print(model_type_stats)
        
        # 2. Pencere boyutu etkisi
        print("\n2. Pencere Boyutu Etkisi:")
        window_stats = stats_df.groupby('window')[['vocabulary_size', 'model_size_mb']].mean()
        print(window_stats)
        
        # 3. Vektör boyutu etkisi
        print("\n3. Vektör Boyutu Etkisi:")
        vector_stats = stats_df.groupby('vector_size')[['vocabulary_size', 'model_size_mb']].mean()
        print(vector_stats)
        
        # 4. Önişleme tekniği etkisi
        print("\n4. Önişleme Tekniği Etkisi (Lemmatize vs Stem):")
        preprocess_stats = stats_df.groupby('preprocess')[['vocabulary_size', 'model_size_mb']].mean()
        print(preprocess_stats)
        
        # 5. En iyi modeller
        print("\n5. En İyi Performans Gösteren Modeller:")
        best_models = stats_df.sort_values('vocabulary_size', ascending=False).head(3)
        print("En Büyük Vokabüler Boyutuna Sahip Modeller:")
        print(best_models[['preprocess', 'model_type', 'window', 'vector_size', 'vocabulary_size']])
        
        # Kapsamlı rapor oluştur
        self.generate_comparison_report(stats_df)
        
        return stats_df
    
    def generate_comparison_report(self, stats_df):
        """Kapsamlı karşılaştırma raporu oluşturma"""
        report_path = os.path.join(self.models_dir, "model_comparison_report.txt")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("TURİZM YORUMLARI WORD2VEC MODELLERİ KARŞILAŞTIRMA RAPORU\n")
            f.write("=" * 70 + "\n\n")
            
            # 1. Genel Bilgiler
            f.write("1. GENEL BİLGİLER\n")
            f.write("-" * 50 + "\n")
            f.write(f"Toplam Model Sayısı: {len(self.models)}\n")
            f.write(f"Modeller: {', '.join(self.models.keys())}\n\n")
            
            # 2. Model Parametrelerinin Etkisi
            f.write("2. MODEL PARAMETRELERİNİN ETKİSİ\n")
            f.write("-" * 50 + "\n")
            
            # Model türü etkisi
            f.write("2.1. Model Türü (CBOW vs SkipGram)\n")
            model_type_stats = stats_df.groupby('model_type')[['vocabulary_size', 'model_size_mb']].mean()
            f.write(f"{model_type_stats.to_string()}\n\n")
            
            # Pencere boyutu etkisi
            f.write("2.2. Pencere Boyutu Etkisi\n")
            window_stats = stats_df.groupby('window')[['vocabulary_size', 'model_size_mb']].mean()
            f.write(f"{window_stats.to_string()}\n\n")
            
            # Vektör boyutu etkisi
            f.write("2.3. Vektör Boyutu Etkisi\n")
            vector_stats = stats_df.groupby('vector_size')[['vocabulary_size', 'model_size_mb']].mean()
            f.write(f"{vector_stats.to_string()}\n\n")
            
            # Önişleme tekniği etkisi
            f.write("2.4. Önişleme Tekniği Etkisi\n")
            preprocess_stats = stats_df.groupby('preprocess')[['vocabulary_size', 'model_size_mb']].mean()
            f.write(f"{preprocess_stats.to_string()}\n\n")
            
            # 3. Kelime Benzerlikleri
            if self.similarity_results:
                f.write("3. KELİME BENZERLİKLERİ\n")
                f.write("-" * 50 + "\n")
                
                for word in self.test_words:
                    f.write(f"3.1. '{word}' Kelimesi için Karşılaştırma\n\n")
                    
                    for model_name, results in self.similarity_results.items():
                        f.write(f"  Model: {model_name}\n")
                        
                        if word in results and results[word]:
                            for i, (similar_word, score) in enumerate(results[word], 1):
                                f.write(f"    {i}. {similar_word}: {score:.4f}\n")
                        else:
                            f.write("    Kelime bulunamadı veya benzer kelime yok.\n")
                        
                        f.write("\n")
            
            # 4. Sonuç ve Değerlendirme
            f.write("4. SONUÇ VE DEĞERLENDİRME\n")
            f.write("-" * 50 + "\n")
            
            # Model türü değerlendirmesi
            if 'model_type' in stats_df.columns:
                cbow_vocab = stats_df[stats_df['model_type'] == 'cbow']['vocabulary_size'].mean()
                skipgram_vocab = stats_df[stats_df['model_type'] == 'skipgram']['vocabulary_size'].mean()
                
                if skipgram_vocab > cbow_vocab:
                    f.write("1. SkipGram modelleri genellikle CBOW modellerinden daha büyük vokabüler boyutuna sahiptir.\n")
                    f.write("   Bu, nadir kelimeler için daha iyi temsil anlamına gelebilir.\n")
                else:
                    f.write("1. CBOW modelleri genellikle SkipGram modellerinden daha büyük vokabüler boyutuna sahiptir.\n")
                    f.write("   Bu, yaygın kelimeler için daha iyi temsil anlamına gelebilir.\n")
            
            # Pencere boyutu değerlendirmesi
            if 'window' in stats_df.columns:
                window2_vocab = stats_df[stats_df['window'] == 2]['vocabulary_size'].mean()
                window4_vocab = stats_df[stats_df['window'] == 4]['vocabulary_size'].mean()
                
                if window4_vocab > window2_vocab:
                    f.write("2. Daha büyük pencere boyutu (4) daha fazla bağlamsal ilişkiyi yakalayabilmektedir.\n")
                    f.write("   Bu, daha geniş kapsamlı semantik ilişkileri öğrenmek için faydalı olabilir.\n")
                else:
                    f.write("2. Daha küçük pencere boyutu (2) daha verimli çalışmaktadır.\n")
                    f.write("   Bu, daha yerel ve doğrudan kelime ilişkilerini öğrenmek için yeterli olabilir.\n")
            
            # Vektör boyutu değerlendirmesi
            if 'vector_size' in stats_df.columns:
                size100_mb = stats_df[stats_df['vector_size'] == 100]['model_size_mb'].mean()
                size300_mb = stats_df[stats_df['vector_size'] == 300]['model_size_mb'].mean()
                
                f.write("3. Vektör boyutu ile model boyutu arasında doğrusal bir ilişki vardır.\n")
                f.write(f"   100 boyutlu modellerin ortalama boyutu: {size100_mb:.2f} MB\n")
                f.write(f"   300 boyutlu modellerin ortalama boyutu: {size300_mb:.2f} MB\n")
                f.write("   Daha büyük vektör boyutu, daha zengin kelime temsillerine izin verir ancak hesaplama maliyeti artar.\n")
            
            # Önişleme tekniği değerlendirmesi
            if 'preprocess' in stats_df.columns:
                lemma_vocab = stats_df[stats_df['preprocess'] == 'lemmatized']['vocabulary_size'].mean()
                stem_vocab = stats_df[stats_df['preprocess'] == 'stemmed']['vocabulary_size'].mean()
                
                f.write("4. Önişleme tekniği, vokabüler boyutunu önemli ölçüde etkilemektedir.\n")
                f.write(f"   Lemmatized vokabüler boyutu: {lemma_vocab:.2f}\n")
                f.write(f"   Stemmed vokabüler boyutu: {stem_vocab:.2f}\n")
                
                if lemma_vocab > stem_vocab:
                    f.write("   Lemmatization, stemming'e göre daha zengin bir vokabüler sağlar, bu da daha iyi anlam ayrımına yol açabilir.\n")
                else:
                    f.write("   Stemming, lemmatization'a göre daha kompakt bir vokabüler sağlar, bu da daha hızlı işleme yol açabilir.\n")
            
            # En iyi model önerisi
            f.write("\n5. Turizm Yorumları Sınıflandırması İçin En Uygun Model\n")
            best_model_index = stats_df['vocabulary_size'].idxmax()
            f.write(f"   En yüksek vokabüler boyutuna sahip model: {best_model_index}\n")
            f.write("   Bu model, en geniş kelime kapsamını sağlar ve turizm alanındaki çeşitli terimleri daha iyi temsil edebilir.\n")
            
            f.write("\nNot: Bu rapor, modellerin sadece temel özelliklerini karşılaştırmaktadır.\n")
            f.write("Gerçek dünya uygulamalarında, modellerin doğruluğu ve performansı daha kapsamlı testlerle değerlendirilmelidir.\n")
        
        print(f"Kapsamlı karşılaştırma raporu '{report_path}' dosyasına kaydedildi.")


class VenueClassifier:
    """Turizm mekanları sınıflandırıcısı"""
    def __init__(self, data=None, use_lemmatized=True, models_folder="models"):
        """
        Sınıflandırıcı başlatma
        
        Parameters:
        -----------
        data: pd.DataFrame, default=None
            İşlenmiş mekan verilerini içeren DataFrame
        use_lemmatized: bool, default=True
            Lemmatize edilmiş metinleri kullanma (False ise stemmed metinler kullanılır)
        models_folder: str, default="models"
            Modellerin kaydedileceği klasör
        """
        self.models_folder = models_folder
        self.use_lemmatized = use_lemmatized
        
        # Models klasörünü oluştur
        if not os.path.exists(models_folder):
            os.makedirs(models_folder)
        
        # Veri yükleme
        self.data = data
        if data is None:
            self._load_default_data()
        
        # Veri hazırlama
        self._prepare_data()
        
        # TF-IDF vektörleştirici
        self.vectorizer = None
        self.tfidf_matrix = None
        
        # Kategori vektörleri
        self.category_vectors = {}
        
        # Benzerlik matrisi
        self.similarity_matrix = None
    
    def _load_default_data(self):
        """Varsayılan veriyi yükleme"""
        try:
            data_path = os.path.join("processed_data", "processed_reviews.csv")
            if os.path.exists(data_path):
                self.data = pd.read_csv(data_path)
                print(f"Veri otomatik olarak yüklendi: {data_path}")
            else:
                raise FileNotFoundError(f"Veri dosyası bulunamadı: {data_path}")
        except Exception as e:
            print(f"Veri yüklenirken hata: {e}")
            self.data = None
    
    def _prepare_data(self):
        """Veriyi hazırlama"""
        if self.data is None:
            raise ValueError("Veri yüklenemedi!")
        
        # Gerekli sütunların varlığını kontrol et
        required_columns = ['review_text', 'category', 'lemmatized_text', 'stemmed_text']
        missing_columns = [col for col in required_columns if col not in self.data.columns]
        
        if missing_columns:
            raise ValueError(f"Veri setinde gerekli sütunlar eksik: {missing_columns}")
        
        # İşlenmiş metni seçme
        print(f"İşlenecek kayıt sayısı: {len(self.data)}")
        
        if self.use_lemmatized:
            print("Lemmatize edilmiş metinler kullanılıyor.")
            self.data['processed_text'] = self.data['lemmatized_text']
        else:
            print("Stem edilmiş metinler kullanılıyor.")
            self.data['processed_text'] = self.data['stemmed_text']
        
        # NaN değerleri temizle
        self.data['processed_text'] = self.data['processed_text'].fillna('')
        
        # İlk birkaç satırı göster
        print("\nVeri örnekleri (ilk 3 satır):")
        sample_data = self.data[['category', 'processed_text']].head(3)
        for i, (_, row) in enumerate(sample_data.iterrows(), 1):
            print(f"\n{i}. Kategori: {row['category']}")
            print(f"   İşlenmiş Metin: {row['processed_text'][:100]}...")
    
    def vectorize_data(self, max_features=5000):
        """Verileri TF-IDF ile vektörleştirme"""
        print("\nVeriler vektörleştiriliyor...")
        
        # TF-IDF vektörleştirici
        self.vectorizer = TfidfVectorizer(max_features=max_features, min_df=2, max_df=0.85)
        
        # Verileri vektörleştirme
        self.tfidf_matrix = self.vectorizer.fit_transform(self.data['processed_text'])
        
        print(f"Vektörleştirme tamamlandı. Şekil: {self.tfidf_matrix.shape}")
        
        # Vektörleştiriciyi kaydet
        joblib.dump(self.vectorizer, os.path.join(self.models_folder, "venue_classifier_vectorizer.pkl"))
        
        return self.tfidf_matrix
    
    def train_classifier(self, max_features=5000):
        """Sınıflandırıcıyı eğitme"""
        # Vektörleştirme
        if self.tfidf_matrix is None:
            self.vectorize_data(max_features=max_features)
        
        # Her kategori için ortalama vektör hesapla
        print("\nKategori vektörleri oluşturuluyor...")
        
        categories = self.data['category'].unique()
        
        for category in categories:
            # Kategori örneklerini seç
            category_indices = self.data[self.data['category'] == category].index
            
            if len(category_indices) == 0:
                continue
            
            # Kategori vektörlerini al
            category_vectors = self.tfidf_matrix[category_indices]
            
            # Ortalama vektörü hesapla
            category_vector = np.asarray(category_vectors.mean(axis=0))
            
            # Saklama
            self.category_vectors[category] = category_vector
        
        print(f"Toplam {len(self.category_vectors)} kategori vektörü oluşturuldu.")
        
        # Kategori vektörlerini kaydet
        joblib.dump(self.category_vectors, os.path.join(self.models_folder, "venue_category_vectors.pkl"))
        
        # Similarity matrix hesapla
        print("\nBenzerlik matrisi hesaplanıyor...")
        self.similarity_matrix = cosine_similarity(self.tfidf_matrix)
        
        print(f"Benzerlik matrisi oluşturuldu. Şekil: {self.similarity_matrix.shape}")
        
        # Benzerlik matrisini kaydet
        joblib.dump(self.similarity_matrix, os.path.join(self.models_folder, "venue_similarity_matrix.pkl"))
        
        return self.category_vectors
    
    def classify_venue(self, text):
        """
        Mekan metnini sınıflandırma
        
        Parameters:
        -----------
        text: str
            Sınıflandırılacak mekan metni
        
        Returns:
        --------
        dict
            Sınıflandırma sonuçları
        """
        # Vektörleştirici kontrolü
        if self.vectorizer is None or not self.category_vectors:
            raise ValueError("Önce sınıflandırıcı eğitilmelidir (train_classifier).")
        
        # Metni vektörleştir
        text_vector = self.vectorizer.transform([text])
        
        # Her kategori ile benzerlik hesapla
        similarities = {}
        
        for category, category_vector in self.category_vectors.items():
            # Kategori vektörünün boyutunu ayarla
            category_vector_reshaped = category_vector.reshape(1, -1)
            
            # Kosinüs benzerliği hesapla
            similarity = cosine_similarity(text_vector, category_vector_reshaped)[0][0]
            similarities[category] = similarity
        
        # Benzerlik skorlarını sırala
        sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        
        # Sonuçları döndür
        return {
            'predicted_category': sorted_similarities[0][0],
            'confidence': sorted_similarities[0][1],
            'all_scores': sorted_similarities
        }
    
    def evaluate_classifier(self, test_size=0.2, random_state=42):
        """
        Sınıflandırıcıyı değerlendirme
        
        Parameters:
        -----------
        test_size: float, default=0.2
            Test seti oranı
        random_state: int, default=42
            Rastgele ayırma için tohum değeri
        
        Returns:
        --------
        dict
            Değerlendirme sonuçları
        """
        # Vektörleştirici kontrolü
        if self.vectorizer is None or not self.category_vectors:
            raise ValueError("Önce sınıflandırıcı eğitilmelidir (train_classifier).")
        
        # Veriyi eğitim ve test olarak ayır
        train_data, test_data = train_test_split(
            self.data, 
            test_size=test_size, 
            random_state=random_state,
            stratify=self.data['category'] if len(self.data['category'].unique()) > 1 else None
        )
        
        # Tahminler
        y_true = []
        y_pred = []
        
        print("\nTest verisi üzerinde değerlendirme yapılıyor...")
        
        for _, row in tqdm(test_data.iterrows(), total=len(test_data)):
            try:
                true_category = row['category']
                review_text = row['processed_text']
                
                # Prediction
                prediction = self.classify_venue(review_text)
                predicted_category = prediction['predicted_category']
                
                y_true.append(true_category)
                y_pred.append(predicted_category)
            except Exception as e:
                print(f"Test satırı değerlendirilirken hata: {e}")
                continue
        
        # Değerlendirme metrikleri
        accuracy = accuracy_score(y_true, y_pred)
        conf_matrix = confusion_matrix(y_true, y_pred)
        report = classification_report(y_true, y_pred, output_dict=True)
        
        # Sonuçları yazdır
        print(f"\nDoğruluk (Accuracy): {accuracy:.4f}")
        print("\nConfusion Matrix:")
        print(conf_matrix)
        print("\nSınıflandırma Raporu:")
        print(classification_report(y_true, y_pred))
        
        # Sonuçları görselleştir
        self._visualize_evaluation_results(y_true, y_pred, conf_matrix, report)
        
        return {
            'accuracy': accuracy,
            'confusion_matrix': conf_matrix,
            'classification_report': report,
            'y_true': y_true,
            'y_pred': y_pred
        }
    
    def _visualize_evaluation_results(self, y_true, y_pred, conf_matrix, report):
        """Değerlendirme sonuçlarını görselleştirme"""
        # Confusion Matrix
        plt.figure(figsize=(10, 8))
        categories = sorted(set(y_true))
        
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=categories, yticklabels=categories)
        
        plt.title('Confusion Matrix')
        plt.xlabel('Tahmin Edilen Kategori')
        plt.ylabel('Gerçek Kategori')
        plt.tight_layout()
        plt.savefig(os.path.join(self.models_folder, "confusion_matrix.png"))
        plt.close()
        
        # Sınıflandırma Metrikleri
        plt.figure(figsize=(12, 6))
        
        # Kategorileri ve metrikleri hazırla
        cats = []
        precision = []
        recall = []
        f1 = []
        
        for category, metrics in report.items():
            if category not in ['accuracy', 'macro avg', 'weighted avg']:
                cats.append(category)
                precision.append(metrics['precision'])
                recall.append(metrics['recall'])
                f1.append(metrics['f1-score'])
        
        # Çubuk grafiği için kategori konumları
        x = np.arange(len(cats))
        width = 0.25
        
        # Çubuk grafiği
        plt.bar(x - width, precision, width, label='Precision')
        plt.bar(x, recall, width, label='Recall')
        plt.bar(x + width, f1, width, label='F1-Score')
        
        plt.xlabel('Kategori')
        plt.ylabel('Skor')
        plt.title('Sınıflandırma Metrikleri')
        plt.xticks(x, cats)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.models_folder, "classification_metrics.png"))
        plt.close()
        
        # Doğruluk Grafiği
        plt.figure(figsize=(8, 5))
        plt.bar(['Accuracy'], [report['accuracy']], color='green')
        plt.ylabel('Doğruluk Skoru')
        plt.title('Sınıflandırıcı Doğruluğu')
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(os.path.join(self.models_folder, "accuracy.png"))
        plt.close()
        
        print("Değerlendirme sonuçları görselleştirildi.")
    
    def find_similar_venues(self, venue_idx, top_n=5):
        """
        Benzer mekanları bulma
        
        Parameters:
        -----------
        venue_idx: int
            Benzerliği hesaplanacak mekan indeksi
        top_n: int, default=5
            Döndürülecek benzer mekan sayısı
        
        Returns:
        --------
        pd.DataFrame
            Benzer mekanların bilgileri
        """
        # Similarity matrix kontrolü
        if self.similarity_matrix is None:
            raise ValueError("Önce benzerlik matrisi hesaplanmalıdır (train_classifier).")
        
        # İndeks kontrolü
        if venue_idx < 0 or venue_idx >= len(self.data):
            raise ValueError(f"Geçersiz mekan indeksi: {venue_idx}. İndeks 0 ile {len(self.data)-1} arasında olmalıdır.")
        
        # Benzerlik skorları
        venue_similarities = self.similarity_matrix[venue_idx]
        
        # En benzer mekanların indekslerini al (kendisi hariç)
        similar_indices = venue_similarities.argsort()[::-1][1:top_n+1]
        
        # Sonuçları DataFrame'e dönüştür
        similar_venues = []
        
        for idx in similar_indices:
            venue = self.data.iloc[idx]
            similar_venues.append({
                'title': venue.get('title', f"Venue {idx}"),
                'category': venue['category'],
                'similarity_score': venue_similarities[idx]
            })
        
        return pd.DataFrame(similar_venues)
    
    def visualize_similar_venues(self, query_venue, similar_venues):
        """
        Benzer mekanları görselleştirme
        
        Parameters:
        -----------
        query_venue: str
            Sorgu mekanının adı
        similar_venues: pd.DataFrame
            Benzer mekanların bilgileri
        """
        plt.figure(figsize=(10, 6))
        
        # Kategorilere göre renkler
        categories = similar_venues['category'].unique()
        category_colors = {}
        
        for i, category in enumerate(categories):
            category_colors[category] = plt.cm.tab10(i)
        
        # Mekanları çiz
        bars = plt.barh(
            similar_venues['title'],
            similar_venues['similarity_score'],
            color=[category_colors[cat] for cat in similar_venues['category']]
        )
        
        # Değerleri çubukların sağına ekle
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{width:.2f}', ha='left', va='center')
        
        # Kategori lejantı
        legend_patches = [plt.Rectangle((0, 0), 1, 1, color=category_colors[cat]) for cat in categories]
        plt.legend(legend_patches, categories, loc='lower right')
        
        plt.title(f"'{query_venue}' Mekanına Benzer Mekanlar")
        plt.xlabel('Benzerlik Skoru')
        plt.ylabel('Mekan Adı')
        plt.xlim(0, 1)
        plt.tight_layout()
        
        plt.savefig(os.path.join(self.models_folder, f"similar_venues_{query_venue.replace(' ', '_')}.png"))
        plt.close()
        
        print(f"Benzer mekanlar grafiği kaydedildi.")
    
    def save_model(self):
        """Modeli kaydetme"""
        # Models klasörünü kontrol et
        if not os.path.exists(self.models_folder):
            os.makedirs(self.models_folder)
        
        # Vektörleştiriciyi kaydet
        if self.vectorizer:
            joblib.dump(self.vectorizer, os.path.join(self.models_folder, "venue_classifier_vectorizer.pkl"))
        
        # Kategori vektörlerini kaydet
        if self.category_vectors:
            joblib.dump(self.category_vectors, os.path.join(self.models_folder, "venue_category_vectors.pkl"))
        
        # Benzerlik matrisini kaydet
        if self.similarity_matrix is not None:
            joblib.dump(self.similarity_matrix, os.path.join(self.models_folder, "venue_similarity_matrix.pkl"))
        
        # Model bilgilerini kaydet
        model_info = {
            'use_lemmatized': self.use_lemmatized,
            'num_categories': len(self.category_vectors) if self.category_vectors else 0,
            'categories': list(self.category_vectors.keys()) if self.category_vectors else [],
            'created_at': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        joblib.dump(model_info, os.path.join(self.models_folder, "venue_classifier_info.pkl"))
        
        print(f"Model başarıyla '{self.models_folder}' dizinine kaydedildi.")
    
    @classmethod
    def load_model(cls, models_folder="models", data=None):
        """
        Kaydedilmiş modeli yükleme
        
        Parameters:
        -----------
        models_folder: str, default="models"
            Modellerin bulunduğu klasör
        data: pd.DataFrame, default=None
            Yeni veri seti (None ise varsayılan veri yüklenir)
        
        Returns:
        --------
        VenueClassifier
            Yüklenmiş model
        """
        # Model bilgilerini yükle
        info_path = os.path.join(models_folder, "venue_classifier_info.pkl")
        
        if not os.path.exists(info_path):
            raise FileNotFoundError(f"Model bilgileri bulunamadı: {info_path}")
        
        model_info = joblib.load(info_path)
        
        # Yeni sınıflandırıcı oluştur
        classifier = cls(data=data, use_lemmatized=model_info['use_lemmatized'], models_folder=models_folder)
        
        # Vektörleştiriciyi yükle
        vectorizer_path = os.path.join(models_folder, "venue_classifier_vectorizer.pkl")
        if os.path.exists(vectorizer_path):
            classifier.vectorizer = joblib.load(vectorizer_path)
        
        # Kategori vektörlerini yükle
        category_vectors_path = os.path.join(models_folder, "venue_category_vectors.pkl")
        if os.path.exists(category_vectors_path):
            classifier.category_vectors = joblib.load(category_vectors_path)
        
        # Benzerlik matrisini yükle
        similarity_matrix_path = os.path.join(models_folder, "venue_similarity_matrix.pkl")
        if os.path.exists(similarity_matrix_path):
            classifier.similarity_matrix = joblib.load(similarity_matrix_path)
        
        print(f"Model başarıyla yüklendi. Desteklenen kategoriler: {model_info['categories']}")
        
        return classifier


def prepare_w2v_corpus(data, use_lemmatized=True):
    """
    Word2Vec eğitimi için korpus hazırlama
    
    Parameters:
    -----------
    data: pd.DataFrame
        İşlenmiş metin verilerini içeren DataFrame
    use_lemmatized: bool, default=True
        Lemmatize edilmiş metinleri kullanma (False ise stemmed metinler kullanılır)
    
    Returns:
    --------
    tuple
        (lemmatized_corpus, stemmed_corpus) içeren tuple
    """
    print("Word2Vec eğitimi için veri hazırlanıyor...")
    
    lemmatized_corpus = []
    stemmed_corpus = []
    
    # DataFrame'den cümleleri al
    if 'processed_reviews_lemmatized.csv' in os.listdir('processed_data'):
        # CSV dosyalarından cümleleri oku
        try:
            lemma_df = pd.read_csv(os.path.join('processed_data', 'processed_reviews_lemmatized.csv'))
            for _, row in tqdm(lemma_df.iterrows(), total=len(lemma_df), desc="Lemmatized cümleler"):
                sentence = row['sentence'].split() if isinstance(row['sentence'], str) else []
                if sentence:
                    lemmatized_corpus.append(sentence)
        except Exception as e:
            print(f"Lemmatized cümleler okunurken hata: {e}")
        
        try:
            stem_df = pd.read_csv(os.path.join('processed_data', 'processed_reviews_stemmed.csv'))
            for _, row in tqdm(stem_df.iterrows(), total=len(stem_df), desc="Stemmed cümleler"):
                sentence = row['sentence'].split() if isinstance(row['sentence'], str) else []
                if sentence:
                    stemmed_corpus.append(sentence)
        except Exception as e:
            print(f"Stemmed cümleler okunurken hata: {e}")
    else:
        # Doğrudan DataFrame'den tokenize et
        for lemma_text, stem_text in zip(
            data['lemmatized_text'] if 'lemmatized_text' in data.columns else [], 
            data['stemmed_text'] if 'stemmed_text' in data.columns else []
        ):
            if isinstance(lemma_text, str) and lemma_text.strip():
                lemmatized_corpus.append(lemma_text.split())
            
            if isinstance(stem_text, str) and stem_text.strip():
                stemmed_corpus.append(stem_text.split())
    
    print(f"Toplam {len(lemmatized_corpus)} lemmatize edilmiş, {len(stemmed_corpus)} stem edilmiş metin hazırlandı.")
    
    return lemmatized_corpus, stemmed_corpus


def main():
    print("Turizm Yorumları Model Eğitimi ve Sınıflandırma Sistemi")
    print("=" * 70)
    
    # Klasör yolları
    data_dir = "processed_data"
    models_dir = "models"
    
    # Veri setini yükleme
    try:
        print("\n1. İşlenmiş Verileri Yükleme")
        print("-" * 70)
        
        data_dir_input = input(f"İşlenmiş verilerin bulunduğu dizin (varsayılan: {data_dir}): ")
        if data_dir_input.strip():
            data_dir = data_dir_input
        
        data_path = os.path.join(data_dir, "processed_reviews.csv")
        
        if not os.path.exists(data_path):
            print(f"Hata: Veri seti bulunamadı - {data_path}")
            print("Lütfen önce 'data_processor.py' kodunu çalıştırarak veri setini hazırlayın.")
            return
        
        print(f"Veri seti yükleniyor: {data_path}")
        df = pd.read_csv(data_path)
        print(f"Veri seti yüklendi: {len(df)} kayıt")
        
        # Kategori dağılımını gösterme
        print("\nKategorilere göre dağılım:")
        category_counts = df['category'].value_counts()
        for category, count in category_counts.items():
            percentage = count / len(df) * 100
            print(f"  - {category}: {count} kayıt ({percentage:.1f}%)")
        
        # TF-IDF Analizi
        print("\n2. TF-IDF Analizi")
        print("-" * 70)
        
        use_lemmatized = input("Lemmatize edilmiş metin kullanmak ister misiniz? (e/h, varsayılan: e): ")
        use_lemmatized = use_lemmatized.lower() != 'h'
        
        # TF-IDF analizi için tfidf_analyzer oluşturma
        print("\nTF-IDF analizi yapılıyor...")
        
        if use_lemmatized:
            texts = df['lemmatized_text'].fillna('').tolist()
            print("Lemmatize edilmiş metinler kullanılıyor...")
        else:
            texts = df['stemmed_text'].fillna('').tolist()
            print("Stem edilmiş metinler kullanılıyor...")
        
        tfidf_analyzer = TFIDFAnalyzer(texts=texts, models_folder=models_dir)
        
        # Benzerlik hesapla
        tfidf_analyzer.calculate_similarity_matrix()
        
        # CSV dosyalarına kaydet
        if use_lemmatized:
            tfidf_analyzer.save_tfidf_matrix_to_csv(os.path.join(data_dir, "tfidf_lemmatized.csv"))
        else:
            tfidf_analyzer.save_tfidf_matrix_to_csv(os.path.join(data_dir, "tfidf_stemmed.csv"))
        
        # Örnek kelimeler için benzerlik analizi
        example_words = ['hotel', 'restaurant', 'food', 'museum', 'experience']
        print("\nTF-IDF ile Kelime Benzerliği Analizi:")
        
        for word in example_words:
            similar_words = tfidf_analyzer.find_similar_words(word)
            if similar_words:
                print(f"\n'{word}' kelimesine benzer kelimeler:")
                for w, score in similar_words:
                    print(f"  {w}: {score:.4f}")
                
                # Görselleştir
                tfidf_analyzer.visualize_word_similarities(word, similar_words)
        
        # Word2Vec Modelleri
        print("\n3. Word2Vec Modelleri Eğitimi")
        print("-" * 70)
        
        train_w2v = input("Word2Vec modellerini eğitmek ister misiniz? (e/h, varsayılan: h): ")
        
        if train_w2v.lower() == 'e':
            # Word2Vec için veri hazırlama
            lemmatized_corpus, stemmed_corpus = prepare_w2v_corpus(df, use_lemmatized)
            
            # Word2Vec trainer oluşturma
            w2v_trainer = Word2VecTrainer(models_folder=models_dir)
            
            # Modelleri eğitme (16 model)
            w2v_models = w2v_trainer.train_all_models(lemmatized_corpus, stemmed_corpus)
            
            # Modelleri test etme
            test_w2v = input("Eğitilen Word2Vec modellerini test etmek ister misiniz? (e/h, varsayılan: h): ")
            
            if test_w2v.lower() == 'e':
                # Model karşılaştırıcı oluştur
                comparator = ModelComparator(models_dir=models_dir)
                
                # Modelleri yükle
                comparator.load_models()
                
                # Kelime benzerliklerini karşılaştır
                test_words = input("Test edilecek kelimeleri virgülle ayırarak girin (varsayılan: hotel,restaurant,food,museum,service): ")
                
                if test_words.strip():
                    test_words_list = [word.strip() for word in test_words.split(",")]
                else:
                    test_words_list = ["hotel", "restaurant", "food", "museum", "service"]
                
                comparator.compare_similar_words(test_words=test_words_list)
                
                # Model parametrelerinin etkisini analiz et
                comparator.analyze_model_parameters()
        
        # Mekan Sınıflandırıcı
        print("\n4. Mekan Sınıflandırıcı Eğitimi")
        print("-" * 70)
        
        # Sınıflandırıcıyı oluştur
        classifier = VenueClassifier(data=df, use_lemmatized=use_lemmatized, models_folder=models_dir)
        
        # Sınıflandırıcıyı eğit
        classifier.train_classifier()
        
        # Sınıflandırıcıyı değerlendir
        try:
            classifier.evaluate_classifier()
        except Exception as e:
            print(f"Sınıflandırıcı değerlendirilirken hata: {e}")
        
        # Modeli kaydet
        classifier.save_model()
        
        # Test arayüzü
        print("\n5. Metin Sınıflandırma Arayüzü")
        print("-" * 70)
        
        while True:
            user_text = input("\nSınıflandırmak istediğiniz metin (çıkmak için 'q'): ")
            
            if user_text.lower() == 'q':
                break
            
            if not user_text.strip():
                continue
            
            # Metni sınıflandır
            try:
                prediction = classifier.classify_venue(user_text)
                
                print(f"\nTahmin edilen kategori: {prediction['predicted_category']}")
                print(f"Güven skoru: {prediction['confidence']:.4f}")
                
                # Tüm kategori skorları
                print("\nTüm kategori skorları:")
                for category, score in prediction['all_scores']:
                    print(f"  - {category}: {score:.4f}")
            except Exception as e:
                print(f"Sınıflandırma sırasında hata: {e}")
        
        print("\nModel eğitimi ve sınıflandırma tamamlandı.")
        print(f"Tüm modeller ve analiz sonuçları '{models_dir}' klasörüne kaydedildi.")
    
    except Exception as e:
        print(f"\nProgram çalıştırılırken hata: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Program çalıştırılırken hata oluştu: {e}")
        import traceback
        traceback.print_exc()