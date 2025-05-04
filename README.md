# Turizm Yorumları Doğal Dil İşleme (NLP) Projesi

## 📋 Proje Özeti

Bu proje, Yelp platformundan elde edilen turizm yorumlarını kullanarak çeşitli doğal dil işleme teknikleri uygulayan kapsamlı bir çalışmadır. Proje kapsamında, metin ön işleme, TF-IDF ve Word2Vec vektörleştirme ve metin sınıflandırma işlemleri gerçekleştirilmiştir. Farklı parametre setleriyle eğitilen Word2Vec modelleri karşılaştırılmış ve metin verilerindeki örüntüler analiz edilmiştir.

## 🔍 Veri Seti Hakkında

Projede kullanılan veri seti, Yelp Academic Dataset'ten (`yelp_academic_dataset_review.json`) alınan ve turizm kategorilerine göre filtrelenmiş yorumlardan oluşmaktadır:

- **Toplam kayıt sayısı**: 21,773 yorum
- **Kategori dağılımı**:
  - Restoran: 16,003 yorum (%73.5)
  - Otel: 3,948 yorum (%18.1)
  - Müze: 662 yorum (%3.0)
  - Turistik mekan (attraction): 599 yorum (%2.8)
  - Park: 561 yorum (%2.6)
- **İşlenmemiş ham veri boyutu**: ~50MB
- **İşlenmiş veri boyutu**: 
  - Lemmatized: ~30MB 
  - Stemmed: ~25MB

## 🛠️ Gerekli Kütüphaneler ve Kurulum

Projeyi çalıştırmak için aşağıdaki kütüphanelere ihtiyacınız vardır:

```bash
pip install pandas numpy nltk scikit-learn matplotlib seaborn tqdm gensim joblib
```

NLTK veri paketlerinin indirilmesi:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

## 📁 Proje Yapısı

```
.
├── data_processor.py           # Veri işleme ve ön işleme kodu
├── model_trainer.py            # Model eğitimi ve değerlendirme kodu
├── temiz.py                    # Yardımcı temizleme fonksiyonları
├── processed_data/             # İşlenmiş veri dosyaları
│   ├── processed_reviews.csv   # Temel işlenmiş veri
│   ├── processed_reviews_lemmatized.csv
│   ├── processed_reviews_stemmed.csv
│   ├── tfidf_lemmatized.csv
│   ├── tfidf_stemmed.csv
│   └── category_distribution.png
├── zipf_graphs/                # Zipf yasası analiz grafikleri
│   ├── zipf_comparison.png
│   ├── zipf_ham_veri.png
│   ├── zipf_lemmatize_edilmis_veri.png
│   └── zipf_stem_edilmis_veri.png
└── models/                     # Modeller ve analiz grafikleri
    ├── accuracy.png
    ├── classification_metrics.png
    ├── confusion_matrix.png
    ├── model_type_comparison.png
    ├── vector_size_comparison.png
    ├── window_size_comparison.png
    ├── w2v_model_stats.csv
    ├── w2v_training_times.png
    ├── w2v_model_sizes.png
    ├── w2v_vocabulary_sizes.png
    ├── word_similarities_*.png  # Kelime benzerlik grafikleri
    ├── venue_category_vectors.pkl
    ├── venue_classifier_info.pkl
    ├── venue_classifier_vectorizer.pkl
    └── venue_similarity_matrix.pkl
```

## 🔄 Adım Adım Kullanım

### 1. Veri İşleme

```bash
python data_processor.py
```

Bu komut çalıştırıldığında:
- Sizden `yelp_academic_dataset_review.json` dosyasının yolunu ister (varsayılan: `C:\Users\gokmu\Desktop\ddisleme\yelp_academic_dataset_review.json`)
- Çıktı dizinini belirtmenizi ister (varsayılan: `processed_data`)
- Maksimum dosya boyutunu MB cinsinden belirtmenizi ister (varsayılan: 50)
- İşlenecek maksimum kayıt sayısını belirtmenizi ister (sınırlama olmaması için boş bırakabilirsiniz)

İşlem sonucunda aşağıdaki dosyalar oluşturulur:
- `processed_reviews.csv`: Temel işlenmiş yorumlar
- `processed_reviews_lemmatized.csv`: Lemmatization uygulanmış yorumlar
- `processed_reviews_stemmed.csv`: Stemming uygulanmış yorumlar
- Zipf yasası analiz grafikleri

### 2. Model Eğitimi ve Analiz

```bash
python model_trainer.py
```

Bu komut çalıştırıldığında interaktif bir şekilde:

1. **Veri Yükleme**
   - İşlenmiş verilerin bulunduğu dizini sorar (varsayılan: `processed_data`)
   - Veri seti yüklenir ve kategori dağılımı gösterilir

2. **TF-IDF Analizi**
   - Lemmatized veya stemmed metinleri kullanmak isteyip istemediğinizi sorar
   - TF-IDF vektörleri oluşturulur
   - Benzerlik matrisi hesaplanır
   - Belirtilen örnek kelimeler için benzerlik analizi yapılır

3. **Word2Vec Modelleri Eğitimi**
   - Word2Vec modellerini eğitmek isteyip istemediğinizi sorar
   - İsteğe bağlı olarak, aşağıdaki 16 parametre setiyle modeller eğitilir:
     - 8 adet lemmatized veri seti için model
     - 8 adet stemmed veri seti için model
   - Her model şu parametrelerle eğitilir:
     ```
     {'model_type': 'cbow', 'window': 2, 'vector_size': 100}
     {'model_type': 'skipgram', 'window': 2, 'vector_size': 100}
     {'model_type': 'cbow', 'window': 4, 'vector_size': 100}
     {'model_type': 'skipgram', 'window': 4, 'vector_size': 100}
     {'model_type': 'cbow', 'window': 2, 'vector_size': 300}
     {'model_type': 'skipgram', 'window': 2, 'vector_size': 300}
     {'model_type': 'cbow', 'window': 4, 'vector_size': 300}
     {'model_type': 'skipgram', 'window': 4, 'vector_size': 300}
     ```
   - Eğitilen modelleri test etmek isteyip istemediğinizi sorar
   - Test kelimeleri belirtmeniz istenir (varsayılan: hotel, restaurant, food, museum, service)
   - Model karşılaştırma grafikleri oluşturulur

4. **Metin Sınıflandırıcı Eğitimi**
   - VenueClassifier sınıfı kullanılarak metin sınıflandırıcı eğitilir
   - Sınıflandırıcı değerlendirilir ve sonuçlar gösterilir
   - Model kaydedilir

5. **Metin Sınıflandırma Arayüzü**
   - İnteraktif bir arayüz ile istediğiniz metinleri sınıflandırabilirsiniz
   - Çıkmak için 'q' yazabilirsiniz

### 3. Oluşturulan Modeller

**Not**: Word2Vec model dosyaları (`.model` uzantılı dosyalar) ve bazı büyük boyutlu veri dosyaları, GitHub'ın dosya boyutu kısıtlamaları nedeniyle bu repoya yüklenmemiştir. Bu modelleri oluşturmak için `model_trainer.py` kodunu çalıştırmanız gerekir.

Eğitim sonrasında aşağıdaki model dosyaları oluşturulacaktır:

1. **TF-IDF Modelleri**:
   - `tfidf_vectorizer_max5000_min2.pkl`: TF-IDF vektörleştirici
   - `similarity_matrix.pkl`: Benzerlik matrisi

2. **Word2Vec Modelleri** (16 farklı model):
   - `lemmatized_model_cbow_window2_dim100.model`
   - `lemmatized_model_skipgram_window2_dim100.model`
   - `lemmatized_model_cbow_window4_dim100.model`
   - ...ve diğerleri

3. **Sınıflandırıcı Modelleri**:
   - `venue_classifier_vectorizer.pkl`: Sınıflandırıcı için vektörleştirici
   - `venue_category_vectors.pkl`: Kategori vektörleri
   - `venue_classifier_info.pkl`: Model bilgileri
   - `venue_similarity_matrix.pkl`: Benzerlik matrisi

### 4. Analiz Grafikleri

Eğitim sonrasında oluşturulan örnek grafikler:

1. **Zipf Analizi Grafikleri**:
   - `zipf_ham_veri.png`: Ham veri Zipf analizi
   - `zipf_lemmatize_edilmis_veri.png`: Lemmatize edilmiş veri Zipf analizi
   - `zipf_stem_edilmis_veri.png`: Stem edilmiş veri Zipf analizi
   - `zipf_comparison.png`: Karşılaştırmalı Zipf analizi

2. **Word2Vec Model Karşılaştırma Grafikleri**:
   - `model_type_comparison.png`: CBOW vs SkipGram karşılaştırması
   - `window_size_comparison.png`: Pencere boyutu karşılaştırması
   - `vector_size_comparison.png`: Vektör boyutu karşılaştırması
   - `w2v_vocabulary_sizes.png`: Vokabüler boyutları
   - `w2v_training_times.png`: Eğitim süreleri
   - `w2v_model_sizes.png`: Model boyutları

3. **Kelime Benzerlik Grafikleri**:
   - `word_similarities_hotel.png`: "Hotel" kelimesi için benzerlikler
   - `word_similarities_restaurant.png`: "Restaurant" kelimesi için benzerlikler
   - ... (diğer kelimeler için benzerlik grafikleri)
   - `word_similarities_comparison_hotel.png`: "Hotel" kelimesi için model karşılaştırması
   - ... (diğer kelimeler için model karşılaştırmaları)

4. **Sınıflandırıcı Performans Grafikleri**:
   - `confusion_matrix.png`: Karışıklık matrisi
   - `classification_metrics.png`: Sınıflandırma metrikleri
   - `accuracy.png`: Doğruluk grafiği

## 📊 Temel Bulgular

### 1. Veri Seti Analizi
- Veri seti kategori dağılımı oldukça dengesizdir (restoran %73.5, otel %18.1)
- Temizleme işlemi sonrasında, veri boyutunda %37-48 oranında küçülme görülmüştür
- Zipf yasası analizleri, veri setinin doğal dil özelliklerini göstermektedir

### 2. TF-IDF Analizi Sonuçları
- TF-IDF, kelimeler arasındaki semantik ilişkileri başarıyla yakalamıştır
- Örnek: "hotel" kelimesi "room", "stay", "bed" gibi kelimelerle yüksek benzerlik göstermiştir
- Örnek: "restaurant" kelimesi "food", "service", "good" gibi kelimelerle ilişkilendirilmiştir

### 3. Word2Vec Model Karşılaştırmaları
- **Model Türü**: SkipGram modelleri nadir kelimeler için daha iyi temsil sağlamıştır
- **Pencere Boyutu**: Daha büyük pencere boyutu (4), daha geniş bağlamsal ilişkileri yakalamıştır
- **Vektör Boyutu**: 300 boyutlu vektörler daha zengin temsiller sağlar, ancak hesaplama maliyeti yüksektir
- **Ön İşleme Tekniği**: Lemmatization, stemming'e göre daha zengin bir vokabüler sağlamıştır (30,349 vs 23,554)

### 4. Sınıflandırma Performansı
- Genel doğruluk: %78.42
- Kategori bazında performans:
  - Restoran: Precision 0.95, Recall 0.81, F1-score 0.88
  - Otel: Precision 0.78, Recall 0.66, F1-score 0.71
  - Müze: Precision 0.27, Recall 0.83, F1-score 0.41
  - Park: Precision 0.36, Recall 0.83, F1-score 0.51
  - Turistik mekan: Precision 0.32, Recall 0.79, F1-score 0.46
- Veri setindeki dengesizlik, sınıflandırma performansını etkilemiştir

## 🔮 Sonuç ve İleriye Dönük Çalışmalar

Bu proje, turizm yorumlarını analiz etmek ve sınıflandırmak için çeşitli NLP tekniklerinin etkinliğini göstermiştir. Temel bulgular şunlardır:

1. Lemmatization, metin temsillerinde stemming'e göre daha zengin bir vokabüler sağlar
2. Word2Vec modellerinde SkipGram, CBOW'a göre nadir kelimeler için daha iyi temsiller üretir
3. Veri setindeki dengesizlik, bazı kategorilerde (müze, park, turistik mekan) düşük precision değerlerine yol açmıştır

İleriye dönük çalışmalar:
- Daha dengeli bir veri seti oluşturulması
- Transfer öğrenimi ile önceden eğitilmiş dil modellerinin (BERT, RoBERTa vb.) kullanılması
- Duygu analizi eklenerek yorumların pozitif/negatif olarak sınıflandırılması
- Daha gelişmiş sınıflandırma algoritmaları kullanılması (XGBoost, derin öğrenme modelleri vb.)

## 🚀 Projeyi Kendi Ortamınızda Oluşturma

1. Bu repoyu klonlayın:
```bash
git clone https://github.com/kullaniciadi/tourism-reviews-nlp-project.git
cd tourism-reviews-nlp-project
```

2. Gerekli kütüphaneleri yükleyin:
```bash
pip install -r requirements.txt
```

3. Veri işleme adımını çalıştırın:
```bash
python data_processor.py
```

4. Model eğitim sürecini başlatın:
```bash
python model_trainer.py
```

## 📝 Not

Bu projede oluşturulan model dosyaları (özellikle Word2Vec .model dosyaları), GitHub'ın dosya boyutu kısıtlamaları nedeniyle repoya yüklenmemiştir. Modelleri oluşturmak için lütfen `model_trainer.py` dosyasını çalıştırınız. Eğitim süreci, bilgisayarınızın özelliklerine bağlı olarak yaklaşık 20-30 dakika sürebilir.

---

© 2025 • Doğal Dil İşleme Dersi Ödevi