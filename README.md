# CutMatchML - Oto Cam Kesiminde Seri Değişimini Tespit Etme

Bu proje, **oto cam fabrikasındaki cam kesim makinesinin** farklı zamanlardaki işlemlerini analiz ederek aynı seriye ait olup olmadığını belirlemeyi amaçlar. **Cam kesim makinesinden kaydedilen hareket noktaları** incelenmiş, veri işleme, özellik çıkarımı, model eğitimi ve sonuç değerlendirme adımları gerçekleştirilmiştir.

---

## İçindekiler
1. [Proje Hakkında](#proje-hakkında)
2. [Veri Hazırlığı](#veri-hazırlığı)
3. [Özellik Çıkarımı](#özellik-çıkarımı)
4. [Model Eğitimi](#model-eğitimi)
5. [Sonuçların Analizi ve Raporlama](#sonuçların-analizi-ve-raporlama)
6. [Sonuç](#sonuç)

---

## Proje Hakkında
Bu proje, **cam kesim makinesinde farklı kesimlerin benzerliğini veya farklılığını** analiz ederek işlemlerin aynı seriye ait olup olmadığını tespit eder.

- Her bir kesim, makinenin kaydettiği `(x, y)` koordinatlarından oluşan nokta kümeleri olarak temsil edilir.
- **"Prev"** sütunu, önceki kesime ait hareket noktalarını; **"Curr"** sütunu, mevcut kesim işlemi sırasında kaydedilen noktaları içerir.

---

## Veri Hazırlığı
1. **Veri Ön İşleme:**  
   - Eksik veya hatalı veriler temizlenir.  
   - Veriler modele uygun formata getirilir.

2. **Veri Yükleme:**  
   - **`FileLoader`** sınıfı, veri setini yüklemek için kullanılır.  
   - **`load_data()`** fonksiyonu, her bir dosya yolundan **önceki (prev_points)** ve **mevcut (curr_points)** kesim noktalarını yükler.

---

## Özellik Çıkarımı
Kesimler arasındaki benzerlik ve farklılıkların tespit edilmesi için çeşitli özellikler çıkarılır:

- **Öklid Mesafesi:** İki nokta arasındaki mesafe.
- **Ortalama Mesafe (Mean Distance):** Öklid mesafelerinin ortalaması.
- **Standart Sapma:** Mesafelerin tutarlılığını gösterir.
- **Seri Etiketi:** İki kesimin aynı seriye ait olup olmadığını belirtir.
- **IoU (Intersection over Union):** İki nokta kümesinin kesişim ve birleşim oranı.
- **Fourier Analizi:** Kesim şekillerinin frekans bileşenleri.
- **Açı Analizi:** Noktaların açıları arasındaki benzerlik ve farklar.

---

## Model Eğitimi
Model eğitimi için **`ModelTrainer`** sınıfı kullanılır:

1. **Veri Ön İşleme:**  
   - Eğitim ve test setleri oluşturulur.  
2. **Model Seçimi:**  
   - **Random Forest** veya **SVM** modelleri kullanılır.  
   - **GridSearchCV** ile hiperparametre optimizasyonu yapılır.  
3. **Çapraz Doğrulama:**  
   - K-Fold yöntemi ile doğruluk değerlendirilir.  
4. **Özellik Önemi:**  
   - Modelin hangi özelliklere daha fazla önem verdiği analiz edilir.

---

## Sonuçların Analizi ve Raporlama
1. **Tahmin Yapma:**  
   - Etiketsiz veriler üzerinde tahmin yapmak için **`predict()`** metodu kullanılır.

2. **Sonuçların Kaydedilmesi:**  
   - Tahminler bir CSV dosyasına yazılır.

3. **Sonuçların Görselleştirilmesi:**  
   - **Confusion Matrix** ile tahminlerin doğruluğu analiz edilir.  
   - Özellik önem analizi görsellerle sunulur.

4. **Raporlama:**  
   - Modelin doğruluk oranı ve tahmin sonuçları raporlanır.

---

## Sonuç
Bu proje, cam kesim işlemlerinin **seri değişimlerini tespit etmek** için makine öğrenimi algoritmalarını başarıyla kullanmıştır. Veri hazırlığından model eğitimine kadar her adım detaylı bir şekilde gerçekleştirilmiş ve uygulama sonuçları raporlanmıştır.
