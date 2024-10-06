from utils.file_loader import load_data, FileLoader
from glass_cut_analysis import GlassCutAnalysis
from shape_analyzer import  ShaperAnalysis
from model_train import  ModelTrainer
from results_manager import  ResultsManager
import  time
import  os
import pandas as pd


def main():
    start_time = time.time()  # Başlangıç zamanını kaydet

    # Veri dosyasını file_loader ile yüklüyoruz
    directory = 'data/'  # Klasör yolunu buraya gir
    file_loader = FileLoader(directory) #klasördeki dosyaları bulmak ve daha sonra işlem yapmak için kullanılıyor.
    file_paths = file_loader.get_file_paths() #data/veriler-aynı-1.csv şeklinde dataset listesini tutar   fonksiyon dosya yollarını return eder


    results_manager = ResultsManager() # Sonuçları yönetecek ResultsManager sınıfını başlatıyoruz

    # Her dosya için analiz işlemlerini gerçekleştiriyoruz
    for file_path in file_paths:
        # Veriyi yüklüyoruz
        file_name = os.path.basename(file_path) # Dosya adı
        print(f"\nİşleniyor: {file_name}")

        if os.path.exists(file_path):
            try:
                print(f"{file_name} dosyası yükleniyor...")
                prev_points, curr_points = load_data(file_path)  #noktalar ayrıştırılır ve liste şeklind elde edilir
                print(f"{file_name} dosyası yüklendi. Toplam {len(prev_points)} previous nokta ve {len(curr_points)} current nokta bulundu.")
            except Exception as e:
                print(f"Veri yüklenirken hata oluştu: {e}")
                continue  # Hata oluşursa bu dosyayı atla
        else:
            print(f"{file_name} bulunamadı.")
            continue

        glass_analysis = GlassCutAnalysis(prev_points, curr_points,file_name) # Analiz sınıfını oluşturuyoruz
        distances = glass_analysis.calculate_euclidean_distances() # Öklid mesafelerini hesaplıyoruz

        if distances:  # Mesafe listesi boş değilse
            mean_distance, std_distance = glass_analysis.calculate_statistics()
        else:
            mean_distance, std_distance = None, None  # Mesafe hesaplanmadıysa None atayın

        same_series_value = glass_analysis.label_same_series()

        # Çizim ve IoU değerine göre seri tespiti
        output_path = "results/visualizations/plots"
        shaper_analysis = ShaperAnalysis(prev_points, curr_points, file_name, output_path)
        shaper_result = shaper_analysis.analyze_data() #  intersection_area, union_area, iou, series_label,

        # açı analizleri
        angle_analysis = glass_analysis.analyze_angle_similarity() # mean_prev,std_prev, mean_curr, std_curr,  mse, similarity_score
        # Tüm sonuçlar listeye toplanıyor
        results_manager.add_result(file_name, mean_distance, std_distance, shaper_result, angle_analysis,same_series_value)

    output_file = "results/analysis/output_analysis.csv"  # Dataset üzerinden özellik çıkarımı yapılan csv dosyası
    results_manager.save_results_to_csv(output_file)
    print(f"\nSonuçlar {output_file} dosyasına kaydedildi.")

    # 1.Model eğitim ve test aşaması
    trainer = ModelTrainer(output_dir="results/visualizations/model",model_choice="svm")  # Yeni eğitim sınıfı
    analysis_data = pd.read_csv(output_file)
    trainer.run_training(analysis_data)  # CSV'den yüklenen verilerle eğitimi başlat

    X, y = trainer.preprocess_data(analysis_data)  # Veriyi işle
    trainer.cross_validate(X, y, cv=5)  # Çapraz doğrulamayı çalıştır

    # 2.Model eğitim ve test aşaması
    # Hazır bir analiz CSV dosyasını yükleyin

    analysis_data2 = pd.read_csv('data/generated_100_variations.csv')  # Gpt ile oluşturulmuş veriler
    trainer = ModelTrainer(output_dir="results/visualizations/model",model_choice="random_forest")
    X, y = trainer.preprocess_data(analysis_data2)  # Veriyi eğitime hazırlayın
    trainer.run_training(analysis_data2) # Eğitim ve değerlendirme işlemlerini başlat


    # Çapraz doğrulama yaparak modelin performansını daha detaylı inceleyin
    trainer.cross_validate(X, y, cv=5)  # 5 katlı çapraz doğrulama
    print("Model eğitimi ve değerlendirme tamamlandı.")


    end_time = time.time()  # Bitiş zamanını al
    total_time = end_time - start_time  # Toplam süreyi hesapla
    print(f"Toplam çalışma süresi: {total_time:.2f} saniye")

    new_data_input = 'results/analysis/output_analysis2.csv'  # Yeni verilerin yer aldığı CSV dosyasının yolu

    # Yeni verileri CSV dosyasından oku
    try:
        new_data = pd.read_csv(new_data_input, header=None)  # Başlık olmadan veri yükle
        print(f"{new_data_input} dosyası başarıyla yüklendi.")
    except Exception as e:
        print(f"Yeni veriler yüklenirken hata oluştu: {e}")
        return  # Hata oluşursa işlemi durdur

    # Tahmin yap
    predictions = trainer.predict(new_data)

    # Tahmin sonuçlarını yazdır
    print("Tahmin Sonuçları:", end="")
    print(predictions)

if __name__ == "__main__":
    main()
