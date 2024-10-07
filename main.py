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

    # Verileri yüklemek için FileLoader sınıfını kullanıyoruz
    directory = 'data/'  # Eğitim verilerinin bulunduğu klasör
    unlabeled_directory = 'unlabeled_data/'  # Tahmin yapılacak etiketsiz verilerin klasörü
    file_loader = FileLoader(directory) # Eğitim verilerini yükle
    file_paths = file_loader.get_file_paths() # Tüm dosya yollarını al

    unlabeled_file_loader = FileLoader(unlabeled_directory)  # Etiketsiz veriler için ayrı dosya yükleyici
    unlabeled_file_paths = unlabeled_file_loader.get_file_paths()  # Etiketsiz verilerin dosya yolları

    results_manager = ResultsManager() # Sonuçları yönetecek ResultsManager sınıfını başlatıyoruz
    results_manager_2= ResultsManager()  # Etiketsiz veriler için ayrı sonuç yöneticisi


    output_path = "results/visualizations"  #Görsel çıktılar için dosya yolu
    # Eğitim verileri üzerinde analiz yap
    for file_path in file_paths:
        file_name = os.path.basename(file_path) # Dosya adını al
        print(f"\nİşleniyor: {file_name}")

        if os.path.exists(file_path):
            try:
                print(f"{file_name} dosyası yükleniyor...")
                prev_points, curr_points = load_data(file_path)  # Veriyi yükle ve işaret noktalarını al
                print(f"{file_name} dosyası yüklendi. Toplam {len(prev_points)} previous nokta ve {len(curr_points)} current nokta bulundu.")
            except Exception as e:
                print(f"Veri yüklenirken hata oluştu: {e}")
                continue  # Hata oluşursa bu dosyayı atla
        else:
            print(f"{file_name} bulunamadı.")
            continue

        glass_analysis = GlassCutAnalysis(prev_points, curr_points,file_name) # Analiz sınıfını oluşturuyoruz
        distances = glass_analysis.calculate_euclidean_distances() # Öklid mesafelerini hesaplıyoruz

        if distances:  # Mesafeler boş değilse istatistikleri hesapla
            mean_distance, std_distance = glass_analysis.calculate_statistics()
        else:
            mean_distance, std_distance = None, None  # Mesafe hesaplanmadıysa None atayın

        same_series_value = glass_analysis.label_same_series() # Seri etiketi

        # Şekil ve Fourier analizlerini yap
        shaper_analysis = ShaperAnalysis(prev_points, curr_points, file_name, output_path)
        shaper_result = shaper_analysis.analyze_data() # Şekil analizi - intersection_area, union_area, iou, series_label,
        fourier_result = shaper_analysis.apply_fourier_transform() # Fourier analizi

        shaper_result["fourier_analysis"] = fourier_result  # Sonuçları birleştir

        # açı analizleri
        angle_analysis = glass_analysis.analyze_angle_similarity() # # Açı benzerliği analizi - mean_prev,std_prev, mean_curr, std_curr,  mse, similarity_score
        # Sonuçları kaydet
        results_manager.add_result(file_name, mean_distance, std_distance, shaper_result, angle_analysis,fourier_result,same_series_value)

    # Eğitim verilerinin sonuçlarını kaydet
    output_file = "results/analysis/feature_extraction_output.csv"  # Dataset üzerinden özellik çıkarımı yapılan csv dosyası
    results_manager.save_results_to_csv(output_file)
    print(f"\nSonuçlar {output_file} dosyasına kaydedildi.")

    # Etiketsiz veriler için analiz işlemleri
    for file_path in unlabeled_file_paths:
        file_name = os.path.basename(file_path)  # Dosya adı
        print(f"\nİşleniyor: {file_name} (etiketsiz)")

        if os.path.exists(file_path):
            try:
                print(f"{file_name} dosyası yükleniyor...")
                prev_points, curr_points = load_data(file_path)  # noktalar ayrıştırılır ve liste şeklinde elde edilir
                print(
                    f"{file_name} dosyası yüklendi. Toplam {len(prev_points)} previous nokta ve {len(curr_points)} current nokta bulundu.")
            except Exception as e:
                print(f"Veri yüklenirken hata oluştu: {e}")
                continue  # Hata oluşursa bu dosyayı atla
        else:
            print(f"{file_name} bulunamadı.")
            continue

        glass_analysis = GlassCutAnalysis(prev_points, curr_points, file_name)  # Analiz sınıfını oluşturuyoruz
        distances = glass_analysis.calculate_euclidean_distances()  # Öklid mesafelerini hesaplıyoruz

        if distances:  # Mesafe listesi boş değilse
            mean_distance, std_distance = glass_analysis.calculate_statistics()
        else:
            mean_distance, std_distance = None, None  # Mesafe hesaplanmadıysa None atayın

        # Shaper ve Fourier analiz sonuçlarını saklamak için varsayılan değerler atayın
        shaper_result = {}
        fourier_result = {}
        angle_analysis = {}

        # Eğer gerekli analizler yapılmışsa sonuçları atayın
        if prev_points and curr_points:  # Noktalar varsa
            shaper_analysis = ShaperAnalysis(prev_points, curr_points, file_name, output_path)
            shaper_result = shaper_analysis.analyze_data()  # intersection_area, union_area, iou, series_label,
            fourier_result = shaper_analysis.apply_fourier_transform()
            angle_analysis = glass_analysis.analyze_angle_similarity()  # mean_prev,std_prev, mean_curr, std_curr, mse, similarity_score

        # Etiketsiz veriler için sonuçları ekle
        results_manager_2.add_result(file_name, mean_distance, std_distance, shaper_result, angle_analysis,fourier_result, None)

    # Etiketsiz verilerin sonuçlarını kaydet
    unlabeled_output_file = "results/analysis/unlabeled_features_output.csv"
    results_manager_2.save_results_to_csv(unlabeled_output_file)
    print(f"\nEtiketsiz veriler {unlabeled_output_file} dosyasına kaydedildi.")

    # Model eğitimi ve test
    trainer = ModelTrainer(output_dir="results/visualizations/model",model_choice="random_forest")  # Yeni eğitim sınıfı
    analysis_data = pd.read_csv(output_file)
    trainer.run_training(analysis_data)  # CSV'den yüklenen verilerle eğitimi başlat

    X, y = trainer.preprocess_data(analysis_data)  # Veriyi işle
    trainer.cross_validate(X, y, cv=10)  # Çapraz doğrulamayı çalıştır

    # Etiketsiz veriler üzerinde tahmin yap

    unlabeled_data = pd.read_csv(unlabeled_output_file)
    X_unlabeled, _ = trainer.preprocess_data(unlabeled_data)  # Hedef değişkeni kullanmadığımız için _ ile aldık

    predictions = trainer.model.predict(X_unlabeled)  # Model üzerinden tahmin yap

    # Tahmin sonuçlarını kaydet
    results_df = pd.DataFrame(predictions, columns=["predictions"])
    results_df['source_file'] = [os.path.basename(path) for path in unlabeled_file_paths]  #Tahminin yapıldığı dosya adını ekliyoruz
    results_df['prediction_label'] = results_df['predictions'].apply(lambda x: 'aynı seri' if x == 1 else 'farklı seri')
    results_df = results_df[['source_file', 'predictions', 'prediction_label']]  # Sütun sırasını ayarla


    results_csv_path = "results/predictions_unlabeled.csv"  # Sonuçların kaydedileceği dosya yolu
    results_df.to_csv(results_csv_path, index=False)
    print(f"Tahmin sonuçları {results_csv_path} dosyasına kaydedildi.")



    # Toplam süreyi hesapla
    end_time = time.time()  # Bitiş zamanını al
    total_time = end_time - start_time  # Toplam süreyi hesapla
    print(f"Toplam çalışma süresi: {total_time:.2f} saniye")
if __name__ == "__main__":
    main()
