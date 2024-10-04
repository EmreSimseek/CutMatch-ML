
from utils.file_loader import load_data, FileLoader
from glass_cut_analysis import GlassCutAnalysis
from shape_analyzer import  ShaperAnalysis
from model_train import  ModelTrainer
import  os
import pandas as pd
def main():
    # Veri dosyasını file_loader ile yüklüyoruz
    file_loader = FileLoader()
    file_paths = file_loader.get_file_paths()
    results = []

    for file_path in file_paths:

        # Veriyi yüklüyoruz
        file_name = os.path.basename(file_path)
        print(f"\nİşleniyor: {file_name}")

        if os.path.exists(file_path):
            pass
        else:
            print(f"{file_name} bulunamadı.")
        # 2 farklı cam için koordinat değerlerini ayırıyoruz
        prev_points, curr_points = load_data(file_path)

        # Analiz sınıfını oluşturuyoruz
        glass_analysis = GlassCutAnalysis(prev_points, curr_points,file_name)

        # Verileri gösteriyoruz - glass_analysis.show_data()

        # Öklid mesafelerini hesaplıyoruz
        distances = glass_analysis.calculate_euclidean_distances()


        if distances:  # Mesafe listesi boş değilse
            mean_distance, std_distance = glass_analysis.calculate_statistics()
            print(f"\n{file_name} için Ortalama Öklid Mesafesi: {mean_distance:.2f}")
            print(f"{file_name} için Standart Sapma: {std_distance:.2f}")
        else:
            mean_distance, std_distance = None, None  # Mesafe hesaplanmadıysa None atayın
        same_series_value = glass_analysis.label_same_series()
        print(f"Same Series Değeri: {same_series_value}")


        # Çizim ve IoU değerine göre seri tespiti
        output_path = "results"
        shaper_analysis = ShaperAnalysis(prev_points, curr_points, file_name, output_path)
        shaper_result = shaper_analysis.analyze_data()

        # açı analizleri
        angle_analysis = glass_analysis.analyze_angle_similarity()
        # Sonuçları listeye ekle
        if shaper_result:
            results.append({
                "file_name": file_name,
                "mean_distance": f"{mean_distance:.3f}" if mean_distance is not None else None,
                "std_distance": f"{std_distance:.3f}" if std_distance is not None else None,
                "intersection_area": f"{shaper_result['intersection_area']:.3f}",
                "union_area": f"{shaper_result['union_area']:.3f}",
                "iou": f"{shaper_result['iou']:.3f}",
                "angle_std_prev": f"{angle_analysis['std_prev']:.3f}",
                "angle_mean_curr": f"{angle_analysis['mean_curr']:.3f}",
                "angle_std_curr": f"{angle_analysis['std_curr']:.3f}",
                "angle_mse": f"{angle_analysis['mse']:.3f}",
                "similarity_score": f"{angle_analysis['similarity_score']:.3f}",
                "same_series_value": same_series_value
            })

    # Sonuçları bir DataFrame'e dönüştür
    results_df = pd.DataFrame(results)

    # DataFrame'i bir CSV dosyasına kaydet
    output_file = "data/analysis_results.csv"
    results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\nSonuçlar {output_file} dosyasına kaydedildi.")

    # Model eğitim ve test aşaması
    trainer = ModelTrainer(output_dir="results")  # Yeni eğitim sınıfı
    analysis_data = pd.read_csv(output_file)
    trainer.run_training(analysis_data)  # CSV'den yüklenen verilerle eğitimi başlat

    X, y = trainer.preprocess_data(analysis_data)  # Veriyi işle
    trainer.cross_validate(X, y, cv=5)  # Çapraz doğrulamayı çalıştır


if __name__ == "__main__":
    main()
