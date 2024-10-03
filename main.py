from pyflakes.checker import counter
from statsmodels.graphics.tukeyplot import results
import numpy as np
from utils.file_loader import load_data
from glass_cut_analysis import GlassCutAnalysis
from overlap_analysis import OverlapAnalysis
import  os

def main():
    # Veri dosyasının yolunu belirtiyoruz
    file_paths = [
        'data/veriler-aynı-1.csv',
        'data/veriler-aynı-2.csv',
        'data/veriler-aynı-3.csv',
        'data/veriler-aynı-4.csv',
        'data/veriler-aynı-5.csv',
        'data/veriler-aynı-6.csv',
        'data/veriler-farklı-1.csv',
        'data/veriler-farklı-2.csv',
        'data/veriler-farklı-3.csv',
        'data/veriler-farklı-4.csv',
        'data/veriler-farklı-5.csv'
    ]
    results = []  # Sonuçları saklayacağımız liste

    for file_path in file_paths:

        file_name = os.path.basename(file_path)
        print(f"\nİşleniyor: {file_name}")

        if os.path.exists(file_path):
            pass
        else:
            print(f"{file_name} bulunamadı.")
        #file_path = 'data/veriler-aynı-3.csv'  # ya da xlsx dosyanız

        # Veriyi yüklüyoruz
        prev_points, curr_points = load_data(file_path)

        # Analiz sınıfını oluşturuyoruz
        glass_analysis = GlassCutAnalysis(prev_points, curr_points)

        # Verileri gösteriyoruz
        glass_analysis.show_data()

        # Öklid mesafelerini hesaplıyoruz
        distances = glass_analysis.calculate_euclidean_distances()
        #print("\nÖklid Mesafeleri:")
        #for i, dist in enumerate(distances):
            #print(f"Nokta {i + 1}: Öklid Mesafesi = {dist:.2f}")

        if distances:  # Mesafe listesi boş değilse
            mean_distance = np.mean(distances)
            std_distance = np.std(distances)
            print(f"\n{file_name} için Ortalama Öklid Mesafesi: {mean_distance:.2f}")
            print(f"{file_name} için Standart Sapma: {std_distance:.2f}")

        # Serilerin aynı olup olmadığını kontrol ediyoruz
        same_series = glass_analysis.compare_series(threshold=61.0)
        if same_series:
            print(f"\n{file_name} dosyası için cam kesimleri aynı seride!")
        else:
            print(f"\n{file_name} dosyası için cam kesimleri farklı seride!")

        # Örtüşme analizi yapıyoruz
        overlap_analysis = OverlapAnalysis(prev_points, curr_points)
        overlaps = overlap_analysis.calculate_overlap(threshold=5.0)

        if "aynı" in file_path:
          expected_same = True
        elif "farklı" in file_path:
          expected_same = False
        else:
           print(f"{file_name} dosyası için beklenmeyen bir isim formatı.")
           continue  # Devam e1tmek için döngünün başına döner

        results.append({
            "file_name": file_name,
            "same_series": same_series,
            "overlaps": overlaps
        })

    print("\nSonuçlar:")
    for result in results:
        file_name = result["file_name"]
        same_series = result["same_series"]
        overlaps = result["overlaps"]

        # Serinin aynı olup olmadığını kontrol et
        if same_series:
            series_message = f"{file_name} dosyası için cam kesimleri aynı seride."
        else:
            series_message = f"{file_name} dosyası için cam kesimleri farklı seride."

        # Örtüşmeleri kontrol et
        if overlaps:
            overlap_count = len(overlaps)  # Örtüşen noktaların sayısını hesapla
            overlap_message = f"{file_name} dosyasında {overlap_count} örtüşen noktalar bulundu:"

            for p1, p2 in overlaps:
                overlap_message += f"\n  Prev: {p1}, Curr: {p2}"

        else:
            overlap_message = f"{file_name} dosyasında örtüşme yok."

        # Sonuçları birleştir ve yazdır
        print(f"\n{series_message}\n{overlap_message}")

if __name__ == "__main__":
    main()
