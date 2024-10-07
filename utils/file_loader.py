import pandas as pd
import os

class FileLoader:   # Bu sınıf, belirtilen dizindeki CSV dosyalarını yükler ve belirli dosyaları atlar.
    def __init__(self, directory):
        """Belirtilen klasördeki tüm CSV ve XLSX dosyalarını alır, ancak 'analysis_results.csv' dosyasını atlar."""
        self.directory = directory
        self.file_paths = self.get_file_paths() # Bu sınıf, belirtilen dizindeki CSV ve XLSX dosyalarını yükler ve belirli dosyaları atlar.

    # Klasördeki tüm CSV dosyalarının tam yollarını döndürür
    def get_file_paths(self):
        file_paths = []
        for file_name in os.listdir(self.directory):  # Klasördeki tüm dosyaları listele
            if file_name.endswith('.csv'):
                file_paths.append(os.path.join(self.directory, file_name))  # Tam dosya yolunu ekle
        return file_paths

# Verilen nokta stringini temizler ve bir listeye dönüştürür; geçersiz değerlerde hata mesajı verir.
def clean_and_split_point(point_str):
    """
    Verilen string içindeki noktayı temizler ve bir listeye çevirir.
    Örnek: '[ 891  598]' -> [891, 598]
    """
    if isinstance(point_str, str):  # Sadece string türündeyse işle
        point_str = point_str.replace('[', '').replace(']', '').strip()
        # Ek kontrol ekleyelim
        if not point_str:
            print("Boş veya geçersiz bir değer geldi.")
            return None
        point_list = point_str.split()
        # Noktaları kontrol et
        try:
            point_list = [int(x) for x in point_list]
        except ValueError as e:
            print(f"Geçersiz bir değer ile karşılaşıldı: {point_str} -> {e}")
            return None
        return point_list
    return None  # NaN veya geçersiz bir değer geldiğinde None döndür

#Dosya formatına göre veri setini yükler ve "Prev" ve "Curr" noktalarını iki ayrı listeye ayırır.
def load_data(file_path):
    """Veri setini dosya formatına göre yükler (csv veya xlsx) ve prev-curr noktalarını ayrıştırır."""
    file_ext = os.path.splitext(file_path)[1]

    if file_ext == '.csv':
        data = pd.read_csv(file_path)
    elif file_ext == '.xlsx':
        data = pd.read_excel(file_path)
    else:
        raise ValueError("Dosya formatı desteklenmiyor. Lütfen CSV veya XLSX kullanın.")

    # 'Prev' ve 'Curr' sütunlarını iki ayrı listeye ayıralım
    prev_points = []
    curr_points = []

    for _, row in data.iterrows():
        # Ek kontrol ekleyelim

        if len(row) < 2:  # Yeterli sütun yoksa
            print("Yetersiz sütun sayısı.")
            continue

        prev_point = clean_and_split_point(row.iloc[0])  # Prev sütunu
        curr_point = clean_and_split_point(row.iloc[1])  # Curr sütunu

        # Eğer prev_point veya curr_point None değilse listeye ekle
        if prev_point is not None and curr_point is not None:
            prev_points.append(prev_point)
            curr_points.append(curr_point)
        else:
            print(f"Geçersiz veri tespit edildi: Prev: {prev_point}, Curr: {curr_point}")

    return prev_points, curr_points
