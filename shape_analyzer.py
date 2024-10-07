from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft
import os

class ShaperAnalysis:
    def __init__(self, prev_points, curr_points, file_name, output_path):
        """Nokta verilerini alır ve analizi yapar."""
        self.prev_points = prev_points
        self.curr_points = curr_points
        self.file_name = file_name
        self.output_path = output_path
    def analyze_data(self):

        # Nokta sayısını kontrol et
        if len(self.prev_points) < 3 or len(self.curr_points) < 3:
            print(f"{self.file_name} dosyasında geçerli nokta yok. Atlanıyor.")
            return None  # En az 3 nokta olması gerekir

        # Poligonları oluşturma
        prev_polygon = Polygon(self.prev_points)
        curr_polygon = Polygon(self.curr_points)

        # Poligonların geçerliliğini kontrol et ve düzelt
        if not prev_polygon.is_valid:
            prev_polygon = prev_polygon.buffer(0)  # Küçük topolojik hataları düzeltir
        if not curr_polygon.is_valid:
           curr_polygon = curr_polygon.buffer(0)

        # Poligonların geçerliliğini kontrol et
        if not prev_polygon.is_valid or not curr_polygon.is_valid:
            print(f"{self.file_name} dosyasındaki poligonlar geçersiz. Atlanıyor.")
            return None

        # Poligonları çizme
        self.plot_polygons()

        # Kesişim ve birleşim alanlarını hesaplama
        intersection_area = prev_polygon.intersection(curr_polygon).area
        union_area = prev_polygon.union(curr_polygon).area

        # IoU'yu hesaplama
        iou = intersection_area / union_area if union_area > 0 else 0

        # IoU'ya göre karar verme
        esik_degeri = 0.9
        series_label = "Aynı seri" if iou > esik_degeri else "Farklı seri"

        return {
            "file_name": self.file_name,
            "intersection_area": intersection_area,
            "union_area": union_area,
            "iou": iou,
            "series_label": series_label,
        }

    def plot_polygons(self):
        """Poligonları çizen fonksiyon."""
        x_prev, y_prev = zip(*self.prev_points)
        x_curr, y_curr = zip(*self.curr_points)

        plt.figure()
        plt.fill(x_prev, y_prev, alpha=0.5, label='Eski Kesim', color='blue')
        plt.fill(x_curr, y_curr, alpha=0.5, label='Yeni Kesim', color='orange')
        plt.plot(x_prev, y_prev, color='blue')
        plt.plot(x_curr, y_curr, color='orange')
        plt.legend()
        plt.title(f"{self.file_name}")
        plt.xlabel("X Koordinatları")
        plt.ylabel("Y Koordinatları")
        plt.grid(True)

        file_base_name = os.path.splitext(self.file_name)[0]
        output_dir = os.path.join(self.output_path,"plots", f"{file_base_name}_plot.png")  # Yeni dosya adı
        # Grafiği belirtilen dosya yoluna kaydet
        plt.savefig(output_dir, bbox_inches='tight')  # Kaydetme işlemi
        print(f"Poligon grafiği kaydedildi: {self.output_path}")
        plt.close()

    def apply_fourier_transform(self):
        """Prev ve Curr noktaları için Fourier dönüşümü uygular ve görselleştirir."""
        prev_x = [p[0] for p in self.prev_points]
        prev_y = [p[1] for p in self.prev_points]
        curr_x = [c[0] for c in self.curr_points]
        curr_y = [c[1] for c in self.curr_points]

        # Fourier dönüşümü
        fft_prev_x = fft(prev_x)
        fft_prev_y = fft(prev_y)
        fft_curr_x = fft(curr_x)
        fft_curr_y = fft(curr_y)

        # Fourier dönüşümünün büyüklük spektrumlarını hesapla
        magnitude_prev = np.sqrt(np.abs(fft_prev_x) ** 2 + np.abs(fft_prev_y) ** 2)
        magnitude_curr = np.sqrt(np.abs(fft_curr_x) ** 2 + np.abs(fft_curr_y) ** 2)

        # Frekans ekseni
        freq = np.fft.fftfreq(len(prev_x))
        summary = {
            "min_freq": freq.min(),
            "max_freq": freq.max(),
            "mean_magnitude_prev": np.mean(magnitude_prev),
            "max_magnitude_prev": np.max(magnitude_prev),
            "mean_magnitude_curr": np.mean(magnitude_curr),
            "max_magnitude_curr": np.max(magnitude_curr)
        }

        # Görselleştirme
        plt.figure(figsize=(12, 6))

        # Prev noktaları için Fourier büyüklük spektrumu
        plt.subplot(1, 2, 1)
        plt.plot(freq, magnitude_prev, color='blue')
        plt.title(f"Prev Noktaları - Fourier Büyüklük Spektrumu {self.file_name}")
        plt.xlabel("Frekans")
        plt.ylabel("Büyüklük")
        plt.grid(True)

        # Klasörü oluştur ve dosyayı kaydet
        prev_output_folder = os.path.join(self.output_path, "plots2/prev_fft")
        os.makedirs(prev_output_folder, exist_ok=True)
        file_base_name = os.path.splitext(self.file_name)[0]
        output_dir_prev = os.path.join(prev_output_folder, f"{file_base_name}_plot.png")
        plt.savefig(output_dir_prev, bbox_inches='tight')
        plt.close()

        # Curr noktaları için Fourier büyüklük spektrumu
        plt.subplot(1, 2, 2)
        plt.plot(freq, magnitude_curr, color='red')
        plt.title(f"Curr Noktaları - Fourier Büyüklük Spektrumu {self.file_name}")
        plt.xlabel("Frekans")
        plt.ylabel("Büyüklük")
        plt.grid(True)

        curr_output_folder = os.path.join(self.output_path, "plots2/curr_fft")
        os.makedirs(curr_output_folder, exist_ok=True)
        output_dir_curr = os.path.join(curr_output_folder, f"{file_base_name}_plot.png")
        plt.savefig(output_dir_curr, bbox_inches='tight')
        plt.close()

        # Fourier dönüşüm sonuçlarını döndür
        return summary



