from shapely.geometry import Polygon
import matplotlib.pyplot as plt
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
            print(f"{self.file_name} dosyasındaki eski kesim poligonu geçersiz. Düzeltme yapılıyor.")
            prev_polygon = prev_polygon.buffer(0)  # Küçük topolojik hataları düzeltir
        if not curr_polygon.is_valid:
            print(f"{self.file_name} dosyasındaki yeni kesim poligonu geçersiz. Düzeltme yapılıyor.")
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

        # Hesaplanan alanları yazdırma
        print(f"Kesişim Alanı: {intersection_area:.2f}")
        print(f"Birleşim Alanı: {union_area:.2f}")
        print(f"IoU: {iou:.2f}")
        print(f"Seri Durumu: {series_label}")

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
        output_dir = os.path.join(self.output_path, f"{file_base_name}_plot.png")  # Yeni dosya adı
        # Grafiği belirtilen dosya yoluna kaydet
        plt.savefig(output_dir, bbox_inches='tight')  # Kaydetme işlemi
        print(f"Poligon grafiği kaydedildi: {self.output_path}")
        plt.close()


