from scipy.spatial.distance import euclidean
import numpy as np
import os
class GlassCutAnalysis:
    def __init__(self, prev_points, curr_points,file_name):
        """Veri noktalarını alır ve analizi yapar."""
        self.prev_points = prev_points
        self.curr_points = curr_points
        self.file_name = file_name

    def show_data(self, num_rows=5):
        """Veri setinin ilk birkaç satırını gösterir"""
        print(f"İlk {num_rows} prev noktaları:")
        print(self.prev_points[:num_rows])
        print(f"\nİlk {num_rows} curr noktaları:")
        print(self.curr_points[:num_rows])

    def calculate_euclidean_distances(self):
        """Prev ve Curr noktaları arasındaki Öklid mesafelerini hesaplar"""
        if self.prev_points is None or self.curr_points is None:
            raise ValueError("Veri yüklenmedi. Lütfen önce veri setini yükleyin.")

        distances = []
        for p, c in zip(self.prev_points, self.curr_points):
            if len(p) == 2 and len(c) == 2:  # Noktaların doğru şekilde olduğunu kontrol et
                dist = euclidean(p, c)
                distances.append(dist)
            else:
                raise ValueError(f"Noktalar yanlış formatta: Prev = {p}, Curr = {c}")

        return distances

    def calculate_statistics(self):
        """Öklid mesafelerinin ortalamasını ve standart sapmasını hesaplar"""
        distances = self.calculate_euclidean_distances()
        if not distances:
            return None, None  # Mesafe hesaplanmadıysa None döndür
        mean_distance = np.mean(distances)  # Ortalama mesafe
        std_distance = np.std(distances)  # Standart sapma

        return mean_distance, std_distance
    def compare_series(self, threshold=5.0):
        """Prev ve Curr noktalarının farklı bir seriye ait olup olmadığını kontrol eder"""
        distances = self.calculate_euclidean_distances()
        is_same_series = all(dist <= threshold for dist in distances)
        return is_same_series

    def label_same_series(self):
        """Dosya ismine göre same_series değerini belirler"""
        if self.file_name is None:
            raise ValueError("Dosya adı verilmemiş.")

        if "aynı" in self.file_name or "same" in self.file_name:
            return 1  # Aynı seri
        elif "farklı" in self.file_name or "different" in self.file_name:
            return 0  # Farklı seri
        else:
            raise ValueError(f"{self.file_name} dosyası için beklenmeyen bir isim formatı.")

    def calculate_angles(self):
        """Prev ve Curr noktaları arasındaki açıları hesaplar."""
        angles = []
        for i in range(len(self.prev_points) - 1):
            p1 = self.prev_points[i]
            p2 = self.prev_points[i + 1]
            c1 = self.curr_points[i]
            c2 = self.curr_points[i + 1]

            # Prev açısını hesapla
            prev_angle = np.arctan2(p2[1] - p1[1], p2[0] - p1[0]) * (180 / np.pi)
            # Curr açısını hesapla
            curr_angle = np.arctan2(c2[1] - c1[1], c2[0] - c1[0]) * (180 / np.pi)

            angles.append((prev_angle, curr_angle))

        return angles

    def analyze_angle_similarity(self):
        """Açı benzerliğini analiz eder ve benzerlik skorunu döndürür."""
        angles = self.calculate_angles()

        prev_angles = [angle[0] for angle in angles]
        curr_angles = [angle[1] for angle in angles]

        # Açı ortalaması ve standart sapmayı hesapla
        mean_prev = np.mean(prev_angles)
        mean_curr = np.mean(curr_angles)
        std_prev = np.std(prev_angles)
        std_curr = np.std(curr_angles)

        # MSE ve benzerlik skorunu hesapla
        mse = np.mean((np.array(prev_angles) - np.array(curr_angles)) ** 2)
        similarity_score = 100 / (1 + mse)  # Benzerlik skoru (0 ile 1 arasında)

        print(f"Prev Açı Ortalaması: {mean_prev:.2f}, Standart Sapma: {std_prev:.2f}")
        print(f"Curr Açı Ortalaması: {mean_curr:.2f}, Standart Sapma: {std_curr:.2f}")
        print(f"MSE: {mse:.2f}, Benzerlik Skoru: {similarity_score:.2f}")

        return {
            "mean_prev": mean_prev,
            "std_prev": std_prev,
            "mean_curr": mean_curr,
            "std_curr": std_curr,
            "mse": mse,
            "similarity_score": similarity_score
        }
