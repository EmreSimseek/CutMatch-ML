from scipy.spatial.distance import euclidean


class GlassCutAnalysis:
    def __init__(self, prev_points, curr_points):
        """Veri noktalarını alır ve analizi yapar."""
        self.prev_points = prev_points
        self.curr_points = curr_points

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

    def compare_series(self, threshold=5.0):
        """Prev ve Curr noktalarının farklı bir seriye ait olup olmadığını kontrol eder"""
        distances = self.calculate_euclidean_distances()
        is_same_series = all(dist <= threshold for dist in distances)
        return is_same_series


    '''
    Düşük Threshold: Eğer threshold değeri düşük (örneğin, 1.0) olarak ayarlanırsa, nokta çiftleri arasındaki mesafe çok küçük olmalıdır. 
    Bu durumda, noktalardaki küçük değişiklikler bile "farklı seri" olarak değerlendirilebilir.

    Yüksek Threshold: Eğer threshold değeri yüksek (örneğin, 10.0 veya daha fazla) olarak ayarlanırsa, daha büyük mesafeler kabul edilebilir. 
    Bu durumda, nokta çiftleri arasında daha büyük farklılıklar olsa bile "aynı seri" olarak değerlendirilebilir.
    '''