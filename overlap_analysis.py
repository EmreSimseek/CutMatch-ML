from scipy.spatial.distance import euclidean


class OverlapAnalysis:
    def __init__(self, points1, points2):
        """İki nokta kümesi arasında örtüşme analizini yapar."""
        self.points1 = points1
        self.points2 = points2

    def calculate_overlap(self, threshold=5.0):
        """Nokta kümeleri arasında örtüşme olup olmadığını kontrol eder."""
        overlaps = []
        for p1 in self.points1:
            for p2 in self.points2:
                if euclidean(p1, p2) <= threshold:
                    overlaps.append((p1, p2))
        return overlaps

