import pandas as pd
class ResultsManager:
    def __init__(self):
        self.results = []

    def add_result(self, file_name, mean_distance, std_distance, shaper_result, angle_analysis, same_series_value):
        """Sonuçları listeye ekler"""
        self.results.append({
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

    def save_results_to_csv(self, output_path):
        """Sonuçları CSV dosyasına kaydeder"""
        pd.DataFrame(self.results).to_csv(output_path, index=False)
        print(f"Sonuçlar {output_path} dosyasına kaydedildi.")
