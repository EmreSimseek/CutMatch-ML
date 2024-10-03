import pandas as pd
import os
import numpy as np


def clean_and_split_point(point_str):
    """
    Verilen string içindeki noktayı temizler ve bir listeye çevirir.
    Örnek: '[ 891  598]' -> [891, 598]
    """
    if isinstance(point_str, str):  # Sadece string türündeyse işle
        point_str = point_str.replace('[', '').replace(']', '').strip()
        point_list = [int(x) for x in point_str.split()]
        return point_list
    return None  # NaN veya geçersiz bir değer geldiğinde None döndür


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
        prev_point = clean_and_split_point(row.iloc[0])  # Prev sütunu
        curr_point = clean_and_split_point(row.iloc[1])  # Curr sütunu

        # Eğer prev_point veya curr_point None değilse listeye ekle
        if prev_point is not None and curr_point is not None:
            prev_points.append(prev_point)
            curr_points.append(curr_point)
        else:
            print(f"Geçersiz veri tespit edildi: Prev: {prev_point}, Curr: {curr_point}")

    return prev_points, curr_points



