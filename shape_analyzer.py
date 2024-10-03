from shapely.geometry import Polygon
import matplotlib.pyplot as plt
from utils.file_loader import load_data
import os

# Dosya yolları
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

for file_path in file_paths:
    file_name = os.path.basename(file_path)
    print(f"\nİşleniyor: {file_name}")

    if not os.path.exists(file_path):
        print(f"{file_name} bulunamadı.")
        continue  # Eğer dosya yoksa bir sonraki döngüye geç

    # Veriyi yükle
    prev_noktalar, curr_noktalar = load_data(file_path)

    # Nokta sayısını kontrol et
    if len(prev_noktalar) < 3 or len(curr_noktalar) < 3:
        print(f"{file_name} dosyasında geçerli nokta yok. Atlanıyor.")
        continue  # En az 3 nokta olması gerekir

    # Poligonları oluşturma
    prev_polygon = Polygon(prev_noktalar)
    curr_polygon = Polygon(curr_noktalar)

    # Poligonların geçerliliğini kontrol et ve geçersizse düzelt
    if not prev_polygon.is_valid:
        print(f"{file_name} dosyasındaki eski kesim poligonu geçersiz. Düzeltme yapılıyor.")
        prev_polygon = prev_polygon.buffer(0)  # Küçük topolojik hataları düzeltir
    if not curr_polygon.is_valid:
        print(f"{file_name} dosyasındaki yeni kesim poligonu geçersiz. Düzeltme yapılıyor.")
        curr_polygon = curr_polygon.buffer(0)

    # Poligonların geçerliliğini kontrol et
    if not prev_polygon.is_valid or not curr_polygon.is_valid:
        print(f"{file_name} dosyasındaki poligonlar geçersiz. Atlanıyor.")
        continue

    # Poligonları çizme
    x_prev, y_prev = zip(*prev_noktalar)
    x_curr, y_curr = zip(*curr_noktalar)

    plt.figure()
    plt.fill(x_prev, y_prev, alpha=0.5, label='Eski Kesim', color='blue')
    plt.fill(x_curr, y_curr, alpha=0.5, label='Yeni Kesim', color='orange')
    plt.plot(x_prev, y_prev, color='blue')
    plt.plot(x_curr, y_curr, color='orange')
    plt.legend()
    plt.title(f"{file_name}")
    plt.xlabel("X Koordinatları")
    plt.ylabel("Y Koordinatları")
    plt.grid(True)
    plt.show()

    # Kesişim ve birleşim alanlarını hesaplama
    intersection_area = prev_polygon.intersection(curr_polygon).area
    union_area = prev_polygon.union(curr_polygon).area

    # IoU'yu hesaplama
    iou = intersection_area / union_area if union_area > 0 else 0

    # IoU'ya göre karar verme
    esik_degeri = 0.9
    if iou > esik_degeri:
        print("Aynı seri")
    else:
        print("Farklı seri")

    # Hesaplanan alanları yazdırma
    print(f"Kesişim Alanı: {intersection_area:.2f}")
    print(f"Birleşim Alanı: {union_area:.2f}")
    print(f"IoU: {iou:.2f}")
