from ultralytics import YOLO

# Nano modeli yükle
print("YOLOv11 Nano modeli NCNN formatına çevriliyor...")
model = YOLO("yolo11n.pt")

# NCNN'e çevir (Bu işlem otomatik olarak gerekli ncnn paketlerini indirebilir)
# half=True: Yarı hassasiyet (FP16) kullanır, daha hızlıdır.
model.export(format="ncnn", half=True)

print("Dönüşüm tamamlandı! 'yolo11n_ncnn_model' klasörü oluşturuldu.")