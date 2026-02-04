from ultralytics import YOLO

# Orijinal model (Nano/Medium)
model = YOLO("yolo11n.pt")

print("Dönüştürme işlemi başlıyor (PyTorch -> ONNX)...")

# Modeli ONNX formatına çevir
model.export(format="onnx", opset=12)

print("\nDönüştürme tamamlandı! 'yolo11n.onnx' dosyası oluşturuldu.")