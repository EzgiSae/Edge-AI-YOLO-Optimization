from ultralytics import YOLO
import time
import os

# Dosya kontrolü
if not os.path.exists("yolo11n.onnx"):
    print("HATA: 'yolo11n.onnx' dosyası bulunamadı! Önce export_onnx.py çalıştırılmalı.")
    exit()

print("ONNX Modeli yükleniyor (Bu işlem PyTorch'tan hızlıdır)...")

model = YOLO("yolo11n.onnx", task="detect") 

# Test resmi
image_url = "https://ultralytics.com/images/bus.jpg"

print("Test başlıyor...")

# Isınma Turu (Warm-up)
model(image_url, verbose=False)

# Hız Testi
start_time = time.time()

# Modeli çalıştır
results = model(image_url)

end_time = time.time()

# Sonuç
duration_ms = (end_time - start_time) * 1000

print(f"\n================================================")
print(f"MODEL: YOLOv11 Nano (ONNX - Optimize Edilmiş)")
print(f"İşlem Süresi: {duration_ms:.2f} ms")
print(f"================================================\n")

# Eski skorla (94.31 ms) kıyaslama
old_score = 94.31
diff = old_score - duration_ms

if diff > 0:
    print(f"SONUÇ: Harika! {diff:.2f} ms daha hızlısın.")
    print(f"Hız Artışı: %{(diff / old_score) * 100:.1f}")
else:
    print("SONUÇ: Beklenen hızlanma olmadı (Bazen ilk turda olabilir). Tekrar dene.")