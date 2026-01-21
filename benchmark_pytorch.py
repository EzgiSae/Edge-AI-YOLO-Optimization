from ultralytics import YOLO    
import cv2
import time

# İlk çalıştırmada bu dosyayı internetten indirecektir.
print("Model yükleniyor, lütfen bekleyin...")
model = YOLO("yolo11n.pt") 

# Modele neye bakacağını söyleriz.
image_url = "https://ultralytics.com/images/bus.jpg"

print("Test başlıyor...")

# Modelin belleğe tam yerleşmesi için boş bir işlem yaptırıyoruz.
model(image_url, verbose=False)

# Benchmark
start_time = time.time()

# Modeli çalıştır
results = model(image_url)

end_time = time.time()

# Sonuçlar
duration_ms = (end_time - start_time) * 1000
print(f"\n------------------------------------------------")
print(f"MODEL: YOLOv11 Nano (PyTorch - Orijinal)")
print(f"İşlem Süresi (Inference Time): {duration_ms:.2f} ms")
print(f"------------------------------------------------\n")

# İlk sonucu görsel olarak kaydeder
results[0].save("result_pytorch.jpg")
print("Sonuç resmi 'result_pytorch.jpg' olarak kaydedildi.")