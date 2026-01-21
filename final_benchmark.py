from ultralytics import YOLO
import time
import numpy as np
import os
import pandas as pd

# Test Resmi
IMAGE_URL = "https://ultralytics.com/images/bus.jpg"

SCENARIOS = [
    # --- 640p (Yüksek Detay) ---
    {"ad": "PyTorch (640p)", "dosya": "yolo11n.pt",            "size": 640},
    {"ad": "ONNX (640p)",    "dosya": "yolo11n.onnx",          "size": 640},
    {"ad": "INT8 (640p)",    "dosya": "yolo11n_int8.onnx",     "size": 640},
    {"ad": "NCNN (640p)",    "dosya": "yolo11n_ncnn_model",    "size": 640},

    # --- 320p (Yüksek Hız) ---
    {"ad": "PyTorch (320p)", "dosya": "yolo11n.pt",            "size": 320}, 
    {"ad": "ONNX (320p)",    "dosya": "yolo11n_320.onnx",      "size": 320},
    {"ad": "INT8 (320p)",    "dosya": "yolo11n_320_int8.onnx", "size": 320},
    {"ad": "NCNN (320p)",    "dosya": "yolo11n_320_ncnn_model","size": 320},
]

def run_benchmark(scenario):
    model_path = scenario["dosya"]
    model_name = scenario["ad"]
    img_sz = scenario["size"]

    print(f"\n Test Ediliyor: {model_name}...")

    # Dosya Kontrolü
    if not os.path.exists(model_path):
        print(f" HATA: Dosya bulunamadı -> {model_path} (Atlanıyor)")
        return None

    try:
        model = YOLO(model_path, task="detect")
        
        # Isınma (Warm-up) 
        print(" Motor ısınıyor...")
        for _ in range(3):
            model(IMAGE_URL, imgsz=img_sz, verbose=False)

        # Sprint Testi 
        times = []
        print(f" Ölçüm başladı (30 Tur)...")
        
        for i in range(30):
            t_start = time.time()
            model(IMAGE_URL, imgsz=img_sz, verbose=False)
            t_end = time.time()
            times.append((t_end - t_start) * 1000) 
            
        avg_time = np.mean(times)
        fps = 1000 / avg_time
        
        print(f" Sonuç: {avg_time:.2f} ms | {fps:.2f} FPS")
        
        return {
            "Model": model_name,
            "Çözünürlük": f"{img_sz}x{img_sz}",
            "Dosya": model_path,
            "Süre (ms)": round(avg_time, 2),
            "FPS": round(fps, 2)
        }

    except Exception as e:
        print(f" BEKLENMEYEN HATA: {e}")
        return None

print("\n" + "="*60)
print(" GRAND PRIX: YOLOv11 NANO (Tüm Formatlar)")
print("="*60)

results = []

for scen in SCENARIOS:
    data = run_benchmark(scen)
    if data:
        results.append(data)

# --- Raporlama ve Kayıt ---
if results:
    df = pd.DataFrame(results)
    
    df = df.sort_values(by="FPS", ascending=False)
    
    print("\n" + "="*60)
    print(" FİNAL SIRALAMASI (Hıza Göre)")
    print("="*60)
    print(df[["Model", "Çözünürlük", "Süre (ms)", "FPS"]].to_string(index=False))
    
    # Dosyaya kaydet
    df.to_csv("benchmark_tum_modeller.csv", index=False)
    print("\n Detaylı rapor 'benchmark_tum_modeller.csv' olarak kaydedildi.")
else:
    print("\n Hiçbir test başarıyla tamamlanamadı.")