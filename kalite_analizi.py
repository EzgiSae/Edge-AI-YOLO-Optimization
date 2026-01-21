from ultralytics import YOLO
import pandas as pd
import os

# Test Edilecek Resim (Otobüs resmi iyidir, bol nesne var)
IMAGE_URL = "https://ultralytics.com/images/bus.jpg"

# Senaryolar (Benchmark ile aynı dosyalar)
SCENARIOS = [
    # --- 640p LİGİ ---
    {"ad": "PyTorch (640p)", "dosya": "yolo11n.pt",            "size": 640, "tur": "ref"}, # Referans
    {"ad": "ONNX (640p)",    "dosya": "yolo11n.onnx",          "size": 640, "tur": "aday"},
    {"ad": "INT8 (640p)",    "dosya": "yolo11n_int8.onnx",     "size": 640, "tur": "aday"},
    {"ad": "NCNN (640p)",    "dosya": "yolo11n_ncnn_model",    "size": 640, "tur": "aday"},

    # --- 320p LİGİ ---
    {"ad": "PyTorch (320p)", "dosya": "yolo11n.pt",            "size": 320, "tur": "ref"}, # Referans
    {"ad": "ONNX (320p)",    "dosya": "yolo11n_320.onnx",      "size": 320, "tur": "aday"},
    {"ad": "INT8 (320p)",    "dosya": "yolo11n_320_int8.onnx", "size": 320, "tur": "aday"},
    {"ad": "NCNN (320p)",    "dosya": "yolo11n_320_ncnn_model","size": 320, "tur": "aday"},
]

def get_confidence_score(model_path, img_sz):
    if not os.path.exists(model_path):
        return None, 0

    try:
        # Modeli yükle
        model = YOLO(model_path, task="detect")
        
        # Tek bir resim üzerinde tahmin yap
        results = model(IMAGE_URL, imgsz=img_sz, verbose=False)
        result = results[0]
        
        # Güven skorlarını al
        confidences = result.boxes.conf.tolist()
        
        if not confidences:
            return 0, 0 # Hiçbir şey bulamadıysa güven 0'dır
            
        # Ortalama güveni hesapla
        avg_conf = sum(confidences) / len(confidences)
        box_count = len(confidences)
        
        return avg_conf, box_count

    except Exception as e:
        print(f"Hata: {e}")
        return None, 0

print("\n" + "="*60)
print("🧐 KALİTE KONTROL: Güven Kaybı Analizi")
print("="*60)

# Verileri toplayacağımız liste
data = []

# Önce Referans (PyTorch) skorlarını hesaplayıp saklayalım
ref_scores = {} # {640: 0.85, 320: 0.72} gibi tutacak

print("1. Referans Skorları Hesaplanıyor (PyTorch)...")
for scen in SCENARIOS:
    if scen["tur"] == "ref":
        score, count = get_confidence_score(scen["dosya"], scen["size"])
        if score is not None:
            ref_scores[scen["size"]] = score
            print(f"   ✅ Referans {scen['size']}p: %{score*100:.2f} (Nesne: {count})")
            
            # Referansı da listeye ekle
            data.append({
                "Model": scen["ad"],
                "Çözünürlük": scen["size"],
                "Ort. Güven (%)": round(score * 100, 2),
                "Nesne Sayısı": count,
                "Güven Kaybı": "REFERANS" # Kayıp yok, kendisi referans
            })

print("\n2. Aday Modeller Karşılaştırılıyor...")
for scen in SCENARIOS:
    if scen["tur"] == "aday":
        score, count = get_confidence_score(scen["dosya"], scen["size"])
        
        if score is not None:
            # İlgili çözünürlüğün referansını bul
            ref_score = ref_scores.get(scen["size"], 0)
            
            # Kayıp hesabı (Pozitif değer kayıp demektir)
            # Örn: Ref 90, Aday 85 -> Kayıp 5
            loss = (ref_score - score) * 100 
            
            data.append({
                "Model": scen["ad"],
                "Çözünürlük": scen["size"],
                "Ort. Güven (%)": round(score * 100, 2),
                "Nesne Sayısı": count,
                "Güven Kaybı": round(loss, 2)
            })
            print(f"   🔹 {scen['ad']:<20} -> Kayıp: {loss:.2f}")

# --- RAPORLAMA ---
if data:
    df = pd.DataFrame(data)
    # Tabloyu Çözünürlüğe göre sıralayalım ki 640lar ve 320ler bir arada dursun
    df = df.sort_values(by=["Çözünürlük", "Ort. Güven (%)"], ascending=[False, False])
    
    print("\n" + "="*60)
    print("🏆 KALİTE ANALİZ RAPORU")
    print("="*60)
    print(df.to_string(index=False))
    
    df.to_csv("kalite_analizi_raporu.csv", index=False)
    print("\n📄 Rapor 'kalite_analizi_raporu.csv' olarak kaydedildi.")