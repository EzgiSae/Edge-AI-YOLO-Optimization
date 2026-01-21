import os
import time
import glob
from ultralytics import YOLO
import pandas as pd
from collections import Counter

RESIM_KLASORU = "C:/Users/ezgi.sarica/Desktop/obj_train_data" 

# Test edilecek modeller (Tüm 320p)
TEST_LISTESI = [
    {"ad": "ONNX (Nano 320p)",    "dosya": "yolo11n_320.onnx"},      
    {"ad": "PyTorch (Nano 320p)", "dosya": "yolo11n.pt"},            
    {"ad": "NCNN (Nano 320p)",    "dosya": "yolo11n_320_ncnn_model"} 
]

def modeli_test_et(model_bilgisi, resim_listesi):
    model_adi = model_bilgisi["ad"]
    dosya_yolu = model_bilgisi["dosya"]

    print(f"\n TEST BAŞLIYOR: {model_adi}")
    
    # NCNN bir klasördür, diğerleri dosyadır. İkisi için de kontrol yapalım.
    if not os.path.exists(dosya_yolu):
        print(f" HATA: Dosya/Klasör bulunamadı -> {dosya_yolu}")
        return None
    
    try:
        model = YOLO(dosya_yolu, task="detect")
    except Exception as e:
        print(f" Model yükleme hatası: {e}")
        return None

    toplam_sure_ms = 0
    toplam_nesne_sayisi = 0
    islenen_resim = 0
    sinif_sayaci = Counter() 
    
    # Isınma
    if len(resim_listesi) > 0:
        model(resim_listesi[0], imgsz=320, conf=0.50, classes=[2, 3, 5, 7], verbose=False)

    print(f" {len(resim_listesi)} adet resim işleniyor...")
    
    for resim in resim_listesi:
        t1 = time.time()
        
        # FİLTRELEME: Sadece Araçlar (Car, Moto, Bus, Truck) ve %50 Güven
        results = model(resim, imgsz=320, conf=0.50, classes=[2, 3, 5, 7], verbose=False)
        
        t2 = time.time()
        toplam_sure_ms += (t2 - t1) * 1000
        
        # İstatistik toplama
        for r in results:
            toplam_nesne_sayisi += len(r.boxes)
            for cls_id in r.boxes.cls:
                if hasattr(model, 'names'):
                    isim = model.names[int(cls_id)]
                else:
                    isim = str(int(cls_id))
                sinif_sayaci[isim] += 1

        islenen_resim += 1
        
        if islenen_resim % 100 == 0:
            print(f"   ... {islenen_resim} resim tamamlandı.")

    if islenen_resim == 0: return None

    # Hesaplama
    ortalama_sure = toplam_sure_ms / islenen_resim
    fps = 1000 / ortalama_sure
    
    print(f" BİTTİ: {model_adi}")
    print(f"    Ortalama Hız: {ortalama_sure:.2f} ms")
    print(f"    FPS: {fps:.2f}")
    print(f"    Toplam Araç: {toplam_nesne_sayisi}")
    print(f"    Detay: {dict(sinif_sayaci)}")
    
    return {
        "Model": model_adi,
        "FPS": round(fps, 2),
        "Araç Sayısı": toplam_nesne_sayisi,
        "Detaylar": str(dict(sinif_sayaci))
    }

uzantilar = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
tum_resimler = []
for ext in uzantilar:
    tum_resimler.extend(glob.glob(os.path.join(RESIM_KLASORU, ext)))

if not tum_resimler:
    print(" HATA: Klasörde hiç resim bulunamadı!")
else:
    print(f" Toplam {len(tum_resimler)} resim bulundu. FİNAL TEST BAŞLIYOR...\n")
    
    sonuclar = []
    
    for test in TEST_LISTESI:
        veri = modeli_test_et(test, tum_resimler)
        if veri:
            sonuclar.append(veri)

    # Sonuçları Kaydet
    if sonuclar:
        df = pd.DataFrame(sonuclar)
        df = df.sort_values(by="FPS", ascending=False)
        
        dosya_adi = "sonuclar_FINAL_320_FILTRELI.csv"
        df.to_csv(dosya_adi, index=False)
        
        print("\n" + "="*50)
        print(" FİNAL SONUÇ TABLOSU")
        print("="*50)
        print(df.to_string(index=False))
        print(f"\n Rapor '{dosya_adi}' olarak kaydedildi.")