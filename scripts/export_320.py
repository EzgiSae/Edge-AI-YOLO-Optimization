from ultralytics import YOLO
from onnxruntime.quantization import quantize_dynamic, QuantType
import os
import shutil

# Hem Medium hem Nano'yu listeye ekleyelim
modeller = ["yolo11n.pt", "yolo11m.pt"]

print(" 320p Dönüşüm Fabrikası Çalışıyor (ONNX + INT8 + NCNN)...")

for model_name in modeller:
    if not os.path.exists(model_name):
        print(f" {model_name} bulunamadı, bu model atlanıyor.")
        continue

    print(f"\n================ {model_name} İŞLENİYOR ================")
    
    # Modeli yükle
    model = YOLO(model_name)
    base_name = model_name.replace(".pt", "") # örn: yolo11n

    # ---------------------------------------------------------
    # ADIM 1: ONNX (320p)
    # ---------------------------------------------------------
    print(f" ONNX (320p) Dönüşümü...")
    model.export(format="onnx", imgsz=320, opset=12)
    
    # Dosya ismini düzeltme (yolo11n.onnx -> yolo11n_320.onnx)
    default_onnx = f"{base_name}.onnx"
    target_onnx = f"{base_name}_320.onnx"
    
    if os.path.exists(default_onnx):
        if os.path.exists(target_onnx):
            os.remove(target_onnx) # Eskisi varsa sil
        os.rename(default_onnx, target_onnx)
        print(f" Hazır: {target_onnx}")
    else:
        print(f" ONNX dosyası zaten isimlendirilmiş olabilir: {target_onnx}")

    # ---------------------------------------------------------
    # ADIM 2: INT8 Sıkıştırma (320p)
    # ---------------------------------------------------------
    print(f" INT8 Sıkıştırma...")
    target_int8 = f"{base_name}_320_int8.onnx"
    
    if os.path.exists(target_onnx):
        quantize_dynamic(
            model_input=target_onnx,
            model_output=target_int8,
            weight_type=QuantType.QUInt8
        )
        print(f"  Hazır: {target_int8}")
    else:
        print(" HATA: ONNX dosyası oluşmadığı için INT8 yapılamadı.")

    # ---------------------------------------------------------
    # ADIM 3: NCNN (320p) - YENİ EKLENEN KISIM
    # ---------------------------------------------------------
    print(f" NCNN (320p) Dönüşümü...")
    # half=True: Mobil uyumlu hızlandırma
    model.export(format="ncnn", imgsz=320, half=True)
    
    # Klasör ismini düzeltme (yolo11n_ncnn_model -> yolo11n_320_ncnn_model)
    default_ncnn_dir = f"{base_name}_ncnn_model"
    target_ncnn_dir = f"{base_name}_320_ncnn_model"
    
    if os.path.exists(default_ncnn_dir):
        if os.path.exists(target_ncnn_dir):
            shutil.rmtree(target_ncnn_dir) # Eskisi varsa temizle
        os.rename(default_ncnn_dir, target_ncnn_dir)
        print(f" Hazır: {target_ncnn_dir} (Klasör)")
    else:
        print(f" NCNN klasörü bulunamadı veya adı farklı: {default_ncnn_dir}")

print("\n TÜM İŞLEMLER TAMAMLANDI!")