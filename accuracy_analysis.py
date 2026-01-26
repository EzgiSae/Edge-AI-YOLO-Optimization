import torch
import numpy as np
from ultralytics import YOLO
import os
import glob
import cv2
import pandas as pd

RESIM_KLASORU = "C:/Users/ezgi.sarica/Desktop/obj_train_data"
CIKIS_KLASORU = "RENKLI_ANALIZ"
RAPOR_DOSYASI = "final_dogruluk_raporu.csv"

TEACHER_MODEL = "yolo11m.pt"        
STUDENT_MODEL = "yolo11n_int8.onnx" # yolo11n_int8.onnx seçilebilir.

IOU_THRESHOLD = 0.5    # %50 örtüşme barajı
MAX_SAVE_COUNT = 50    

if not os.path.exists(CIKIS_KLASORU):
    os.makedirs(CIKIS_KLASORU)

def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0

print(" FİNAL ANALİZ BAŞLIYOR (Görsel + İstatistik)...")
print("Modeller yükleniyor...")
teacher = YOLO(TEACHER_MODEL)
student = YOLO(STUDENT_MODEL, task="detect")

files = glob.glob(os.path.join(RESIM_KLASORU, "*.jpg")) + \
        glob.glob(os.path.join(RESIM_KLASORU, "*.png"))

files = files[:500] # Test için sadece ilk 500 resme bak

stats = {
    "TP": 0, # True Positive 
    "FP": 0, # False Positive 
    "FN": 0, # False Negative 
    "Total_Ref": 0 # Öğretmenin gördüğü toplam araç
}

saved_img_count = 0
print(f" Toplam {len(files)} resim analiz edilecek...")

for idx, img_path in enumerate(files):
    # TAHMİNLER (Sadece Araçlar: 2,3,5,7)
    # Teacher
    res_t = teacher(img_path, conf=0.5, classes=[2, 3, 5, 7], verbose=False)[0]
    boxes_t = res_t.boxes.xyxy.cpu().numpy()
    
    # Student
    res_s = student(img_path, conf=0.5, classes=[2, 3, 5, 7], verbose=False)[0]
    boxes_s = res_s.boxes.xyxy.cpu().numpy()
    
    stats["Total_Ref"] += len(boxes_t)

    # EŞLEŞTİRME VE PUANLAMA
    matched_teacher_indices = set()
    current_img_results = [] 
    
    for box_s in boxes_s:
        best_iou = 0
        best_t_idx = -1
        
        for i, box_t in enumerate(boxes_t):
            iou = calculate_iou(box_s, box_t)
            if iou > best_iou:
                best_iou = iou
                best_t_idx = i
        
        if best_iou >= IOU_THRESHOLD:
            stats["TP"] += 1
            matched_teacher_indices.add(best_t_idx)
            current_img_results.append((box_s, 'TP')) # Yeşil çizilecek
        else:
            stats["FP"] += 1
            current_img_results.append((box_s, 'FP')) # Turuncu çizilecek

    missed_indices = []
    for i in range(len(boxes_t)):
        if i not in matched_teacher_indices:
            stats["FN"] += 1
            missed_indices.append(i) # Kırmızı çizilecek

    # GÖRSELLEŞTİRME (Hata Varsa ve Limit Dolmadıysa)
    hata_var = (len(missed_indices) > 0) or (any(r[1] == 'FP' for r in current_img_results))
    
    if hata_var and saved_img_count < MAX_SAVE_COUNT:
        img = cv2.imread(img_path)
        
        for box, type_ in current_img_results:
            x1, y1, x2, y2 = map(int, box)
            if type_ == 'TP':
                color = (0, 255, 0) 
                label = "DOGRU"
            else:
                color = (0, 165, 255) 
                label = "HATALI"
            
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        for i in missed_indices:
            box = boxes_t[i]
            x1, y1, x2, y2 = map(int, box)
            color = (0, 0, 255) 
            
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, "KACAN", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Kaydet
        file_name = os.path.basename(img_path)
        cv2.imwrite(os.path.join(CIKIS_KLASORU, f"analiz_{file_name}"), img)
        saved_img_count += 1

    if idx % 50 == 0:
        print(f"   ... {idx} resim işlendi.")

# Sonuç Hesaplama
tp = stats["TP"]
fp = stats["FP"]
fn = stats["FN"]

precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print("\n" + "="*50)
print(" FİNAL PERFORMANS RAPORU")
print("="*50)
print(f"Toplam Referans Araç: {stats['Total_Ref']}")
print(f"✅ Doğru Tespit (TP) : {tp}")
print(f"🟥 Gözden Kaçan (FN) : {fn} (Körlük)")
print(f"🟧 Yanlış Alarm (FP) : {fp} (Halüsinasyon)")
print("*" * 30)
print(f" Precision (Doğruluk) : %{precision*100:.2f}")
print(f" Recall (Yakalama)    : %{recall*100:.2f}")
print(f" F1 Score             : %{f1*100:.2f}")
print("="*50)

# CSV Kayıt
df_new = pd.DataFrame([{
    "Tarih": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"), 
    "Model": STUDENT_MODEL,
    "Referans_Model": TEACHER_MODEL,
    "TP": tp,
    "FN": fn,
    "FP": fp,
    "Precision": round(precision, 4),
    "Recall": round(recall, 4),
    "F1_Score": round(f1, 4)
}])

dosya_var_mi = os.path.exists(RAPOR_DOSYASI)

df_new.to_csv(RAPOR_DOSYASI, mode='a', header=not dosya_var_mi, index=False)

print(f"\n Rapor '{RAPOR_DOSYASI}' dosyasına EKLENDİ.")
print(f" Görsel analizler '{CIKIS_KLASORU}' klasörüne güncellendi.")