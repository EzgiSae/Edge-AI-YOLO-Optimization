import torch
import numpy as np
from ultralytics import YOLO
import os
import glob
import pandas as pd

# ================= AYARLAR =================
RESIM_KLASORU = "C:/Users/ezgi.sarica/Desktop/obj_train_data"  # Resimlerinin olduğu yer
TEACHER_MODEL = "yolo11m.pt"          # Referans (Doğru kabul ettiğimiz)
STUDENT_MODEL = "yolo11n_int8.onnx"   # Test ettiğimiz (Sıkıştırılmış Nano)
IOU_THRESHOLD = 0.5                   # %50 üstü örtüşme varsa "Buldu" sayacağız
# ===========================================

def calculate_iou(box1, box2):
    # Kutu formatı: [x1, y1, x2, y2]
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0

print("DOĞRULUK ANALİZİ BAŞLIYOR (Teacher-Student Yaklaşımı)...")

# Modelleri Yükle
print(f"1. Öğretmen Yükleniyor ({TEACHER_MODEL})...")
teacher = YOLO(TEACHER_MODEL)

print(f"2. Öğrenci Yükleniyor ({STUDENT_MODEL})...")
student = YOLO(STUDENT_MODEL, task="detect")

# Resimleri Bul
files = glob.glob(os.path.join(RESIM_KLASORU, "*.jpg")) + \
        glob.glob(os.path.join(RESIM_KLASORU, "*.png")) + \
        glob.glob(os.path.join(RESIM_KLASORU, "*.jpeg"))
files = files[:500] # ÖRNEK: Hız için ilk 500 resme bakalım (İstersen bu satırı silip hepsine bak)

stats = {
    "True Positive (Başarılı)": 0,
    "False Negative (Kaçan)": 0,
    "False Positive (Hayal)": 0,
    "Total Objects (Referans)": 0
}

print(f"{len(files)} resim analiz edilecek...")

for idx, img_path in enumerate(files):
    # 1. Öğretmen Tahmini (Referans)
    res_t = teacher(img_path, verbose=False)[0]
    boxes_t = res_t.boxes.xyxy.cpu().numpy() # [x1, y1, x2, y2]
    
    # 2. Öğrenci Tahmini (Test)
    res_s = student(img_path, verbose=False)[0]
    boxes_s = res_s.boxes.xyxy.cpu().numpy()
    
    stats["Total Objects (Referans)"] += len(boxes_t)
    
    # Eşleştirme (Hangi öğrenci kutusu hangi öğretmen kutusuna denk geliyor?)
    matched_teacher_indices = set()
    
    for box_s in boxes_s:
        best_iou = 0
        best_t_idx = -1
        
        # Öğrencinin bu kutusu, öğretmenin kutularından hangisine benziyor?
        for i, box_t in enumerate(boxes_t):
            iou = calculate_iou(box_s, box_t)
            if iou > best_iou:
                best_iou = iou
                best_t_idx = i
        
        if best_iou >= IOU_THRESHOLD:
            # Eşleşme var! (True Positive)
            stats["True Positive (Başarılı)"] += 1
            matched_teacher_indices.add(best_t_idx)
        else:
            # Öğrenci bir şey bulmuş ama Öğretmen orada bir şey görmüyor (Yanlış Alarm)
            stats["False Positive (Hayal)"] += 1
            
    # Öğretmenin gördüğü ama Öğrencinin eşleşemediği her şey False Negative'dir
    missed_count = len(boxes_t) - len(matched_teacher_indices)
    stats["False Negative (Kaçan)"] += missed_count

    if idx % 50 == 0:
        print(f"   ... {idx} resim tamamlandı.")

# --- SONUÇ RAPORU ---
tp = stats["True Positive (Başarılı)"]
fn = stats["False Negative (Kaçan)"]
fp = stats["False Positive (Hayal)"]

# Precision (Keskinlik): "Bulduklarımın ne kadarı gerçekten araba?"
precision = tp / (tp + fp) if (tp + fp) > 0 else 0

# Recall (Duyarlılık): "Gerçek arabaların ne kadarını bulabildim?"
recall = tp / (tp + fn) if (tp + fn) > 0 else 0

# F1 Score (Denge Puanı)
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print("\n==================================================")
print(f"SONUÇLAR: {STUDENT_MODEL} vs {TEACHER_MODEL}")
print("==================================================")
print(f"True Positive (Doğru Tespit): {tp}")
print(f"False Negative (Gözden Kaçan): {fn}  <-- MENTÖRÜN SORDUĞU")
print(f"False Positive (Yanlış Alarm): {fp}")
print("--------------------------------------------------")
print(f"Precision (Doğruluk): %{precision*100:.2f}")
print(f"Recall (Yakalama)   : %{recall*100:.2f}")
print(f"F1 Score            : %{f1*100:.2f}")
print("==================================================")

# CSV Kayıt
df = pd.DataFrame([stats])
df["Precision"] = precision
df["Recall"] = recall
df["F1_Score"] = f1
df.to_csv("accuracy_report_INT8(1).csv", index=False)
print("Rapor 'accuracy_report_INT8(1).csv' olarak kaydedildi.")