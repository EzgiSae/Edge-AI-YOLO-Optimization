# 🚀 YOLOv11 Optimization for Autonomous Drones: Ensuring Safe Flight

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![YOLOv11](https://img.shields.io/badge/YOLO-v11-green)
![ONNX](https://img.shields.io/badge/Export-ONNX-purple)
![Device](https://img.shields.io/badge/Device-Intel_i5-lightgrey)
![Status](https://img.shields.io/badge/Status-Optimized-brightgreen)

## 📖 Project Overview & The Problem
In autonomous drone navigation, **Latency = Risk**. 
Our baseline tests showed the standard YOLO model running at **10 FPS** (approx. 94ms latency). For a drone moving at 20km/h, this means flying **~1 meter "blind"** before processing the next frame. This delay is unacceptable for collision avoidance.

This project focuses on optimizing the **YOLOv11** object detection model to achieve **real-time inference speeds (30 FPS)** on edge devices without compromising safety-critical accuracy.

> **Goal:** Eliminate the "blind flight" risk by reducing inference latency under 35ms.

---

## 📊 Key Results (Benchmark)

We compared the original PyTorch model against optimized formats (ONNX, NCNN) on a standard CPU (Intel i5) environment.

| Model Format | Strategy | Latency (ms) | FPS | Speedup |
| :--- | :--- | :--- | :--- | :--- |
| **PyTorch (Baseline)** | .pt (640p) | 94.31 ms | ~10 FPS | 1x |
| **NCNN (Mobile)** | .ncnn (320p) | 46.55 ms | ~21.5 FPS | 2.2x |
| **ONNX (Optimized)** | .onnx (320p/INT8) | **33.42 ms** | **29.92 FPS** | **3.0x 🚀** |

> **Conclusion:** The **ONNX Nano (320p/INT8)** model was selected for deployment. It meets the real-time requirement (>24 FPS) while maintaining **98% Precision**, ensuring the drone doesn't stop for "ghost" obstacles.

---

## 🛠️ Engineering Approach: The Strategy

To achieve these results, we employed specific engineering strategies:

### 1. The "Word vs. PDF" Optimization Strategy
Why convert to ONNX?
* **PyTorch (.pt) is like a Word file:** It is editable, heavy, and carries full training metadata. It burdens the CPU unnecessarily during inference.
* **ONNX (Runtime) is like a PDF file:** It is read-only, streamlined, and highly optimized for reading/execution. It strips away training layers, allowing the CPU to focus solely on inference.

### 2. "Sprint vs. Marathon" Testing Protocol
We didn't just test for speed; we tested for reliability using two distinct approaches:

* **Sprint (Latency Test) -> `final_benchmark.py`:** * *Question:* "How fast can you react to a single image?"
    * *Purpose:* Determines the drone's **Reflex Speed** (Latency).
* **Marathon (Stability Test) -> `toplu_test.py`:**
    * *Question:* "Do you slow down after processing 2000 images? Do you get hot?"
    * *Purpose:* Tests **Thermal Throttling** and **Memory Stability** over long flights.

---

## 📂 Repository Structure & File Guide

Below is the explanation of the scripts and why they exist in this project:

### 🔹 1. Benchmarking & Testing
* **`benchmark_pytorch.py` (The Baseline):** * Runs the raw `.pt` model without optimization. 
    * *Result:* 94.31 ms (~10 FPS). This confirmed the "blind flight" risk and set our target for optimization.
* **`final_benchmark.py` (The Sprint):** * A comprehensive benchmark comparing 640p vs 320p models side-by-side. 
    * Measures the pure **Latency** (reaction time) for a single frame.
* **`toplu_test.py` (The Marathon):** * Processes a real-world dataset of thousands of images. 
    * Filters only for traffic classes (Car, Truck, Bus) and reports **Average FPS** and **Detection Accuracy** under load.
* **`benchmark_onnx.py`:** * Initially showed slower results (105ms), leading to a change in strategy: instead of a single run, we implemented a loop (avg of 50 runs) to warm up the ONNX Runtime engine.

### 🔹 2. Optimization & Conversion
* **`export_onnx.py`:** * Converts the heavy PyTorch model to the lightweight ONNX format.
* **`quantize_model.py`:** * Reduces model precision from 32-bit (Float) to 8-bit (Integer). 
    * *Impact:* Reduces file size from **10 MB -> 2.5 MB** and significantly improves CPU cache usage.
* **`export_ncnn.py`:** * Converts models for ARM-based devices (Raspberry Pi/Android). 
    * *Note:* While slower on PC (x86), NCNN is expected to outperform ONNX on actual drone hardware (ARM).
* **`export_320.py`:** * Re-exports models specifically with a 320x320 input resolution for maximum speed.

### 🔹 3. Quality Assurance
* **`kalite_analizi.py` (Confidence Analysis):** * Compares the "Confidence Score" of the optimized model against the Baseline. 
    * *Finding:* In some cases, optimization yielded negative loss (better confidence), proving that quantization didn't break the model.
* **`accuracy_analysis.py` (Teacher-Student Validation):** * Uses a "Teacher" (YOLOv11 Medium) to verify the "Student" (Nano INT8).
    * Generates **Visual Error Analysis** images (Green/Red boxes) to debug False Negatives.

---

## 💻 How to Run

1. **Install Dependencies:**
   ```bash
   pip install ultralytics opencv-python onnx onnxruntime-gpu
Run The "Marathon" Benchmark (Stability):
python scripts/toplu_test.py

Visualize the Errors (Quality Check):
python scripts/accuracy_analysis.py --source data/test_video.mp4

📄 Full Technical Report
For a deep dive into the methodology, confusion matrices, and detailed error analysis, please refer to the full report:

👉 [ YOLOv11 Optimizasyon Raporunu İndir (PDF)](docs/YOLOv11_Optimizasyon_Raporu.pdf)

👤 Author
Ezgi Sarıca Computer Engineering Student

[LinkedIn Profile](https://www.linkedin.com/in/ezgisar%C4%B1ca/)
