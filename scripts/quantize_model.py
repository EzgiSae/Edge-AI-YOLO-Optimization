from onnxruntime.quantization import quantize_dynamic, QuantType

input_model_path = "yolo11m.onnx"
output_model_path = "yolo11m_int8.onnx"

print("INT8 Sıkıştırma işlemi başlıyor...")

quantize_dynamic(
    model_input=input_model_path,
    model_output=output_model_path,
    weight_type=QuantType.QUInt8  # Ağırlıkları 8-bit tam sayıya çevir
)

print(f"Tamamlandı! Yeni model: {output_model_path}")