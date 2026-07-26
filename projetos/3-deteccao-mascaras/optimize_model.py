import os
from ultralytics import YOLO

# Só roda o export se o arquivo model.tflite não existir localmente
tflite_path = "model.tflite"

if not os.path.exists(tflite_path):
    print("Gerando model.tflite...")
    # Carrega o modelo apenas se for realmente precisar exportar
    model = YOLO("model.pt")
    # Garante que está em modo de avaliação
    model.eval()
    model.export(format="tflite", imgsz=640)
else:
    print("✅ model.tflite já existe! Pulando exportação.")