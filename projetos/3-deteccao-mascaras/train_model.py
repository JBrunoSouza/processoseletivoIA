import shutil

from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Projeto 3 — Detecção de Máscaras Faciais (Fine-tuning do YOLO11n)
#
# Requisitos (veja README.md desta pasta para detalhes completos):
#   1. Carregar o modelo pré-treinado YOLO11n: YOLO("yolo11n.pt")
#      (única exceção à regra de "sem modelos pré-treinados" do processo seletivo)
#   2. Fazer fine-tuning em dataset/data.yaml, em CPU (device="cpu"),
#      com um número de épocas modesto (ex: 15-30)
#   3. Copiar os pesos resultantes (results.save_dir / "weights" / "best.pt")
#      para "model.pt", na raiz desta pasta
# ---------------------------------------------------------------------------

# insira seu código aqui

# Dica de estrutura (não é obrigatório seguir exatamente assim):
#
# model = YOLO("yolo11n.pt")
# results = model.train(
#     data="dataset/data.yaml",
#     epochs=...,
#     imgsz=...,
#     batch=...,
#     device="cpu",
# )
# shutil.copy(results.save_dir / "weights" / "best.pt", "model.pt")

import os

def main():
    print("Iniciando o fine-tuning do modelo...")
    
    # 1. Carrega o modelo pré-treinado (nosso 'cérebro' inicial)
    model = YOLO("yolo11n.pt")
    
    # 2. Inicia o treinamento (Fine-tuning)
    print("Treinando...")
    model.train(
        data="dataset/data.yaml",
        epochs=20,          
        imgsz=640,          
        batch=16,
        device="cpu",       
        name="train",
        exist_ok=True
    )
    
    # 3. Caminho onde o YOLO salva o melhor modelo treinado
    best_model_path = "runs/detect/train/weights/best.pt"
    
    # 4. Nome do arquivo final exigido pela correção do GitHub
    destination_path = "model.pt"
    
    # Copia da pasta oculta para a raiz do seu projeto 3
    if os.path.exists(best_model_path):
        shutil.copyfile(best_model_path, destination_path)
        print(f"✅ Sucesso! Modelo extraído para: {destination_path}")
    else:
        print("❌ Erro: O arquivo best.pt não foi gerado.")

if __name__ == "__main__":
    main()