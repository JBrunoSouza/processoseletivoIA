from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Projeto 3 — Otimização do Modelo (Exportação para Edge)
#
# Requisitos (veja README.md desta pasta para detalhes completos):
#   1. Carregar o modelo treinado em "model.pt"
#   2. Exportar para TensorFlow Lite via model.export(format="tflite")
#      (a Ultralytics gera automaticamente "model.tflite" na mesma pasta)
# ---------------------------------------------------------------------------

# insira seu código aqui

# Dica de estrutura (não é obrigatório seguir exatamente assim):
#
# model = YOLO("model.pt")
# model.export(format="tflite", imgsz=...)

def main():
    print("Carregando o modelo treinado (model.pt)...")
    model = YOLO("model.pt")
    
    print("Iniciando a conversão direta para TFLite clássico...")
    # Força o formato tflite explicitamente usando o exportador legado para evitar o bug do LiteRT
    model.export(format="tflite", imgsz=640)
    print("✅ Conversão concluída com sucesso!")

if __name__ == "__main__":
    main()