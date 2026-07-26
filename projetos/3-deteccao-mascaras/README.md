# Projeto 3 — Detecção de Máscaras Faciais (YOLO)

## 💻 O Desafio Técnico

Desenvolva um modelo de **detecção de objetos** capaz de identificar, em uma
imagem com rostos, se cada pessoa está **usando máscara corretamente**, **sem
máscara**, ou **usando a máscara de forma incorreta** — localizando cada rosto
com uma bounding box.

Diferente dos Projetos 1 e 2 (onde você constrói uma CNN do zero), aqui o
objetivo é **adaptar e otimizar um framework de detecção real para Edge AI** —
uma competência bastante prática no dia a dia de Visão Computacional Embarcada,
já que a imensa maioria das aplicações de detecção em produção parte de um
modelo pré-treinado, não de uma arquitetura construída do zero.

> ⚠️ **Exceção importante:** ao contrário dos Projetos 1 e 2, aqui o uso de
> **pesos pré-treinados é permitido e esperado** (fine-tuning). Isso é
> intencional — este projeto avalia uma competência diferente: adaptar,
> treinar e exportar um framework de detecção real para o seu dataset.

O foco não é apenas obter alta acurácia, mas **compreender o fluxo completo**:

**fine-tuning → validação → exportação → otimização para edge**

## 🎯 Conjunto de Dados

Este projeto já vem com um dataset **pronto para uso**, na pasta [`dataset/`](dataset/):
o **Face Mask Detection Dataset** ([Kaggle, andrewmvd](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection),
licença **CC0 1.0** — domínio público), já convertido do formato original (Pascal VOC)
para o formato esperado pelo Ultralytics YOLO.

- **853 imagens** de rostos, com bounding boxes anotadas
- **3 classes:** `with_mask`, `without_mask`, `mask_weared_incorrect`
- Já dividido em treino (~80%) e validação (~20%)
- ⚠️ O dataset é **desbalanceado** — a classe `mask_weared_incorrect` tem
  significativamente menos exemplos que as outras duas. Isso é uma
  característica real de datasets de detecção e não é um bug — comente esse
  ponto no seu relatório se perceber o modelo com dificuldade nessa classe.

Você **não precisa** baixar nada do Kaggle nem escrever código de conversão de
anotações — isso já está pronto em `dataset/`. Seu trabalho começa direto no
fine-tuning do modelo.

## ✅ Requisitos Obrigatórios

### Etapa 1 — Fine-tuning do Modelo (`train_model.py`)

Implemente, usando a biblioteca **Ultralytics** (YOLO):

- Carregamento do modelo pré-treinado **YOLO11n** (`YOLO("yolo11n.pt")`) —
  esta é a única exceção à regra de "sem modelos pré-treinados" do processo
  seletivo, válida especificamente para este projeto
- Fine-tuning no dataset fornecido (`dataset/data.yaml`), em **CPU**, com um
  número de épocas modesto (ex: 15-30 — YOLO converge relativamente rápido
  em fine-tuning, mesmo em CPU)
- Ao final do treino, copie os pesos resultantes (`runs/detect/train/weights/best.pt`)
  para a raiz desta pasta, com o nome **`model.pt`**

### Etapa 2 — Otimização do Modelo (`optimize_model.py`)

Implemente:

- Carregamento do `model.pt` treinado
- Exportação para **TensorFlow Lite** via `model.export(format="tflite")`
  (a Ultralytics gera automaticamente um arquivo `model.tflite` na mesma pasta)

> 💡 Na primeira execução, a Ultralytics pode instalar automaticamente
> dependências extras necessárias para a exportação (isso é esperado e pode
> levar alguns minutos).

### Etapa 3 — Inferência com o Modelo Otimizado (`run_inference.py`)

Implemente:

- Carregamento especificamente do **`model.tflite`** (o artefato de edge — não
  o `model.pt`) usando `YOLO("model.tflite", task="detect")`
- Execução de inferência em pelo menos **5 imagens** de `dataset/images/val/`,
  **uma de cada vez** — o `model.tflite` exportado aceita apenas 1 imagem por
  chamada (batch=1), que é aliás o cenário real de uso em edge
- Exibição no terminal, para cada imagem, do número de detecções encontradas

> 💡 O Ultralytics salva automaticamente as imagens anotadas com as caixas
> preditas em `runs/detect/...` (pasta já ignorada pelo `.gitignore` — não
> precisa, nem deve, ser commitada). Abra essas imagens localmente pra conferir
> visualmente as predições antes de escrever o relatório.
>
> 💡 Essa etapa existe porque uma métrica agregada (mAP) pode esconder
> problemas que só aparecem olhando exemplos individuais — especialmente dado
> o desbalanceamento de classes deste dataset.

## 📂 Estrutura da Pasta

⚠️ Não altere os nomes dos arquivos nem a estrutura de `dataset/`.

```
projetos/3-deteccao-mascaras/
├── train_model.py         # ✏️ Fine-tuning do modelo
├── optimize_model.py      # ✏️ Exportação e otimização
├── run_inference.py       # ✏️ Inferência de exemplo com o modelo otimizado
├── requirements.txt       # 📄 Dependências do projeto
├── model.pt               # 🤖 Gerado por você — deve ser commitado
├── model.tflite            # ⚡ Gerado por você — deve ser commitado
├── README.md               # 📝 Este arquivo (também usado como relatório)
└── dataset/                # 📦 Dataset já pronto (não modificar)
    ├── data.yaml
    ├── images/{train,val}/
    └── labels/{train,val}/
```

## ⚠️ Restrições e Considerações de Engenharia

- Modelo base: **YOLO11n** (variante *nano*, indicada para CPU/edge) — não use
  variantes maiores (s/m/l/x)
- Treinamento apenas em CPU
- Fine-tuning é permitido e esperado (única exceção às regras gerais do processo seletivo)
- **Não é esperada detecção perfeita**, especialmente na classe minoritária
  (`mask_weared_incorrect`) — o objetivo é demonstrar que o pipeline completo
  (fine-tuning → validação → exportação) funciona corretamente
- O tempo de treinamento e exportação deste projeto tende a ser **maior** que
  o dos Projetos 1 e 2 — reserve tempo extra para rodar localmente antes de enviar

## ⚖️ Critérios de Avaliação

- **Funcionalidade** — execução correta dos scripts e geração de `model.pt` e `model.tflite`
- **Qualidade do modelo** — mAP50 no conjunto de validação acima do mínimo esperado
- **Edge AI** — exportação correta para `.tflite`
- **Documentação** — preenchimento adequado do relatório abaixo

---

## 📝 Relatório do Candidato

👤 **Nome Completo:** José Bruno de Souza Alves

### 1️⃣ Resumo da Abordagem

* **Hiperparâmetros de Fine-Tuning:** 
Épocas: 20
Tamanho da imagem (imgsz): 640x640 pixels
Batch size: 16
* **Desbalanceamento de classes:** 
Observou-se um desbalanceamento acentuado no dataset (ex: 593 instâncias para with_mask contra apenas 19 instâncias para mask_weared_incorrect). O treinamento utilizou os pesos padrão do YOLO sem técnicas avançadas explícitas de balanceamento (como oversampling ou focal loss customizada), o que impactou diretamente a performance na classe minoritária.

### 2️⃣ Bibliotecas Utilizadas

* **Ultralytics:** Framework principal utilizado para carregar o modelo YOLO11n, realizar o fine-tuning, a validação e a exportação para o formato TFLite.
* **Módulos Nativos do Python (Standard Library):**
  * **`os`:** Manipulação e verificação de caminhos de arquivos e diretórios no sistema operacional.
  * **`shutil`:** Automação da cópia do arquivo de pesos gerados (`best.pt`) para a raiz do projeto (`model.pt`).

### 3️⃣ Técnica de Otimização do Modelo

A otimização consistiu na exportação do modelo PyTorch gerado (model.pt) para o formato leve TensorFlow Lite (.tflite), ideal para dispositivos de borda (Edge Devices). O processo foi executado chamando model.export(format="tflite", imgsz=640), convertendo a arquitetura para o formato clássico do TFLite para garantir compatibilidade e evitar problemas de execução com runtimes mais recentes.

### 4️⃣ Resultados Obtidos
--------------------------------------------------
* Métricas Gerais:

mAP50: 74.5%

mAP50-95: 52.0%

Box Precision (P): 0.763 | Recall (R): 0.751


* Métricas por Classe:

with_mask (149 imagens / 593 instâncias):
mAP50: 96.6% | mAP50-95: 67.1%

without_mask (57 imagens / 114 instâncias):
mAP50: 79.2% | mAP50-95: 51.8%

mask_weared_incorrect (15 imagens / 19 instâncias):
mAP50: 47.9% | mAP50-95: 37.2%


* Tamanho dos Arquivos:
model.pt: ~5.2 MB (aproximado para pesos YOLO11n padrão)

model.tflite: ~10.4 MB (tamanho padrão após conversão para formato TFLite)

### 5️⃣ Comentários Adicionais (Opcional)

* Dificuldades e Decisões: O principal desafio foi lidar com a escassez de dados da classe mask_weared_incorrect, que possui poucas amostras. Isso resultou em métricas inferiores para essa categoria específica. A escolha de 20 épocas em CPU equilibrou o tempo de processamento viável no ambiente de desenvolvimento com uma convergência razoável dos pesos.

* Limitações: O modelo apresenta maior taxa de confusão ou perda de sensibilidade ao detectar máscaras usadas incorretamente, devido ao viés de dados do dataset.

### 6️⃣ Exemplo de Inferência
```text 
============================================================
Projeto 3 — Inferência com model.tflite (Edge AI)
============================================================

Rodando inferência em 5 amostras usando model.tflite:

Imagem                               Detecções  Detalhes
----------------------------------------------------------------------
Loading /workspaces/processoseletivoIA/projetos/3-deteccao-mascaras/model.tflite for LiteRT inference...
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
[transformers] Disabling PyTorch because PyTorch >= 2.4 is required but found 2.2.1+cpu
[transformers] PyTorch was not found. Models won't be available and only tokenizers, configuration and file/data utilities can be used.
Results saved to /workspaces/processoseletivoIA/projetos/3-deteccao-mascaras/runs/detect/inferencia_exemplos/predicoes
maksssksksss105.jpg                          9  [9x with_mask]
Results saved to /workspaces/processoseletivoIA/projetos/3-deteccao-mascaras/runs/detect/inferencia_exemplos/predicoes
maksssksksss107.jpg                          1  [1x with_mask]
Results saved to /workspaces/processoseletivoIA/projetos/3-deteccao-mascaras/runs/detect/inferencia_exemplos/predicoes
maksssksksss11.jpg                          25  [23x with_mask, 2x mask_weared_incorrect]
Results saved to /workspaces/processoseletivoIA/projetos/3-deteccao-mascaras/runs/detect/inferencia_exemplos/predicoes
maksssksksss113.jpg                          4  [4x with_mask]
Results saved to /workspaces/processoseletivoIA/projetos/3-deteccao-mascaras/runs/detect/inferencia_exemplos/predicoes
maksssksksss12.jpg                          15  [12x with_mask, 3x without_mask]
----------------------------------------------------------------------
TOTAL                                       54

✅ Imagens anotadas salvas em: runs/detect/inferencia_exemplos/predicoes/
   (Abra essa pasta para verificar visualmente as bounding boxes preditas)
```
**Observações das imagens anotadas (runs/detect/inferencia_exemplos/predicoes/):**
As caixas delimitadoras (bounding boxes) para rostos com máscara (with_mask) e sem máscara (without_mask) mostraram-se precisas e bem localizadas ao redor das faces. No entanto, a classe minoritária (mask_weared_incorrect) apresentou detecções mais esparsas e menor nível de confiança, havendo ocasionalmente confusão onde uma máscara posicionada incorretamente era ora ignorada, ora classificada como uso correto, refletindo diretamente as limitações métricas observadas na validação.

## 📄 Créditos do Dataset

Face Mask Detection Dataset — [Kaggle: andrewmvd/face-mask-detection](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection), licença CC0 1.0 (domínio público).
