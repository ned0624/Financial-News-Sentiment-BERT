# Financial News Sentiment Analysis with BERT

Fine-tuning **BERT** (bert-base-uncased) to classify financial news sentences into **positive / negative / neutral** sentiment.

## 📌 Project Overview

Sentiment analysis of financial news is a core component in quantitative trading and financial AI systems. By understanding whether news carries positive or negative sentiment, models can be used to inform trading signals, risk assessment, and market analysis.

**Dataset:** [Financial PhraseBank](https://huggingface.co/datasets/financial_phrasebank) (Malo et al., 2014)  
**Model:** bert-base-uncased (HuggingFace Transformers)  
**Task:** 3-class text classification (positive / neutral / negative)

| Label | Class | Example |
|-------|-------|---------|
| 0 | Negative | *"Operating profit fell to EUR 22.4 mn from EUR 34.2 mn."* |
| 1 | Neutral | *"The board approved the acquisition of a Finnish company."* |
| 2 | Positive | *"The company reported record profits and raised its forecast."* |

## 📊 Results

| Metric | Score |
|--------|-------|
| **Test Accuracy** | — |
| **F1 Score (weighted)** | — |

*Run the notebook to populate results.*

## 🏗️ Model Architecture

```
bert-base-uncased (110M parameters)
  → [CLS] token embedding
  → Dropout(0.1)
  → Linear(768 → 3)
  → Softmax
Output: 3-class probability distribution
```

| Hyperparameter | Value |
|----------------|-------|
| Max Sequence Length | 128 |
| Batch Size | 16 |
| Epochs | 4 |
| Learning Rate | 2e-5 |
| Optimizer | AdamW |
| Scheduler | Linear warmup |
| Gradient Clipping | 1.0 |

## 📁 Project Structure

```
Financial-News-Sentiment-BERT/
├── sentiment_analysis.ipynb   # Main notebook
├── training_curve.png         # Loss & accuracy plots (generated)
├── confusion_matrix.png       # Confusion matrix (generated)
├── eda_plots.png              # EDA visualizations (generated)
├── best_model.pth             # Best model weights (generated)
├── requirements.txt
└── README.md
```

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/ned0624/Financial-News-Sentiment-BERT.git
cd Financial-News-Sentiment-BERT
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the notebook

Open `sentiment_analysis.ipynb` in Jupyter and run all cells.  
The dataset is automatically downloaded from HuggingFace — no manual download needed.

> 💡 GPU recommended but not required. CPU training takes approximately 20–30 minutes.

## 🛠️ Tech Stack

- **PyTorch** — training loop, GPU support
- **HuggingFace Transformers** — BERT model and tokenizer
- **HuggingFace Datasets** — Financial PhraseBank loading
- **scikit-learn** — evaluation metrics, train/test split
- **Matplotlib / Seaborn** — visualization

## 💡 Key Concepts Demonstrated

- BERT fine-tuning for downstream classification tasks
- Custom PyTorch `Dataset` class for NLP
- AdamW optimizer with linear warmup scheduler
- Gradient clipping for stable transformer training
- Evaluation with Accuracy, F1 Score, and Confusion Matrix
- Inference on custom text inputs with confidence scores

## 🔗 Related Projects

- [AOI Defect Classification with CNN](https://github.com/ned0624/Defect-Classifications-of-AOI)
- [Retinal Vessel Segmentation with U-Net](https://github.com/ned0624/Retinal-Vessel-Segmentation)

---

*Dataset: Malo, P., Sinha, A., Korhonen, P., Wallenius, J., & Takala, P. (2014). Good debt or bad debt: Detecting semantic orientations in economic texts. Journal of the American Society for Information Science and Technology.*
