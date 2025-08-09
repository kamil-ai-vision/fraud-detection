# 💳 Fraud Detection

A machine learning project that classifies fraudulent transactions from a simulated dataset using **LightGBM** and the **SMOTE** oversampling technique.

---

## 🌟 Features
- ✅ **High Recall:** Achieves a **Recall of 82.08%**, making it highly effective at detecting fraud.  
- 🧠 **Robust Model:** Uses **LightGBM** (state-of-the-art gradient boosting) with **SMOTE** to handle class imbalance.  
- 🛠️ **Complete Workflow:** Includes scripts for training (`train.py`), evaluation (`evaluate.py`), and prediction (`predict.py`).  
- 📈 **Detailed Evaluation:** Performance measured with **Precision**, **Recall**, and **F1-Score**—ideal for imbalanced datasets.  

---

## 🛠️ Setup and Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/kamil-ai-vision/fraud-detection.git
cd fraud-detection
```

### 2️⃣ Download Dataset

Download the dataset folder from the following link:  
[📂 Download Dataset](https://drive.google.com/drive/folders/1-NGRlR6FybosBYVAIogEjHu1uOJSmNm0?usp=sharing)

Inside the downloaded folder, you will find multiple `.pkl` files.  
Move **all** of these `.pkl` files into the `data/` directory of the project.

### 3️⃣ Set Up Virtual Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.\.venv\Scripts\activate

# Activate (macOS/Linux)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 How to Use

### Train the Model
To train the model and save the final pipeline:

```bash
python src/train.py
```

### Evaluate the Model
To evaluate the saved model's performance on the test set:

```
python src/evaluate.py
```

### Predict a New Transaction
1. Open predict.py and modify the sample_transaction dictionary with your desired features. 
2. Run the script:

```bash
python src/predict.py
```

---

## 📊 Performance Metrics

The model was optimized for recall and achieved the following results on the test set:

- **Precision:** 0.5446  
- **Recall:** 0.8208  
- **F1-Score:** 0.6548  

**Confusion Matrix:**
```
[[345880 2015]
[ 526 2410]]
```

---

## 👤 Author
Kamil - [https://github.com/kamil-ai-vision](https://github.com/kamil-ai-vision)
