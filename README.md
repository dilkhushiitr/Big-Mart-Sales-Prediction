# 🛒 BigMart Sales Prediction

## 📌 Overview

This project focuses on predicting product sales for BigMart outlets using advanced machine learning techniques. The goal is to build a high-performance regression model leveraging feature engineering, target encoding, and ensemble learning.

---

## 🚀 Key Highlights

* 🔥 **Cross Target Encoding (Item × Outlet)** — strongest predictive signal
* ⚙️ **Optuna Hyperparameter Tuning** for LightGBM
* 🎯 **Multi-seed Ensembling** to reduce variance
* 🧠 **Stacking (Level-2 model with Ridge)**
* 📊 Extensive **feature engineering**
* 🤖 5 diverse base models for robust predictions

---

## 🧠 Models Used

* LightGBM (Optuna tuned)
* XGBoost
* CatBoost
* ExtraTrees Regressor
* MLP Regressor (Neural Network)

---

## 🏗️ Project Structure

```
Big-Mart-Sales-Prediction/
│
├── data/                  # Input datasets (not uploaded)
├── src/
│   └── train_v5.py       # Main training script
│
├── notebooks/            # EDA / experimentation (optional)
├── submission/           # Predictions (ignored in git)
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚙️ Features Engineering

* Item-level features (weight, visibility, category)
* Outlet-level features (type, size, age)
* Interaction features (MRP × Outlet, Visibility × Age)
* Target Encoding:

  * Single column TE
  * Cross TE (Item × Outlet, Category × Outlet, etc.)
* Aggregated statistics:

  * Outlet sales mean/std/median
  * Item sales statistics across outlets

---

## 📈 Model Pipeline

1. Data Cleaning & Missing Value Imputation
2. Feature Engineering
3. Target Encoding (CV-safe)
4. Model Training (5 models × 3 seeds)
5. OOF Predictions
6. Blending (Nelder-Mead optimization)
7. Stacking (Ridge)
8. Final Weighted Prediction

---

## 📊 Performance

* Evaluation Metric: **RMSE (log scale)**
* Cross-Validation Score: **~0.52–0.54** *(update with your best score)*

---

## ▶️ How to Run

### 1. Clone repo

```bash
git clone https://github.com/dilkhushiitr/Big-Mart-Sales-Prediction.git
cd Big-Mart-Sales-Prediction
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run training

```bash
python src/train_v5.py
```

---

## ⚠️ Notes

* Update file paths to:

```python
train = pd.read_csv('data/train.csv')
test  = pd.read_csv('data/test.csv')
```

* Large datasets are excluded via `.gitignore`
