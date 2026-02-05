# 🚢 Titanic Survival Prediction — Production-Style ML Pipeline

## 📌 Overview

This repository contains an end-to-end Machine Learning pipeline designed using a **production-style modular architecture** to predict passenger survival on the Titanic dataset.

The project focuses on **clean software engineering practices**, statistical preprocessing, and reproducible ML workflows, moving beyond monolithic notebooks into a scalable and maintainable code structure.

---

## 📊 Model Performance

The current baseline model (Logistic Regression) achieves:

| Metric | Score |
|---|---|
| Accuracy | **80%** |
| Precision (Survived) | 0.76 |
| Recall (Survived) | 0.74 |

### Key Insights
- Balanced performance between survivor and non-survivor classes  
- Stable generalization using an 80/20 train-test split  
- Solid baseline for future model experimentation  

---

## 🧰 Tech Stack

- **Language:** Python  
- **Data Processing:** Pandas, NumPy  
- **Machine Learning:** Scikit-learn  
- **Model:** Logistic Regression  
- **Environment:** Virtualenv  

---

## 🏗️ Project Architecture

```
├── data/               # Raw dataset (train.csv)
├── src/
│   ├── data_loader.py   # Data ingestion
│   ├── preprocessing.py # Feature engineering & imputation
│   ├── model.py         # Model definition
│   ├── train.py         # Training pipeline
│   └── evaluate.py      # Evaluation metrics
└── venv/
```

---

## 🛠️ Engineering Decisions

### 📌 Data Cleaning
- Dropped high-noise or non-informative features:
  - PassengerId
  - Name
  - Ticket
  - Cabin

---

### 📌 Missing Value Strategy
- **Age → Median Imputation**
  - Robust to outliers
  - Preserves distribution stability

- **Embarked → Mode Imputation**
  - Appropriate for categorical variables

---

### 📌 Feature Engineering
- **Sex → Binary Encoding**
  - male → 0  
  - female → 1  

- **Embarked → One-Hot Encoding**
  - Avoids artificial ordinal relationships
  - Uses `drop='first'` to prevent multicollinearity

---

### 📌 Modular Design Philosophy
Each ML stage is isolated:

| Module | Responsibility |
|---|---|
| data_loader | Data ingestion |
| preprocessing | Cleaning + feature engineering |
| model | Model selection |
| train | Training orchestration |
| evaluate | Metrics + reporting |

This allows easy swapping of models or preprocessing strategies.

---

## 🚀 Running the Project

### 1️⃣ Clone Repository
```
git clone https://github.com/ROCO03/titanic-survival-prediction.git
cd titanic-survival-prediction
```

---

### 2️⃣ Setup Environment
```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

### 3️⃣ Train Model
```
python src/train.py
```

---

## 📈 Example Output

```
Accuracy: 0.80

Confusion Matrix:
[[89 17]
 [19 54]]
```

## 🧠 Learning Goals

This project was built to practice:

- Writing ML code outside notebooks  
- Structuring real-world ML repositories  
- Applying statistical preprocessing correctly  
- Building reproducible ML experiments  

---

## 👨‍💻 Author

**Obed Rodriguez**  
Applied Mathematics & Computer Science Student — UNAM  
Aspiring Machine Learning Engineer  
