# 🧠 Multiclass Text Classification with Scikit-Learn

This project demonstrates a complete pipeline for multiclass text classification using classical machine learning models. The goal is to classify text documents into multiple categories using pre-processing, feature extraction, and model evaluation techniques.

---

## 📁 Project Structure

- `df.csv` — Contains the dataset used for training and testing.
- `notebook` — Jupyter notebooks with code.

- `README.md` — You're here!

---

## 🛠️ Technologies Used

- Python 🐍
- Pandas & NumPy
- Scikit-learn
- NLTK (for stopwords and stemming)
- Matplotlib / Seaborn (optional for plots)

---

## 📊 Models Implemented

- Logistic Regression ✅
- Support Vector Machine (SVC) ✅
- Decision Tree ✅
- Random Forest ✅

All models were evaluated on:
- Accuracy
- Precision
- Recall
- F1 Score

---

## 📈 Results Snapshot

| Model              | Accuracy | Precision | Recall | F1 Score |
|-------------------|----------|-----------|--------|----------|
| LogisticRegression| 0.654    | 0.686     | 0.402  | 0.453    |
| SVM               | 0.628    | 0.621     | 0.388  | 0.434    |
| Decision Tree     | 0.493    | 0.332     | 0.312  | 0.320    |
| Random Forest     | 0.356    | 0.175     | 0.130  | 0.113    |

---

## 🧹 Text Preprocessing Steps

1. **Lowercasing**
2. **Removing punctuation and special characters**
3. **Tokenization**
4. **Stopword removal** (using NLTK)
5. **Stemming** (using Porter Stemmer)
6. **TF-IDF vectorization**

---

## 🔍 Hyperparameter Tuning

Performed using:
- `GridSearchCV`
- `RandomizedSearchCV`

Example best parameters for Logistic Regression:
```python
{'C': 10, 'max_iter': 100, 'solver': 'saga'}
