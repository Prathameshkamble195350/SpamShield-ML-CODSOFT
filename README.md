# 📱 SMS Spam Detection using Machine Learning | CODSOFT

## 📌 Project Overview

SMS Spam Detection is a Machine Learning project that automatically classifies SMS messages as either Spam or Ham (Not Spam). The project uses Natural Language Processing (NLP) techniques and Machine Learning algorithms to identify unwanted messages.

This project was developed as part of the CODSOFT Machine Learning Internship.

---

## 🎯 Objective

The objective of this project is to build a machine learning model capable of detecting spam messages from SMS text data.

---

## 📂 Dataset

Dataset: SMS Spam Collection Dataset

Features:

| Column | Description |
|----------|-------------|
| Label | Spam or Ham |
| Message | SMS Text Message |

Target Variable:

- Spam
- Ham

---

## 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-Learn
- WordCloud
- Jupyter Notebook

---

## 🔄 Project Workflow

### 1️⃣ Data Collection

- Load SMS dataset
- Explore dataset structure
- Check class distribution

### 2️⃣ Data Cleaning

- Remove missing values
- Remove duplicates
- Clean text messages

### 3️⃣ Exploratory Data Analysis (EDA)

Performed various analyses:

- Spam vs Ham Distribution
- Message Length Analysis
- Word Frequency Analysis
- Word Cloud Visualization
- Most Common Spam Words

### 4️⃣ Text Preprocessing

- Lowercase Conversion
- Punctuation Removal
- Stopword Removal
- Tokenization

### 5️⃣ Feature Engineering

TF-IDF Vectorization converts text into numerical features.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(
    stop_words='english',
    max_features=5000
)
```

### 6️⃣ Train-Test Split

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

### 7️⃣ Model Training

Machine Learning Algorithm:

- Naive Bayes

```python
from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()
model.fit(X_train, y_train)
```

### 8️⃣ Model Evaluation

Evaluation Metrics:

- Accuracy Score
- Precision
- Recall
- F1 Score
- Confusion Matrix

```python
from sklearn.metrics import classification_report

predictions = model.predict(X_test)

print(classification_report(y_test, predictions))
```

### 9️⃣ Model Saving

```python
import pickle

pickle.dump(model, open('spam_model.pkl', 'wb'))
pickle.dump(tfidf, open('tfidf.pkl', 'wb'))
```

### 🔟 Spam Prediction

Predict whether a new SMS message is Spam or Ham.

Example:

```python
message = ["Congratulations! You have won a free iPhone."]

vector = tfidf.transform(message)

prediction = model.predict(vector)

print(prediction)
```

---

## 📊 Visualizations Included

✔ Spam vs Ham Distribution Chart

✔ Message Length Histogram

✔ Word Cloud for Spam Messages

✔ Word Cloud for Ham Messages

✔ Most Frequent Words Analysis

✔ Confusion Matrix

---

## 🚀 How to Run

### Clone Repository

```bash
git clone https://github.com/Prathameshkamble195350/SMS-Spam-Detection-CODSOFT.git
```

### Install Dependencies

```bash
pip install pandas numpy matplotlib scikit-learn wordcloud
```

### Launch Jupyter Notebook

```bash
jupyter notebook
```

Open:

```text
SMS spam detection.ipynb
```

---

## 📈 Results

The model successfully classifies SMS messages into Spam and Ham categories using Natural Language Processing and Machine Learning techniques.

Evaluation Metrics:

- Accuracy
- Precision
- Recall
- F1 Score

---

## 🔮 Future Improvements

- Random Forest Classifier
- Logistic Regression
- XGBoost
- Deep Learning Models
- BERT-Based Text Classification
- Flask Web Application
- Streamlit Deployment

---

## 💡 Skills Demonstrated

- Data Cleaning
- Data Analysis
- Data Visualization
- Natural Language Processing (NLP)
- TF-IDF Vectorization
- Machine Learning
- Naive Bayes Classification
- Model Evaluation
- Model Deployment

---

## 👨‍💻 Author

**Prathamesh Kamble**

B.Sc. Physics Graduate

Machine Learning Enthusiast

CODSOFT Machine Learning Intern

---

## ⭐ Internship Task

Task: SMS Spam Detection

Internship: CODSOFT Machine Learning Internship

Domain: Machine Learning & Natural Language Processing

Status: Completed ✅

---

### If you found this project useful, please give it a ⭐ on GitHub!
