# 📩 SMS Spam Detection using Naive Bayes

A standalone Python-based project that detects spam SMS messages using Natural Language Processing (NLP) techniques and a Naive Bayes classifier. The project includes data cleaning, feature engineering, EDA, model building, and evaluation.

---

## 🧠 Model Used

- **Multinomial Naive Bayes**
  - Ideal for text classification problems using bag-of-words or frequency-based features.

---

## 📂 Dataset

- **Source**: [UCI SMS Spam Collection Dataset](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection)
- **Fields**:
  - `v1`: Label (ham/spam)
  - `v2`: Text message

---

## 🧼 Text Preprocessing

Steps applied to clean and prepare the text:
- Convert text to lowercase
- Tokenize sentences and words
- Remove stopwords and non-alphanumeric tokens
- Apply stemming using Porter Stemmer

---

## 📊 Exploratory Data Analysis

- Class distribution (Spam vs Ham)
- Feature distributions:
  - Number of characters
  - Number of words
  - Number of sentences
- Pairplot of features
- Correlation heatmap

---

## 📈 Visualizations

- Word clouds for spam and ham messages
- Top 30 most common words:
  - Spam messages
  - Ham messages
- Count plot and pie chart for class distribution

---

## 🔍 Feature Engineering

Created the following features:
- `num_characters`: Length of message
- `num_words`: Number of words
- `num_sentence`: Number of sentences

---

## ✅ Model Evaluation

- Confusion Matrix
- Accuracy Score
- Precision Score
- Classification Report

---

## 📦 Dependencies

Install the required packages:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn nltk wordcloud
```
## 🚀 How to Run
python spam_detector.py
Ensure that spam.csv (the dataset) is in the same directory.
 ---
 ## 📸 Example Visuals
 
-📊 Spam vs Ham Distribution

![Figure_5](https://github.com/user-attachments/assets/29391768-3622-48b7-9d80-cd074a9ebb20)

![Figure_4](https://github.com/user-attachments/assets/2b41cccd-443e-4bb8-a928-e55b7b94423a)

![Figure_3](https://github.com/user-attachments/assets/fbff9113-5248-465b-a08a-45ba922a7018)

-☁️ Word Clouds for Spam and Ham

![Figure_8](https://github.com/user-attachments/assets/7d530d9b-b780-496e-989c-5aa31b742b40)

![Figure_9](https://github.com/user-attachments/assets/bc0fff45-ea26-4feb-8480-791a56703d69)


-📈 Top 30 Frequent Words in Spam and Ham

![Figure_10](https://github.com/user-attachments/assets/dfcc8126-24cb-488b-ac9c-3c92b2649d40)

![Figure_11](https://github.com/user-attachments/assets/4926f7cb-293e-4c35-95c5-6474050b7b9d)


