# SMS Spam Detection using Naive Bayes

A standalone Python project for detecting spam SMS messages using Natural Language Processing (NLP) techniques and a Naive Bayes classifier. This project covers the complete machine learning pipeline including data preprocessing, exploratory data analysis (EDA), feature engineering, model training, and evaluation.

---

## Model Used

### Multinomial Naive Bayes

* Specifically designed for text classification problems.
* Efficient for features based on word counts or term frequencies.

---

## Dataset

* **Source**: [UCI SMS Spam Collection Dataset](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection)
* **Fields**:

  * `v1`: Message label (ham or spam)
  * `v2`: SMS text message

---

## Text Preprocessing

Applied the following preprocessing steps:

* Convert text to lowercase
* Tokenize text into words
* Remove stopwords and non-alphanumeric tokens
* Apply stemming using Porter Stemmer

---

## Exploratory Data Analysis (EDA)

Analyzed the following aspects:

* Class distribution: Spam vs Ham
* Feature distributions:

  * Number of characters
  * Number of words
  * Number of sentences
* Pairplot of engineered features
* Correlation heatmap of features

---

## Visualizations

* Word clouds for spam and ham messages
* Top 30 most frequent words:

  * In spam messages
  * In ham messages
* Count plot and pie chart for class label distribution

---

## Feature Engineering

Constructed additional features to enhance model performance:

* `num_characters`: Total number of characters in the message
* `num_words`: Total number of words in the message
* `num_sentences`: Estimated number of sentences in the message

---

## Model Evaluation

Performance metrics used for evaluation:

* Confusion matrix
* Accuracy score
* Precision score
* Detailed classification report

---

## Dependencies

Install the required Python packages using pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn nltk wordcloud
```

---

## How to Run the Project

1. Ensure the dataset file `spam.csv` is in the same directory as your script.
2. Run the script using:

```bash
python spam_detector.py
```

---

## Sample Visual Outputs

### Spam vs Ham Distribution

![Spam vs Ham Distribution](https://github.com/user-attachments/assets/29391768-3622-48b7-9d80-cd074a9ebb20)

![Class Distribution Pie Chart](https://github.com/user-attachments/assets/2b41cccd-443e-4bb8-a928-e55b7b94423a)

![Character Distribution Plot](https://github.com/user-attachments/assets/fbff9113-5248-465b-a08a-45ba922a7018)

### Word Clouds

#### Spam Messages

![Spam Word Cloud](https://github.com/user-attachments/assets/7d530d9b-b780-496e-989c-5aa31b742b40)

#### Ham Messages

![Ham Word Cloud](https://github.com/user-attachments/assets/bc0fff45-ea26-4feb-8480-791a56703d69)

### Top 30 Frequent Words

#### In Spam Messages

![Spam Frequent Words](https://github.com/user-attachments/assets/dfcc8126-24cb-488b-ac9c-3c92b2649d40)

#### In Ham Messages

![Ham Frequent Words](https://github.com/user-attachments/assets/4926f7cb-293e-4c35-95c5-6474050b7b9d)


