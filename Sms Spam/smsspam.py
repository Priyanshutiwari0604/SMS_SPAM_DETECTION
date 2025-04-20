# -------------------- IMPORTS --------------------
import pandas as pd  # For handling tabular data
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For plotting graphs
import seaborn as sns  # For statistical data visualization
from sklearn.preprocessing import LabelEncoder  # For converting labels to numeric
from sklearn.model_selection import train_test_split  # For splitting data into training and testing
from sklearn.feature_extraction.text import TfidfVectorizer  # For converting text to numerical vectors
from sklearn.naive_bayes import MultinomialNB  # For Naive Bayes classification
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score  # For evaluation
import nltk  # For natural language processing
from nltk.tokenize import word_tokenize, sent_tokenize  # For splitting text into words/sentences
from nltk.corpus import stopwords  # For removing common stopwords like "the", "is", etc.
from nltk.stem.porter import PorterStemmer  # For stemming words to their root form
from wordcloud import WordCloud  # For generating word clouds
from collections import Counter  # For counting word frequencies

# -------------------- NLTK SETUP --------------------
# Download tokenizers and stopword list (force=True ensures redownload)
nltk.download('punkt', force=True)  # Ensure tokenizer is available
nltk.download('stopwords', force=True)  # Ensure stopwords list is available

# Initialize the stemmer (PorterStemmer is widely used for stemming in NLP)
ps = PorterStemmer()

# -------------------- DATA LOADING --------------------
# Load the dataset (ensure correct path & encoding)
df = pd.read_csv(r"D:\MACHINE LEARNING PROJECTS\Sms Spam\spam.csv", encoding='latin1')

# -------------------- DATA CLEANING --------------------
# Clean column names by stripping whitespace
df.columns = df.columns.str.strip()  # Ensure no trailing spaces in column names

# Drop irrelevant columns that are not useful for analysis
df = df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'])

# Rename relevant columns for clarity
df = df.rename(columns={'v1': 'label', 'v2': 'message'})

# Encode target column: 'ham' → 0, 'spam' → 1
encoder = LabelEncoder()
df["label"] = encoder.fit_transform(df["label"])  # Convert 'ham' and 'spam' to 0 and 1

# Remove duplicate rows from the dataset to avoid redundancy
df = df.drop_duplicates(keep="first")

# -------------------- EDA (EXPLORATORY DATA ANALYSIS) --------------------
# Print the distribution of labels (ham vs spam messages)
print("Class distribution:")
print(df["label"].value_counts())  # Count number of ham and spam messages
print("Ham messages:", df['label'].value_counts()[0])
print("Spam messages:", df['label'].value_counts()[1])

# Visualize the class distribution using a pie chart
plt.figure(figsize=(6, 6))
plt.pie(df["label"].value_counts(), labels=["Ham", "Spam"],
        autopct="%0.2f%%", colors=["skyblue", "lightcoral"])
plt.title("Spam vs Ham Distribution")  # Display the title
plt.show()  # Show the pie chart

# Visualize the class distribution using a bar plot
plt.figure(figsize=(6, 4))
sns.countplot(x='label', data=df, hue='label',
              palette=['skyblue', 'lightcoral'], legend=False)
plt.xticks(ticks=[0, 1], labels=["Ham", "Spam"])  # Set custom labels for the x-axis
plt.title("Count of Ham vs Spam Messages")  # Display the title
plt.xlabel("Label")  # x-axis label
plt.ylabel("Count")  # y-axis label
plt.show()  # Show the bar plot

# -------------------- FEATURE ENGINEERING --------------------
# Add new features to the dataframe
# Add feature: number of characters in each message
df['num_characters'] = df['message'].apply(len)

# Add feature: number of words in each message
df['num_words'] = df['message'].apply(lambda x: len(word_tokenize(x)))

# Add feature: number of sentences in each message
df['num_sentence'] = df['message'].apply(lambda x: len(sent_tokenize(x)))

# Print statistical summary for ham and spam messages for the new features
print("\nSample data with new features:")
print(df[df['label'] == 0][['num_characters', 'num_words', 'num_sentence']].describe())
print(df[df['label'] == 1][['num_characters', 'num_words', 'num_sentence']].describe())

# Plot: Distribution of character count (ham vs spam)
plt.figure(figsize=(6, 4))
sns.histplot(df[df['label'] == 0]['num_characters'], color='skyblue', kde=True, label="Ham", stat='density', bins=30)
sns.histplot(df[df['label'] == 1]['num_characters'], color='lightcoral', kde=True, label="Spam", stat='density', bins=30)
plt.title('Distribution of Number of Characters in Ham vs Spam Messages')
plt.legend()
plt.show()

# Plot: Distribution of word count (ham vs spam)
plt.figure(figsize=(6, 4))
sns.histplot(df[df['label'] == 0]['num_words'], color='skyblue', kde=True, label="Ham", stat='density', bins=30)
sns.histplot(df[df['label'] == 1]['num_words'], color='lightcoral', kde=True, label="Spam", stat='density', bins=30)
plt.title('Distribution of Number of Words in Ham vs Spam Messages')
plt.legend()
plt.show()

# Plot: Distribution of sentence count (ham vs spam)
plt.figure(figsize=(6, 4))
sns.histplot(df[df['label'] == 0]['num_sentence'], color='skyblue', kde=True, label="Ham", stat='density', bins=30)
sns.histplot(df[df['label'] == 1]['num_sentence'], color='lightcoral', kde=True, label="Spam", stat='density', bins=30)
plt.title('Distribution of Number of Sentences in Ham vs Spam Messages')
plt.legend()
plt.show()

# Pairplot to visualize feature relationships between ham and spam
sns.pairplot(df, hue='label')  # Plots all pairwise combinations
plt.show()

# Compute correlation matrix between features
correlation_matrix = df[['num_characters', 'num_words', 'num_sentence', 'label']].corr()
print("\nCorrelation Matrix:")
print(correlation_matrix)

# Heatmap to visualize correlations between features
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", cbar=True)
plt.title('Correlation Matrix')
plt.show()

# Preview the dataset with new features
print(df.head())

# -------------------- TEXT PREPROCESSING FUNCTION --------------------
# Function to preprocess the text:
# - Lowercase all text
# - Tokenize text (split into words)
# - Remove non-alphanumeric characters
# - Remove stopwords (e.g., "the", "a", "is")
# - Apply stemming (to get root words)

def transform_text(text):
    text = text.lower()  # Convert text to lowercase
    text = nltk.word_tokenize(text)  # Tokenize the message into words

    # Remove non-alphanumeric tokens (numbers and symbols)
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    # Remove stopwords (common words like 'the', 'a', etc.)
    text = []
    for i in y:
        if i not in stopwords.words('english'):
            text.append(i)

    # Apply stemming (reduce words to their root form)
    y = []
    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Apply preprocessing function to the message column
df['transformed_message'] = df['message'].apply(transform_text)

# Display some of the original and transformed messages for verification
print(df[['message', 'transformed_message']].head())

# -------------------- WORD CLOUD VISUALIZATION --------------------
# Create word cloud for spam messages
spam_corpus = []  # Empty list to store words from spam messages
for msg in df[df['label'] == 1]['transformed_message'].tolist():
    for word in msg.split():
        spam_corpus.append(word)

# Generate and display the word cloud for spam messages
spam_wc = WordCloud(width=500, height=500, min_font_size=10, background_color='white').generate(" ".join(spam_corpus))
plt.figure(figsize=(6, 6))
plt.imshow(spam_wc)  # Show the word cloud
plt.axis("off")  # Turn off axis labels
plt.title("Spam Word Cloud")
plt.show()

# Create word cloud for ham messages
ham_corpus = []  # Empty list to store words from ham messages
for msg in df[df['label'] == 0]['transformed_message'].tolist():
    for word in msg.split():
        ham_corpus.append(word)

# Generate and display the word cloud for ham messages
ham_wc = WordCloud(width=500, height=500, min_font_size=10, background_color='white').generate(" ".join(ham_corpus))
plt.figure(figsize=(6, 6))
plt.imshow(ham_wc)  # Show the word cloud
plt.axis("off")  # Turn off axis labels
plt.title("Ham Word Cloud")
plt.show()

# -------------------- TOP 30 MOST COMMON WORDS IN SPAM --------------------
# Create a Counter object to get the most common words in the spam corpus
spam_word_counts = Counter(spam_corpus).most_common(30)

# Convert the list of tuples into a DataFrame
spam_word_df = pd.DataFrame(spam_word_counts, columns=["Word", "Count"])

# Plot the 30 most common words in spam messages
plt.figure(figsize=(10, 6))
sns.barplot(x="Count", y="Word", data=spam_word_df, palette="Blues_d")
plt.title("Top 30 Most Common Words in Spam Messages")
plt.show()

# -------------------- TOP 30 MOST COMMON WORDS IN HAM --------------------
# Create a Counter object to get the most common words in the ham corpus
ham_word_counts = Counter(ham_corpus).most_common(30)

# Convert the list of tuples into a DataFrame
ham_word_df = pd.DataFrame(ham_word_counts, columns=["Word", "Count"])

# Plot the 30 most common words in ham messages
plt.figure(figsize=(10, 6))
sns.barplot(x="Count", y="Word", data=ham_word_df, palette="Reds_d")
plt.title("Top 30 Most Common Words in Ham Messages")
plt.show()

# -------------------- MODEL TRAINING & EVALUATION --------------------
# Split data into train and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(df['transformed_message'], df['label'], test_size=0.2, random_state=42)

# Convert the text data to numerical data using TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=5000)  # Limit to top 5000 features for efficiency
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Train the Naive Bayes classifier
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)

# Predict on the test set
y_pred = nb_model.predict(X_test_tfidf)

# -------------------- MODEL EVALUATION --------------------
# Calculate precision score
precision = precision_score(y_test, y_pred)
print(f"Precision Score: {precision:.4f}")

# Calculate other metrics for evaluation
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Print all evaluation metrics
print(f"Accuracy: {accuracy:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
