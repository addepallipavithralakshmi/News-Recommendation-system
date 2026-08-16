# 📰 Personalized News Recommendation System

A machine learning-based **Personalized News Recommendation System** that recommends relevant news articles to users based on the similarity between news content.

The system uses **TF-IDF (Term Frequency–Inverse Document Frequency)** to convert news articles into numerical vectors and **K-Nearest Neighbors (KNN)** to identify and recommend similar articles.

## 📌 Project Overview

With thousands of news articles published every day, finding relevant news can be difficult and time-consuming.

This project provides a personalized recommendation system that analyzes the content of news articles and recommends similar articles based on their textual features.

The system follows a **content-based filtering approach**, meaning recommendations are generated based on the content and characteristics of the articles rather than other users' preferences.

## ✨ Features

* 📰 News article recommendation
* 🔍 Content-based filtering
* 📊 TF-IDF text vectorization
* 🤖 KNN-based similarity detection
* 🎯 Personalized and relevant recommendations
* ⚡ Fast similarity-based recommendations
* 🧹 Text preprocessing and feature extraction
* 📈 Recommendation accuracy evaluation

## 🛠️ Technologies Used

| Technology           | Purpose                          |
| -------------------- | -------------------------------- |
| **Python**           | Main programming language        |
| **Pandas**           | Data processing and manipulation |
| **NumPy**            | Numerical operations             |
| **Scikit-learn**     | Machine learning implementation  |
| **TF-IDF**           | Text feature extraction          |
| **KNN**              | Finding similar news articles    |
| **Jupyter Notebook** | Development and experimentation  |

## 🧠 Machine Learning Approach

### 1. Data Collection

A dataset containing news articles and their textual information is used as the input.

The important information from each article is extracted for recommendation.

### 2. Data Preprocessing

The news text is cleaned before applying machine learning techniques.

Typical preprocessing steps include:

* Removing unnecessary characters
* Handling missing values
* Converting text to lowercase
* Removing unnecessary words
* Preparing text for feature extraction

### 3. TF-IDF Vectorization

TF-IDF converts textual news content into numerical vectors.

It assigns importance to words based on:

* How frequently a word appears in an article
* How frequently the word appears across all articles

This allows the system to represent news articles mathematically.

### 4. K-Nearest Neighbors

KNN is used to identify articles that are most similar to a selected news article.

The system compares the TF-IDF representations and finds the nearest articles.

### 5. Recommendation

After finding similar articles, the system returns the most relevant news articles as recommendations.

## 🔄 System Workflow

```text
News Dataset
     ↓
Data Cleaning
     ↓
Text Preprocessing
     ↓
TF-IDF Vectorization
     ↓
Convert Articles into Vectors
     ↓
KNN Similarity Analysis
     ↓
Find Similar Articles
     ↓
Generate Recommendations
```

## 📊 Recommendation Method

The project uses **content-based recommendation**.

For example:

If a user selects an article about:

> "Artificial Intelligence and Machine Learning"

the system analyzes the article's textual features and recommends other articles containing similar concepts, such as:

* Artificial Intelligence
* Machine Learning
* Deep Learning
* Neural Networks
* Generative AI

## 🎯 Project Objective

The main objective of this project is to:

* Reduce the time required to find relevant news.
* Provide personalized news recommendations.
* Apply NLP techniques to news articles.
* Demonstrate the practical use of machine learning in recommendation systems.
* Improve the relevance of recommended articles using content similarity.

## 📈 Results

The recommendation system achieved approximately **85% recommendation accuracy** based on the evaluation performed for the project.

The combination of **TF-IDF feature extraction and KNN similarity analysis** helped identify relevant and similar news articles effectively.

## 📂 Project Structure

```text
Personalized-News-Recommendation-System/
│
├── dataset/
│   └── news_dataset.csv
│
├── notebooks/
│   └── news_recommendation.ipynb
│
├── src/
│   └── recommendation.py
│
├── requirements.txt
├── README.md
└── LICENSE
```

## 🚀 Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/personalized-news-recommendation-system.git
```

### Step 2: Navigate to the Project

```bash
cd personalized-news-recommendation-system
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

## 📦 Required Libraries

```text
pandas
numpy
scikit-learn
jupyter
```

## ▶️ How to Run

Start Jupyter Notebook:

```bash
jupyter notebook
```

Open the recommendation notebook and run the cells sequentially.

The system will:

1. Load the news dataset.
2. Preprocess the news content.
3. Generate TF-IDF vectors.
4. Apply KNN.
5. Calculate similarity between articles.
6. Display recommended news articles.

## 💡 Example

### Input

```text
Artificial Intelligence is transforming the technology industry
through machine learning and deep learning.
```

### Recommended News

```text
1. Latest developments in Artificial Intelligence
2. Applications of Machine Learning
3. Deep Learning technologies
4. Future of AI in the technology industry
5. AI-powered applications
```

## 🔑 Key Concepts Demonstrated

* Natural Language Processing
* Text preprocessing
* Feature extraction
* TF-IDF
* K-Nearest Neighbors
* Content-based filtering
* Recommendation systems
* Machine learning
* Similarity analysis

## 🔮 Future Enhancements

* Add user login and personalized profiles.
* Add a web interface using Flask or Streamlit.
* Include user reading history.
* Add category-based recommendations.
* Implement collaborative filtering.
* Use advanced NLP models such as BERT.
* Add real-time news APIs.
* Improve recommendation accuracy using hybrid recommendation techniques.

## 👩‍💻 Author

**Pavitra Lakshmi**

Computer Science / Software Engineering Student

---

⭐ If you find this project useful, consider giving the repository a star!
