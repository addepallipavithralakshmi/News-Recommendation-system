"""
NEWS RECOMMENDATION SYSTEM
Using TF-IDF and Nearest Neighbors Algorithm

This system provides personalized news recommendations based on:
1. Category selection
2. Article similarity using TF-IDF vectors
3. K-Nearest Neighbors algorithm

Author: ML Project
Dataset: News Category Dataset from Kaggle
"""

# ============================================================================
# STEP 1: INSTALL AND IMPORT REQUIRED LIBRARIES
# ============================================================================

# Run this first in your environment:
# !pip install --quiet kagglehub pandas numpy scikit-learn matplotlib seaborn

import os
import glob
import warnings
warnings.filterwarnings('ignore')

# Data manipulation
import pandas as pd
import numpy as np

# Machine Learning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Kaggle dataset download
import kagglehub

# For interactive display
from IPython.display import display, HTML

print("=" * 80)
print("NEWS RECOMMENDATION SYSTEM - ML PROJECT")
print("=" * 80)
print("\n✓ All libraries imported successfully!\n")


# ============================================================================
# STEP 2: DOWNLOAD AND LOAD DATASET FROM KAGGLE
# ============================================================================

def download_and_load_dataset():
    """
    Downloads the News Category Dataset from Kaggle using kagglehub
    and loads it into a pandas DataFrame
    """
    print("-" * 80)
    print("STEP 2: DOWNLOADING DATASET FROM KAGGLE")
    print("-" * 80)

    try:
        # Download dataset using kagglehub (handles authentication automatically)
        print("\n📥 Downloading News Category Dataset from Kaggle...")
        path = kagglehub.dataset_download("rmisra/news-category-dataset")
        print(f"✓ Dataset downloaded to: {path}")

        # Find JSON file in downloaded directory
        json_files = glob.glob(os.path.join(path, "*.json"))
        if not json_files:
            json_files = glob.glob(os.path.join(path, "**", "*.json"), recursive=True)

        if not json_files:
            raise FileNotFoundError(f"No JSON file found in {path}")

        json_path = json_files[0]
        print(f"✓ Found dataset file: {os.path.basename(json_path)}")

        # Load dataset
        print("\n📊 Loading dataset into DataFrame...")
        df = pd.read_json(json_path, lines=True)
        print(f"✓ Successfully loaded {len(df):,} articles")

        return df

    except Exception as e:
        print(f"❌ Error loading dataset: {str(e)}")
        raise


# ============================================================================
# STEP 3: DATA PREPROCESSING AND EXPLORATION
# ============================================================================

def preprocess_data(df):
    """
    Cleans and preprocesses the dataset:
    - Combines text fields
    - Removes duplicates and empty entries
    - Creates unique identifiers
    - Displays statistics
    """
    print("\n" + "-" * 80)
    print("STEP 3: DATA PREPROCESSING")
    print("-" * 80)

    print("\n📋 Original Dataset Info:")
    print(f"   Total articles: {len(df):,}")
    print(f"   Columns: {', '.join(df.columns)}")

    # Handle missing values
    print("\n🔍 Checking for missing values...")
    df['headline'] = df['headline'].fillna('').astype(str)
    df['short_description'] = df['short_description'].fillna('').astype(str)
    df['category'] = df['category'].fillna('UNKNOWN').astype(str)

    # Combine headline and description for better content representation
    print("\n🔗 Combining headline and description into content field...")
    df['content'] = (df['headline'] + ' ' + df['short_description']).str.strip()

    # Remove empty content
    initial_count = len(df)
    df = df[df['content'].str.len() > 0].reset_index(drop=True)
    removed_empty = initial_count - len(df)
    if removed_empty > 0:
        print(f"   Removed {removed_empty} articles with empty content")

    # Remove duplicates
    initial_count = len(df)
    df = df.drop_duplicates(subset=['content']).reset_index(drop=True)
    removed_duplicates = initial_count - len(df)
    if removed_duplicates > 0:
        print(f"   Removed {removed_duplicates} duplicate articles")

    # Add unique identifier
    df['article_id'] = df.index

    print(f"\n✓ Final dataset: {len(df):,} articles")
    
    return df


def explore_categories(df):
    """
    Analyzes and displays category distribution
    """
    print("\n" + "-" * 80)
    print("STEP 4: CATEGORY ANALYSIS")
    print("-" * 80)

    category_counts = df['category'].value_counts()

    print(f"\n📊 Found {len(category_counts)} unique categories:")
    print("\nCategory Distribution (Top 15):")
    print("-" * 50)
    for i, (cat, count) in enumerate(category_counts.head(15).items(), 1):
        bar = "█" * int(count / category_counts.max() * 40)
        print(f"{i:2d}. {cat:20s} | {bar} {count:,}")

    # Display all categories for selection
    print("\n📝 All Available Categories:")
    print("-" * 50)
    all_categories = sorted(df['category'].unique())
    for i, cat in enumerate(all_categories, 1):
        count = category_counts[cat]
        print(f"{i:2d}. {cat:25s} ({count:,} articles)")

    return all_categories


# ============================================================================
# STEP 5: BUILD TF-IDF VECTORIZER AND NEAREST NEIGHBORS MODEL
# ============================================================================

class NewsRecommendationEngine:
    """
    Main recommendation engine using TF-IDF and K-Nearest Neighbors
    """

    def __init__(self, df, category=None, max_features=5000, ngram_range=(1, 2)):
        """
        Initialize the recommendation engine

        Parameters:
        -----------
        df : pandas DataFrame
            Complete news dataset
        category : str
            Category to filter articles (None for all categories)
        max_features : int
            Maximum number of TF-IDF features
        ngram_range : tuple
            N-gram range for TF-IDF (unigrams and bigrams by default)
        """
        self.df_full = df
        self.category = category
        self.max_features = max_features
        self.ngram_range = ngram_range

        # Filter by category if specified
        if category:
            self.df = df[df['category'] == category].reset_index(drop=True)
            print(f"\n🎯 Filtered to category '{category}': {len(self.df):,} articles")
        else:
            self.df = df.copy()
            print(f"\n🌐 Using all categories: {len(self.df):,} articles")

        if len(self.df) == 0:
            raise ValueError(f"No articles found for category: {category}")

        # Adjust max_features if dataset is small
        if len(self.df) < 100:
            self.max_features = min(max_features, 1000)

        self.vectorizer = None
        self.tfidf_matrix = None
        self.nn_model = None

    def build_model(self):
        """
        Builds TF-IDF vectors and trains Nearest Neighbors model
        """
        print("\n" + "-" * 80)
        print("STEP 5: BUILDING RECOMMENDATION MODEL")
        print("-" * 80)

        print("\n🔢 Building TF-IDF Vectorizer...")
        print(f"   Max features: {self.max_features}")
        print(f"   N-gram range: {self.ngram_range}")
        print(f"   Stop words: English")

        # Create TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=2,  # Ignore terms that appear in less than 2 documents
            max_df=0.8  # Ignore terms that appear in more than 80% of documents
        )

        # Fit and transform documents
        print("\n⚙️  Transforming documents to TF-IDF vectors...")
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df['content'])

        print(f"✓ TF-IDF matrix shape: {self.tfidf_matrix.shape}")
        print(f"   (articles: {self.tfidf_matrix.shape[0]}, features: {self.tfidf_matrix.shape[1]})")

        # Build Nearest Neighbors model
        print("\n🤖 Training K-Nearest Neighbors model...")
        print("   Metric: Cosine similarity")
        print("   Algorithm: Brute force")

        self.nn_model = NearestNeighbors(
            metric='cosine',
            algorithm='brute',
            n_jobs=-1  # Use all CPU cores
        )
        self.nn_model.fit(self.tfidf_matrix)

        print("✓ Model training complete!")

        return self

    def recommend_by_article_index(self, article_index, top_k=5):
        """
        Recommends similar articles based on article index

        Parameters:
        -----------
        article_index : int
            Index of the article in the filtered dataset
        top_k : int
            Number of recommendations to return

        Returns:
        --------
        pandas DataFrame with recommendations
        """
        if self.nn_model is None:
            raise ValueError("Model not built. Call build_model() first.")

        if not (0 <= article_index < len(self.df)):
            raise ValueError(f"Article index must be between 0 and {len(self.df)-1}")

        # Find k+1 nearest neighbors (including the article itself)
        n_neighbors = min(top_k + 1, len(self.df))
        distances, indices = self.nn_model.kneighbors(
            self.tfidf_matrix[article_index],
            n_neighbors=n_neighbors
        )

        # Convert to 1D arrays
        distances = distances.flatten()
        indices = indices.flatten()

        # Calculate similarity scores (1 - distance)
        similarities = 1 - distances

        # Create recommendations DataFrame (skip first item as it's the article itself)
        recommendations = []
        for idx, sim in zip(indices[1:], similarities[1:]):
            article = self.df.iloc[idx]
            recommendations.append({
                'article_id': int(article['article_id']),
                'category': article['category'],
                'headline': article['headline'],
                'short_description': article['short_description'],
                'link': article.get('link', ''),
                'similarity_score': float(sim)
            })

        return pd.DataFrame(recommendations)
    
    def _engine(self):
        """
        Returns the internal Nearest Neighbors model
        """
        return self.nn_model

    
    def recommend_by_text_query(self, query_text, top_k=5):
        """
        Recommends similar articles based on a free-text query.

        Parameters:
        -----------
        query_text : str
            The input text to find similar articles for
        top_k : int
            Number of recommendations to return

        Returns:
        --------
        pandas DataFrame with recommendations
        """
        if self.nn_model is None or self.vectorizer is None:
            raise ValueError("Model not built. Call build_model() first.")
        if not query_text or not isinstance(query_text, str):
            raise ValueError("query_text must be a non-empty string.")

        # Transform the query to TF-IDF vector
        query_vec = self.vectorizer.transform([query_text])

        n_neighbors = min(top_k, len(self.df))  # don't ask for more than available
        distances, indices = self.nn_model.kneighbors(query_vec, n_neighbors=n_neighbors)

        distances = distances.flatten()
        indices = indices.flatten()
        similarities = 1 - distances

        recommendations = []
        for idx, sim in zip(indices, similarities):
            article = self.df.iloc[idx]
            recommendations.append({
                'article_id': int(article['article_id']),
                'category': article['category'],
                'headline': article['headline'],
                'short_description': article['short_description'],
                'link': article.get('link', ''),
                'similarity_score': float(sim)
            })

        return pd.DataFrame(recommendations)


    def get_random_articles(self, n=5):
        """
        Returns random articles from the dataset for exploration
        """
        sample_size = min(n, len(self.df))
        sample = self.df.sample(n=sample_size)

        return sample[['article_id', 'category', 'headline', 'short_description']].reset_index(drop=True)


# ============================================================================
# STEP 6: DISPLAY FUNCTIONS FOR RECOMMENDATIONS
# ============================================================================

def display_article(df, index):
    """
    Displays a single article with formatting
    """
    article = df.iloc[index]
    print("\n" + "=" * 80)
    print("SELECTED ARTICLE")
    print("=" * 80)
    print(f"\n📰 {article['headline']}")
    print(f"\n📁 Category: {article['category']}")
    print(f"\n📝 Description:")
    print(f"   {article['short_description']}")
    if 'link' in article and article['link']:
        print(f"\n🔗 Link: {article['link']}")
    print("=" * 80)



def display_recommendations(recommendations_df, title="RECOMMENDATIONS"):
    """
    Displays recommendations in a formatted way
    """
    if recommendations_df.empty:
        print("\n❌ No recommendations found.")
        return

    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    for i, row in recommendations_df.iterrows():
        print(f"\n{i+1}. 📊 Similarity: {row['similarity_score']:.3f}")
        print(f"   📰 {row['headline']}")
        print(f"   📁 Category: {row['category']}")
        print(f"   📝 {row['short_description'][:150]}...")
        if row['link']:
            print(f"   🔗 {row['link']}")
        print("-" * 80)


# ============================================================================
# STEP 7: INTERACTIVE RECOMMENDATION INTERFACE
# ============================================================================

def interactive_recommendation_system(df, categories):
    """
    Main interactive interface for the recommendation system
    """
    print("\n" + "=" * 80)
    print("INTERACTIVE NEWS RECOMMENDATION SYSTEM")
    print("=" * 80)

    while True:
        print("\n" + "-" * 80)
        print("MAIN MENU")
        print("-" * 80)
        print("\n1. Get recommendations by category")
        print("2. Get recommendations by text query")
        print("3. Explore random articles")
        print("4. View category statistics")
        print("5. Exit")

        try:
            choice = input("\n👉 Enter your choice (1-5): ").strip()

            if choice == '1':
                # Category-based recommendations
                print("\n📋 Available Categories:")
                for i, cat in enumerate(categories, 1):
                    print(f"{i:2d}. {cat}")

                cat_choice = int(input(f"\n👉 Select category (1-{len(categories)}): ").strip())

                if not (1 <= cat_choice <= len(categories)):
                    print("❌ Invalid category selection!")
                    continue

                selected_category = categories[cat_choice - 1]

                # Build model for selected category
                print(f"\n🔨 Building model for '{selected_category}'...")
                engine = NewsRecommendationEngine(df, category=selected_category)
                engine.build_model()

                # Show sample articles
                print("\n📚 Sample articles from this category:")
                sample = engine.get_random_articles(n=10)
                for i, row in sample.iterrows():
                    print(f"{i}. {row['headline'][:80]}...")

                article_idx = int(input(f"\n👉 Select article index (0-{len(engine.df)-1}): ").strip())

                if not (0 <= article_idx < len(engine.df)):
                    print("❌ Invalid article index!")
                    continue

                # Display selected article
                display_article(engine.df, article_idx)

                # Get recommendations
                top_k = int(input("\n👉 How many recommendations? (1-20): ").strip())
                top_k = max(1, min(20, top_k))

                print("\n🔍 Finding similar articles...")
                recommendations = engine.recommend_by_article_index(article_idx, top_k=top_k)
                display_recommendations(recommendations, f"TOP {top_k} SIMILAR ARTICLES")

            elif choice == '2':
                # Query-based recommendations
                print("\n📋 Available Categories:")
                for i, cat in enumerate(categories, 1):
                    print(f"{i:2d}. {cat}")
                print(f"{len(categories)+1}. All Categories")

                cat_choice = int(input(f"\n👉 Select category (1-{len(categories)+1}): ").strip())

                if cat_choice == len(categories) + 1:
                    selected_category = None
                elif 1 <= cat_choice <= len(categories):
                    selected_category = categories[cat_choice - 1]
                else:
                    print("❌ Invalid category selection!")
                    continue

                # Build model
                print(f"\n🔨 Building model...")
                engine = NewsRecommendationEngine(df, category=selected_category)
                engine.build_model()

                # Get query
                query = input("\n👉 Enter your search query: ").strip()

                if not query:
                    print("❌ Query cannot be empty!")
                    continue

                # Get number of results
                top_k = int(input("👉 How many recommendations? (1-20): ").strip())
                top_k = max(1, min(20, top_k))

                # Get recommendations
                print(f"\n🔍 Searching for articles matching: '{query}'...")
                recommendations = engine.recommend_by_text_query(query, top_k=top_k)
                display_recommendations(recommendations, f"TOP {top_k} MATCHES FOR '{query}'")

            elif choice == '3':
                # Explore random articles
                n = int(input("\n👉 How many random articles to show? (1-20): ").strip())
                n = max(1, min(20, n))

                sample = df.sample(n=n)
                print("\n" + "=" * 80)
                print(f"RANDOM ARTICLES (Showing {n})")
                print("=" * 80)

                for i, (idx, row) in enumerate(sample.iterrows(), 1):
                    print(f"\n{i}. 📰 {row['headline']}")
                    print(f"   📁 Category: {row['category']}")
                    print(f"   📝 {row['short_description'][:150]}...")
                    print("-" * 80)

            elif choice == '4':
                # Show statistics
                explore_categories(df)

            elif choice == '5':
                print("\n👋 Thank you for using the News Recommendation System!")
                break

            else:
                print("❌ Invalid choice! Please select 1-5.")

        except ValueError as e:
            print(f"❌ Invalid input: {e}")
        except KeyboardInterrupt:
            print("\n\n👋 Exiting...")
            break
        except Exception as e:
            print(f"❌ An error occurred: {e}")


# ============================================================================
# STEP 8: MAIN EXECUTION
# ============================================================================

def main():
    """
    Main function to run the complete news recommendation system
    """
    try:
        # Step 1: Download and load dataset
        df = download_and_load_dataset()

        # Step 2: Preprocess data
        df = preprocess_data(df)

        # Step 3: Explore categories
        categories = explore_categories(df)

        # Step 4: Run interactive system
        interactive_recommendation_system(df, categories)

    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()


# ============================================================================
# RUN THE SYSTEM
# ============================================================================

if __name__ == "__main__":
    main()