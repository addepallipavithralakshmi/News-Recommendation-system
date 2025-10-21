from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
from model import NewsRecommendationEngine, download_and_load_dataset, preprocess_data

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

# Load and prepare dataset
df = download_and_load_dataset()
df = preprocess_data(df)

if df.empty:
    raise SystemExit("Dataset not found or empty.")

base_engine = NewsRecommendationEngine(df)
base_engine.build_model()


# -----------------------------------------------------------
# FRONTEND: Serve index.html
# -----------------------------------------------------------
@app.route("/")
def home():
    return render_template("index.html")


# -----------------------------------------------------------
# API: Get category statistics
# -----------------------------------------------------------
@app.route("/api/categories", methods=["GET"])
def categories():
    categories = sorted(df["category"].unique())
    category_stats = [
        {
            "name": cat,
            "article_count": len(df[df["category"] == cat]),
            "percentage": round((len(df[df["category"] == cat]) / len(df)) * 100, 2)
        }
        for cat in categories
    ]
    return jsonify({
        "total_articles": len(df),
        "total_categories": len(categories),
        "categories": category_stats
    })


# -----------------------------------------------------------
# API: Recommend by text
# -----------------------------------------------------------
@app.route("/api/recommend", methods=["POST"])
def recommend():
    payload = request.get_json(force=True)
    query_text = payload.get("query", "")
    top_k = int(payload.get("top_k", 5))
    try:
        results_df = base_engine.recommend_by_text_query(query_text, top_k=top_k)
        return jsonify(results_df.to_dict(orient="records"))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -----------------------------------------------------------
# API: Random articles
# -----------------------------------------------------------
@app.route("/api/random_articles", methods=["GET"])
def random_articles():
    n = int(request.args.get("n", 5))
    results_df = base_engine.get_random_articles(n=n)
    return jsonify(results_df.to_dict(orient="records"))


# -----------------------------------------------------------
# Run the app
# -----------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
