from flask import Flask, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Home route for sanity check
@app.route('/')
def home():
    return "GenAI Assessment Recommender is live!"

# Load data
data = pd.read_csv("assessments_with_descriptions.csv")

# Vectorize descriptions
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(data['Description'])

@app.route('/recommend', methods=['POST'])
def recommend():
    content = request.get_json()
    assessment_name = content.get("assessment_name")

    if assessment_name not in data["Assessment_Name"].values:
        return jsonify({"error": "Assessment not found"}), 404

    idx = data[data["Assessment_Name"] == assessment_name].index[0]
    cosine_similarities = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    similar_indices = cosine_similarities.argsort()[::-1][1:6]
    recommendations = data.iloc[similar_indices][["Assessment_Name", "Category", "Difficulty"]]

    return jsonify({
        "recommended_assessments": recommendations.to_dict(orient="records")
    })

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))  
    app.run(host='0.0.0.0', port=port)
