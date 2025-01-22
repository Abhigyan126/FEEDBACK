import os
import re
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from textblob import TextBlob
from sentence_transformers import SentenceTransformer
import markdown
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
from llm import LLM

load_dotenv()
model = SentenceTransformer('all-MiniLM-L6-v2')

class CommentCleaner:
    @staticmethod
    def clean_comment(comment):
        clean_text = re.sub(r'<.*?>', '', comment)
        clean_text = re.sub(r'[\x00-\x7F]+', ' ', clean_text)
        return clean_text

class SentimentAnalyzer:
    @staticmethod
    def analyze_sentiment(comment):
        blob = TextBlob(comment)
        return blob.sentiment.polarity

    @staticmethod
    def summarize_sentiments(comments):
        positive, negative, neutral = 0, 0, 0
        total_sentiment = 0
        for comment in comments:
            cleaned_comment = CommentCleaner.clean_comment(comment)
            sentiment_score = SentimentAnalyzer.analyze_sentiment(cleaned_comment)
            if sentiment_score > 0:
                positive += 1
            elif sentiment_score < 0:
                negative += 1
            else:
                neutral += 1
            total_sentiment += sentiment_score
        average_sentiment = total_sentiment / len(comments) if comments else 0
        return {
            "positive": positive,
            "negative": negative,
            "neutral": neutral,
            "average_sentiment": average_sentiment
        }

class InsightGenerator:
    def __init__(self, llm_class):
        self.llm_class = llm_class

    def generate_insight(self, comments, similarity_threshold=0.85):
        cleaned_comments = [CommentCleaner.clean_comment(comment) for comment in comments]
        embeddings = model.encode(cleaned_comments)
        similarity_matrix = cosine_similarity(embeddings)
        unique_comments = []
        for i in range(len(comments)):
            if all(similarity_matrix[i][j] < similarity_threshold for j in range(i)):
                unique_comments.append(comments[i])
        removed_count = len(comments) - len(unique_comments)
        print(f"Removed {removed_count} similar comments based on cosine similarity.")
        all_comments_text = " ".join(unique_comments)
        prompt = (
            f"Here are the comments from a CSV file:\n"
            f"{all_comments_text}\n"
            f"Can you provide insights on how the audience feels about the content? "
            f"Give detailed suggestions to improve, based on these comments."
        )
        llm_instance = self.llm_class()
        insight = llm_instance.model(prompt)
        insight = markdown.markdown(insight)
        return insight

app = Flask(__name__)
sentiment_analyzer = SentimentAnalyzer()
insight_generator = InsightGenerator(LLM)
comments = []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_csv', methods=['POST'])
def upload_csv():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    try:
        df = pd.read_csv(file)
        if 'comment' not in df.columns:
            return jsonify({"error": "CSV must contain a 'comment' column."}), 400
        global comments
        comments = df['comment'].dropna().tolist()
        return jsonify({"message": "CSV uploaded successfully.", "total_comments": len(comments)}), 200
    except Exception as e:
        return jsonify({"error": f"Failed to process CSV file: {str(e)}"}), 500

@app.route('/analyze', methods=['POST'])
def analyze():
    max_comments = int(request.form.get('max_comments', 1000))
    similarity_threshold = float(request.form.get('similarity_threshold', 0.85))
    if not comments:
        return jsonify({"error": "No comments available. Upload a CSV first."})
    selected_comments = comments[:max_comments]
    sentiment_summary = sentiment_analyzer.summarize_sentiments(selected_comments)
    insight = insight_generator.generate_insight(selected_comments, similarity_threshold=similarity_threshold)
    return jsonify({
        "sentiment_summary": sentiment_summary,
        "insight": insight
    })

@app.route('/send_message', methods=['POST'])
def send_message():
    user_message = request.form.get('user_message').strip()
    if user_message == "":
        return jsonify({"response": "Please enter a message."})
    all_comments_text = "\n".join(comments)
    prompt = (
        f"Here are the comments from a CSV file:\n"
        f"{all_comments_text}\n\n"
        f"User question: {user_message}\n"
        "Based on the above comments, please provide a detailed response to the user's question.\n"
        "Format the response with proper spacing, and use **bold** for key points and *italics* for emphasis."
    )
    llm_instance = insight_generator.llm_class()
    response = llm_instance.model(prompt)
    formatted_response = markdown.markdown(response)
    return jsonify({"response": formatted_response})

if __name__ == "__main__":
    app.run(debug=True)
