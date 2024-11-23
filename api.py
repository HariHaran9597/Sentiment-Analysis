from flask import Flask, request, jsonify, send_file, render_template
import pandas as pd
import pickle
import re
from io import BytesIO
import base64
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

app = Flask(__name__)
STOPWORDS = set(stopwords.words("english"))

@app.route("/")
def landing_page():
    return render_template("landing.html")

@app.route("/predict-page")
def prediction_page():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    predictor = pickle.load(open("Models/model_xgb.pkl", "rb"))
    scaler = pickle.load(open("Models/scaler.pkl", "rb"))
    cv = pickle.load(open("Models/countVectorizer.pkl", "rb"))

    if "file" in request.files:
        file = request.files["file"]
        data = pd.read_csv(file)
        predictions, graph = bulk_prediction(predictor, scaler, cv, data)

        response = send_file(
            predictions,
            mimetype="text/csv",
            as_attachment=True,
            download_name="Predictions.csv",
        )
        response.headers["X-Graph-Exists"] = "true"
        response.headers["X-Graph-Data"] = base64.b64encode(graph.getbuffer()).decode("ascii")
        return response

    elif "text" in request.json:
        text_input = request.json["text"]
        predicted_sentiment = single_prediction(predictor, scaler, cv, text_input)
        return jsonify({"prediction": predicted_sentiment})

    return jsonify({"error": "Invalid input"}), 400

def single_prediction(predictor, scaler, cv, text_input):
    stemmer = PorterStemmer()
    review = re.sub("[^a-zA-Z]", " ", text_input)
    review = review.lower().split()
    review = [stemmer.stem(word) for word in review if word not in STOPWORDS]
    review = " ".join(review)
    X_prediction = cv.transform([review]).toarray()
    scaled_data = scaler.transform(X_prediction)
    prediction = predictor.predict(scaled_data)
    return "Positive" if prediction == 1 else "Negative"

def bulk_prediction(predictor, scaler, cv, data):
    corpus = []
    stemmer = PorterStemmer()
    for text in data["text"]:
        review = re.sub("[^a-zA-Z]", " ", text)
        review = review.lower().split()
        review = [stemmer.stem(word) for word in review if word not in STOPWORDS]
        corpus.append(" ".join(review))
    X = cv.transform(corpus).toarray()
    scaled_X = scaler.transform(X)
    predictions = predictor.predict(scaled_X)

    results = pd.DataFrame({"Text": data["text"], "Prediction": predictions})
    predictions_csv = BytesIO()
    results.to_csv(predictions_csv, index=False)
    predictions_csv.seek(0)

    plt.figure(figsize=(6, 4))
    plt.hist(predictions, bins=2, alpha=0.7, color="blue", rwidth=0.85)
    plt.title("Sentiment Distribution")
    plt.xlabel("Sentiment")
    plt.ylabel("Frequency")
    graph = BytesIO()
    plt.savefig(graph, format="png")
    graph.seek(0)

    return predictions_csv, graph

if __name__ == "__main__":
    app.run(debug=True)
