# Sentiment-Analysis
A Flask-based web application that predicts the sentiment (positive or negative) of user-provided text data. This project is designed to handle both individual text input and bulk predictions via CSV upload, making it a versatile tool for analyzing textual data.

# Text Sentiment Prediction

A Flask-based web application that predicts the sentiment (positive or negative) of user-provided text data. This project is designed to handle both individual text input and bulk predictions via CSV upload, making it a versatile tool for analyzing textual data.



---

## Features

- **Single Text Prediction**: Enter text directly into the app and get instant sentiment predictions.
- **Bulk Prediction**: Upload a CSV file with text data, and download a CSV file containing predictions for each entry.
- **Visual Analytics**: Generates a histogram to show sentiment distribution for bulk predictions.
- **Interactive and User-Friendly UI**: Seamlessly navigate between the landing page and prediction functionality.

---

## Project Structure

├── templates/ │ ├── landing.html # Landing page with project overview │ ├── index.html # Main interface for sentiment analysis ├── Models/ │ ├── model_xgb.pkl # Pre-trained XGBoost sentiment analysis model │ ├── scaler.pkl # Scaler for feature normalization │ ├── countVectorizer.pkl# CountVectorizer for text preprocessing ├── app.py # Backend Flask server └── requirements.txt # Python dependencies

---

## How It Works

1. **Data Preprocessing**:
   - The text is cleaned to remove unwanted characters, convert to lowercase, and remove stopwords.
   - Stemmed and vectorized using `CountVectorizer` before being scaled for prediction.

2. **Machine Learning Model**:
   - Pre-trained XGBoost model trained on sentiment-labeled data.
   - Predicts binary sentiment: `Positive` (1) or `Negative` (0).

3. **Output**:
   - Single text: Sentiment prediction displayed on the interface.
   - CSV file: A downloadable file with predictions and a histogram visualization of sentiment distribution.

---

## Installation and Setup

1. Clone this repository:
   
   git clone https://github.com/username/text-sentiment-prediction.git
   cd text-sentiment-prediction

2.Install dependencies:

    pip install -r requirements.txt
3.Run the application:
    python app.py
4.Open the application in your browser at http://127.0.0.1:5000/

Usage

Single Text Prediction

Navigate to the "Prediction" page.
Enter the text in the input field and click "Predict" to view the sentiment.

Bulk Prediction
Upload a CSV file with a column named text.
Download the predictions as a new CSV file and view the sentiment distribution graph.

Technologies Used

Backend: Flask
Frontend: HTML, CSS, JavaScript
Machine Learning: XGBoost, CountVectorizer
Data Visualization: Matplotlib

Screenshots

Landing Page
![image](https://github.com/user-attachments/assets/a4c55f7a-e723-42b7-a8fc-8279bea705b6)


Prediction Interface

![image](https://github.com/user-attachments/assets/7f74cc51-d508-461f-a393-4bf193f896d6)

Future Enhancements

Add support for multilingual sentiment analysis.
Expand the model to handle multi-class sentiment labels (e.g., neutral).
Include advanced visualizations like word clouds or sentiment trends.

Contributing
We welcome contributions! Feel free to fork the repository and submit a pull request with your improvements.


