# SMS #Spam Detection Using Python and Machine Learning

## 1. Project Overview

SMS Spam Detection is a Python-based machine learning project that identifies whether a text message is Spam or Ham (Not Spam).

The main purpose of this project is to classify SMS messages automatically and help users identify unwanted or suspicious messages.

The project uses Natural Language Processing (NLP) techniques and a Machine Learning algorithm to make predictions.

## 2. Objectives

- To identify spam and genuine SMS messages.
- To apply Machine Learning for text classification.
- To understand text preprocessing and feature extraction.
- To develop a simple web application using Flask.
- To store and manage previous prediction records.

## 3. Technologies Used

- **Python:** Main programming language.
- **Flask:** Used to develop the web application.
- **Pandas:** Used for data handling.
- **Scikit-learn:** Used for machine learning and model training.
- **TF-IDF:** Used to convert text messages into numerical features.
- **Multinomial Naive Bayes:** Used for SMS classification.
- **SQLite:** Used to store prediction history.
- **HTML and CSS:** Used for the web interface.

## 4. Project Features

- SMS message input for spam checking.
- Automatic classification into Spam or Not Spam.
- Machine Learning-based prediction.
- Prediction history storage.
- View previous prediction records.
- Clear prediction history.
- Simple and user-friendly interface.

## 5. Project Structure

```text
SMS-Spam-Detection-Python/
│
├── data/
│   └── SMSspamCollection
│
├── database/
│   ├── predictions.db
│   └── schema.sql
│
├── model/
│   ├── spam_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── static/
│   └── style.css
│
├── templates/
│   ├── index.html
│   ├── history.html
│   └── daa_analysis.html
│
├── app.py
├── train_model.py
├── requirement.txt
└── README.md
```

## 6. How the Project Works

1. The user enters an SMS message into the web application.
2. The message is converted into numerical features using TF-IDF.
3. The trained Machine Learning model analyses the message.
4. The model predicts whether the message is Spam or Not Spam.
5. The prediction result is displayed to the user.
6. The prediction record is stored in the SQLite database.

## 7. Installation and Execution

**Step 1:** Clone the repository.

```bash
git clone https://github.com/suhani1005/SMS-Spam-Detection-Python.git
```

**Step 2:** Open the project folder.

```bash
cd SMS-Spam-Detection-Python
```

**Step 3:** Install the required libraries.

```bash
pip install -r requirement.txt
```

**Step 4:** Run the Flask application.

```bash
python app.py
```

**Step 5:** Open the local application in your browser using the URL displayed in the terminal (usually http://127.0.0.1:5000).

## 8. Machine Learning Model

The project uses the Multinomial Naive Bayes algorithm for text classification.

TF-IDF (Term Frequency-Inverse Document Frequency) is used for feature extraction. It converts text messages into numerical values that can be processed by the machine learning model.

## 9. Database

SQLite is used to maintain prediction history. It stores the messages, their predicted categories and prediction timestamps.

## 10. Conclusion

This project demonstrates how Python and Machine Learning can be used to solve a real-world problem like SMS spam detection.

It provides practical knowledge of text classification, feature extraction, model training, Flask web development and database management.
