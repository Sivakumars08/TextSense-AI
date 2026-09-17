# TextSense AI — Intelligent Text Analytics & Predictor

## Overview

TextSense AI is an SMS spam classification system that uses natural language processing and machine learning to classify messages as HAM or SPAM.

The system preprocesses SMS text, converts it into numerical features using TF-IDF, and uses Logistic Regression to make the final classification.

## Problem Statement

SMS users receive unwanted promotional or fraudulent messages mixed with legitimate personal messages. Manually identifying spam can be difficult, especially when spam messages use different wording and formats.

TextSense AI addresses this problem by automatically classifying SMS messages as HAM or SPAM using natural language processing and machine learning.

## Objectives

- Clean and preprocess SMS text.
- Analyze characteristics of HAM and SPAM messages.
- Convert text into numerical features using TF-IDF.
- Train machine learning models for SMS classification.
- Compare different modelling approaches using evaluation metrics.
- Analyze incorrect predictions and identify common error patterns.
- Tune the final model to improve its performance.
- Provide predictions for new SMS messages.

## Dataset

The project uses the SMS Spam Collection dataset, which contains labeled SMS messages classified as HAM or SPAM.

### Dataset Statistics

- Original messages: 5,572
- HAM messages: 4,825
- SPAM messages: 747
- Duplicate records removed: 403
- Final dataset size: 5,169 messages
- Final HAM messages: 4,516
- Final SPAM messages: 653

The dataset was cleaned by removing duplicate records before model training.

## Technologies Used

- Python
- Pandas
- NumPy
- Regular Expressions (Regex)
- Scikit-learn
- TF-IDF Vectorization
- Logistic Regression
- Multinomial Naive Bayes
- Matplotlib
- Kaggle / Jupyter Notebook 

## Methodology

The project follows a sequence of steps to classify SMS messages:

1. **Data Loading** — Load the SMS Spam Collection dataset.
2. **Data Cleaning** — Remove unnecessary columns and inspect the dataset for missing values.
3. **Duplicate Removal** — Remove duplicate SMS records.
4. **Text Preprocessing** — Convert messages to lowercase and remove unnecessary characters.
5. **Train-Test Split** — Split the dataset into training and testing sets using stratified sampling.
6. **TF-IDF Vectorization** — Convert cleaned SMS text into numerical TF-IDF features.
7. **Model Training** — Train Naive Bayes and Logistic Regression models.
8. **Model Evaluation** — Compare models using accuracy, precision, recall, and F1-score.
9. **Experimentation** — Evaluate bigrams, decision thresholds, and Logistic Regression hyperparameters.
10. **Final Prediction** — Use the selected model to classify new SMS messages as HAM or SPAM.

## Machine Learning Models

Two primary machine learning algorithms were evaluated for SMS classification:

### 1. Multinomial Naive Bayes

Multinomial Naive Bayes was used as a baseline classification model for text data.

### 2. Logistic Regression

Logistic Regression was trained using TF-IDF features and provided better overall performance than the Naive Bayes model.

The Logistic Regression model was further evaluated through threshold analysis and hyperparameter tuning.

## Experiments

Several experiments were conducted to evaluate different approaches and improve the SMS classification system.

### Experiment 1 — Naive Bayes

Multinomial Naive Bayes was trained as a baseline model using TF-IDF features.

### Experiment 2 — Logistic Regression

Logistic Regression was evaluated as an alternative to Naive Bayes and achieved better overall performance.

### Experiment 3 — Unigrams and Bigrams

TF-IDF was tested using both unigrams and bigrams. The bigram configuration did not improve the results, so the unigram configuration was retained.

### Experiment 4 — Decision Threshold Analysis

Different classification thresholds were evaluated to study the trade-off between precision and recall.

A threshold of 0.30 produced higher recall and F1-score, but also increased false positives. Therefore, the standard 0.50 threshold was retained for the final model.

### Experiment 5 — Hyperparameter Tuning

The Logistic Regression regularization parameter `C` was changed from 1.0 to 2.0.

The tuned model improved the overall balance of precision, recall, and F1-score and was selected as the final model.

## Final Model

The final model uses TF-IDF unigrams with Logistic Regression.

### Final Configuration

- Feature Extraction: TF-IDF
- N-gram Range: Unigrams
- Algorithm: Logistic Regression
- Regularization Parameter (C): 2.0
- Classification Threshold: 0.50

The model was selected because it provides a strong balance between precision, recall, and F1-score while maintaining only one false positive on the test set.

## Results

The final selected model is:

- Model: Logistic Regression
- TF-IDF Representation: Unigrams
- Hyperparameter: C = 2.0
- Decision Threshold: 0.50

The final model achieved the following results on the test set:

Metric| Score
Accuracy| 97.29%
Precision| 99.05%
Recall| 79.39%
F1-Score| 88.14%

The model produced 1 false positive and 27 false negatives on the test set.

## Confusion Matrix

| Predicted HAM| Predicted SPAM
Actual HAM| 902| 1
Actual SPAM| 27| 104

Confusion Matrix Values:

- True Negatives (TN): 902
- False Positives (FP): 1
- False Negatives (FN): 27
- True Positives (TP): 104

## Error Analysis

The baseline Logistic Regression model produced 39 incorrect predictions on the test set.

- False Negatives: 38
- False Positives: 1

### Common Error Patterns

The incorrect predictions included messages with:

- Abbreviations or unusual spelling
- Phone numbers or numeric codes
- Promotional wording
- Conversational-looking spam
- Short or ambiguous messages
- Variations of common spam vocabulary

One false positive was the legitimate message:

> "Waiting for your call"

This message was classified as SPAM even though it was actually HAM.

Error analysis helped identify cases where the model may have difficulty distinguishing legitimate conversational messages from spam-like wording.

## Example Predictions

The final model was tested with new SMS messages to verify its prediction behavior.

| Message | Prediction | Probability |
|---|---|---:|
| Hey, are you coming to college tomorrow? | HAM | 97.00% |
| Congratulations! You have won a free prize! Call now! | SPAM | 96.33% |
| Can you send me the notes when you get home? | HAM | 98.87% |
| URGENT! Claim your special reward today before it expires! | SPAM | 58.52% |
| What time should we meet at the bus stop? | HAM | 95.18% |

All five test messages were correctly classified by the final model.

## Project Architecture

The system follows this pipeline:

   text
SMS Message
     ↓
Input Validation
     ↓
Text Cleaning
     ↓
TF-IDF Vectorization
     ↓
8,164 Numerical Features
     ↓
Logistic Regression
     ↓
HAM / SPAM Prediction
     ↓
Predicted Probability

## Installation

### 1. Clone the repository

bash
git clone https://github.com/Sivakumars08/TextSense-AI.git
cd TextSense-AI

### 2. Install the required libraries
pip install pandas numpy scikit-learn matplotlib jupyter

### 3. Open the notebook
Open the project notebook using Jupyter Notebook or JupyterLab.

## Usage

The project can be run through the Jupyter Notebook.

1. Open the `TextSense_AI.ipynb` notebook.
2. Run the cells from top to bottom.
3. The notebook loads and preprocesses the SMS dataset.
4. The model is trained and evaluated.
5. Use the `predict_message()` function to classify a new SMS message.

Example:

predict_message("Congratulations! You have won a free prize!")

## 📓 Notebook

[View the complete TextSense AI notebook](Notebook/TextSense_AI.ipynb)

## Future Improvements

The current project focuses on SMS spam classification using traditional machine learning techniques. Future improvements could include:

- Expanding the dataset with more recent SMS messages.
- Improving text preprocessing for abbreviations, slang, and unusual spelling.
- Exploring additional machine learning algorithms.
- Testing more advanced NLP techniques.
- Developing a simple user interface for real-time message classification.
- Deploying the model as a web application or API.

## Author

Sivakumar S







