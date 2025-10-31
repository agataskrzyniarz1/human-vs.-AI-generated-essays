# Human vs AI Essay Classification

This project demonstrates a text classification pipeline that distinguishes between human and AI-generated essays using machine learning models (Logistic Regression and Linear SVM).
It uses a dataset from Kaggle: [Human vs AI Generated Essays](https://www.kaggle.com/datasets/navjotkaushal/human-vs-ai-generated-essays).

## Dataset

The dataset `balanced_ai_human_prompts.csv` contains essays labeled as either `human` or `AI-generated` in the `generated` column, along with the essay text in the `text` column.

## Installation

1. Clone the repository
    ```bash
   git clone git@github.com:agataskrzyniarz1/human-vs.-AI-generated-essays.git
    ```
2. Navigate to the project directory
    ```bash
    cd human-vs.-AI-generated-essays
    ```
3. Install dependencies
    ```bash
    pip install -r requirements.txt
    ```

## Pipeline Overview

### Data Preprocessing
- Convert text to lowercase
- Remove URLs
- Remove non-alphabetic characters
- Remove extra spaces


### Feature Extraction
- TF-IDF vectorization of words and bigrams
- Maximum of 50,000 features
- Minimum word length of 2

### Model Training
- Logistic Regression: sklearn.linear_model.LogisticRegression
- Linear SVM: sklearn.svm.LinearSVC

Training and evaluation are performed on an 80/20 train/test split.

### Evaluation

The models were evaluated on the test set and using 5-fold cross-validation.

#### Logistic Regression

Accuracy (train): 0.997

Accuracy (test): 0.995

5-fold CV mean accuracy: 0.974

#### Linear SVM

Accuracy (train): 1.0

Accuracy (test): 0.998

5-fold CV mean accuracy: 0.968

### Top Features

The script identifies the top 20 words/phrases indicative of AI-generated essays and human-written essays using the SVM model coefficients. This helps understand which patterns are most discriminative.

### User Input Prediction

Users can input their own essay text for classification. The model returns:
- Predicted class: human or AI-generated
- Confidence score
