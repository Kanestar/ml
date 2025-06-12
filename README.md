# Week 3: Machine Learning, Deep Learning, and NLP Projects

This folder contains a set of projects covering fundamental concepts in Classical Machine Learning, Deep Learning, and Natural Language Processing, culminating in a web application deployment example.

## Table of Contents

1.  [Getting Started (General Setup)](#getting-started-general-setup)
2.  [Task 1: Classical ML - Iris Species Classification](#task-1-classical-ml---iris-species-classification)
3.  [Task 2: Deep Learning - MNIST Handwritten Digits Classification](#task-2-deep-learning---mnist-handwritten-digits-classification)
4.  [Task 3: NLP - Amazon Product Reviews (NER & Sentiment)](#task-3-nlp---amazon-product-reviews-ner--sentiment)
5.  [Deployment: MNIST Digit Classifier Web App (Streamlit)](#deployment-mnist-digit-classifier-web-app-streamlit)

---

## 1. Getting Started (General Setup)

Before running any of the project scripts, please follow these general setup steps to ensure your environment is ready.

### 1.1 Python Installation

Ensure you have Python installed on your system. Python **3.7 to 3.10** is recommended for compatibility with all libraries, especially TensorFlow.

*   If you don't have Python, download it from [python.org](https://www.python.org/downloads/).
*   **Crucial:** During installation, make sure to check the box that says **"Add Python to PATH"**.

### 1.2 Install Project Dependencies

All necessary Python packages for these projects are listed in the `requirements.txt` file within this `week 3` folder.

1.  **Open your terminal or command prompt.**
2.  **Navigate to the `week 3` directory** where this `README.md` file and the project scripts are located:
    ```bash
    cd path/to/your/ai for sotware engineering/week 3
    ```
    (Replace `path/to/your/ai for sotware engineering` with the actual path to your main project directory).
3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: The installation of TensorFlow might take some time depending on your internet speed and system specifications.* 

---

## 2. Task 1: Classical ML - Iris Species Classification

**Goal:** To preprocess the Iris dataset, train a Decision Tree Classifier, and evaluate its performance using key metrics.

**Dataset:** Iris Species Dataset (a classic dataset, loaded directly from `scikit-learn`).

**File:** `iris_classification.py`

### How to Run:

1.  Ensure you have completed the [General Setup](#1-getting-started-general-setup).
2.  From your terminal, while inside the `week 3` directory, execute the script:
    ```bash
    python iris_classification.py
    ```

### Expected Output:

 The script will print messages to your console detailing the dataset loading, confirmation of no missing values, data splitting, and model training. Finally, it will display the model's performance metrics: Accuracy, Precision (weighted), Recall (weighted), and a comprehensive Classification Report that shows these metrics for each individual Iris species class.

---

## 3. Task 2: Deep Learning - MNIST Handwritten Digits Classification

**Goal:** To build and train a Convolutional Neural Network (CNN) to classify handwritten digits, aiming for high accuracy, and visualize its predictions.

**Dataset:** MNIST Handwritten Digits (loaded directly from `tensorflow.keras.datasets`).

**File:** `mnist_cnn.py`

### How to Run:

1.  Ensure you have completed the [General Setup](#1-getting-started-general-setup).
2.  From your terminal, while inside the `week 3` directory, execute the script:
    ```bash
    python mnist_cnn.py
    ```

### Expected Output:

 The script will first display information about the dataset loading and preprocessing. It will then show the CNN model summary (architecture details). During training, you'll see epoch-by-epoch progress including loss and accuracy. After training, the final test loss and accuracy will be printed. A separate Matplotlib window will then appear, displaying 5 random handwritten digit images from the test set with their true and predicted labels. **You must close this plot window to allow the script to finish execution.**

**Important Note:** This script saves the trained CNN model as `mnist_cnn_model.h5` in the `week 3` folder if the test accuracy achieved is greater than 95%. This saved model is essential for the Streamlit web application deployment in [Task 5](#5-deployment-mnist-digit-classifier-web-app-streamlit).

---

## 4. Task 3: NLP - Amazon Product Reviews (NER & Sentiment)

**Goal:** To perform Named Entity Recognition (NER) to extract product names and brands from text reviews, and to analyze their sentiment using a rule-based approach.

**Text Data:** Sample user reviews (defined directly within the script for demonstration purposes).

**File:** `nlp_spacy.py`

### How to Run:

1.  Ensure you have completed the [General Setup](#1-getting-started-general-setup).
2.  **Crucial: Download the spaCy English Language Model (One-Time Setup):**
    Before running `nlp_spacy.py` for the first time, you must download the necessary spaCy language model. Open your terminal or command prompt and run:
    ```bash
    python -m spacy download en_core_web_sm
    ```
    This command downloads a small English model trained on web data, which is required for NER.
3.  From your terminal, while inside the `week 3` directory, execute the script:
    ```bash
    python nlp_spacy.py
    ```

### Expected Output:

 The script will first confirm that the spaCy model has loaded. It will then print the results for each sample review, clearly showing the extracted entities (including potential brands and product names identified by heuristics), and the determined sentiment (Positive, Negative, or Neutral) based on the rule-based analysis.

---

## 5. Deployment: MNIST Digit Classifier Web App (Streamlit)

**Goal:** To deploy the trained MNIST Handwritten Digits Classifier as an interactive web application that can be accessed via a web browser.

**Live App Link:** [https://7bkqwtwovbiortken8uzuh.streamlit.app/](https://7bkqwtwovbiortken8uzuh.streamlit.app/)

**Prerequisites:**

*   You **must** have successfully run `mnist_cnn.py` (from [Task 2](#task-2-deep-learning---mnist-handwritten-digits-classification)) at least once.
*   This is important because `mnist_cnn.py` trains the model and saves it as `mnist_cnn_model.h5`, which the Streamlit app (`mnist_app.py`) loads to make predictions.

**Files Involved:**

*   `mnist_app.py`: The main Streamlit application code.
*   `mnist_cnn_model.h5`: The pre-trained neural network model (generated by `mnist_cnn.py`).

### How to Run Locally (for testing and development):

If you wish to run the Streamlit application on your local machine:

1.  Ensure you have completed the [General Setup](#1-getting-started-general-setup).
2.  From your terminal, while inside the  directory, execute the Streamlit command:
    ```bash
    streamlit run mnist_app.py
    ```
    Your default web browser should automatically open to a local address (e.g., `http://localhost:8501/`) displaying the MNIST Digit Classifier web application. You can then upload an image of a handwritten digit to get a prediction.

---

## Folder Structure

```
week 3/
├── iris_classification.py      # Classical ML Task
├── mnist_cnn.py                # Deep Learning Task
├── mnist_app.py                # Streamlit Web App for MNIST Classifier
├── nlp_spacy.py                # NLP Task
├── requirements.txt            # Python dependencies for all tasks
└── README.md                   # This documentation file
``` 