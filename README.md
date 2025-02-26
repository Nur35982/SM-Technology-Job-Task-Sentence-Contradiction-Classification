# Sentence Contradiction Classification

## Project Overview
This project focuses on classifying pairs of sentences into one of three categories: **Contradiction**, **Neutral**, or **Entailment**. The goal is to develop machine learning models that can accurately predict the semantic relationship between two sentences. Four models were implemented and evaluated:
1. **Random Forest (RF)**
2. **Artificial Neural Network (ANN)**
3. **Long Short-Term Memory (LSTM)**
4. **XLM-RoBERTa (XLM-R)** (under development)

The project includes exploratory data analysis (EDA), text preprocessing, model training, hyperparameter tuning, and evaluation.

---

## Dataset Description
The dataset contains sentence pairs labeled with their semantic relationship. Each pair consists of:
- **Premise**: The first sentence.
- **Hypothesis**: The second sentence.
- **Label**: The relationship between the sentences:
  - `0`: Contradiction (sentences have opposite meanings).
  - `1`: Neutral (sentences are related but do not imply each other).
  - `2`: Entailment (one sentence logically follows from the other).

The dataset includes sentences in **15 different languages**, with the following distribution:
- English: 6870
- Chinese: 411
- Arabic: 401
- French: 390
- Swahili: 385
- Urdu: 381
- Vietnamese: 379
- Russian: 376
- Hindi: 374
- Greek: 372
- Thai: 371
- Spanish: 366
- Turkish: 351
- German: 351
- Bulgarian: 342

---

## Model Implementation Details

### 1. Random Forest (RF)
- **Preprocessing**: Text data was preprocessed using tokenization, lowercasing, stopword removal, and lemmatization. TF-IDF was used for feature extraction.
- **Hyperparameter Tuning**: RandomizedSearchCV was used to optimize hyperparameters:
  - `n_estimators`: 50 to 200
  - `max_depth`: None, 10, 20, 30
  - `min_samples_split`: 2 to 10
- **Best Parameters**: `max_depth=10`, `min_samples_split=3`, `n_estimators=131`

### 2. Artificial Neural Network (ANN)
- **Preprocessing**: Text data was preprocessed similarly to RF. TF-IDF was used for feature extraction.
- **Architecture**: A feedforward neural network with:
  - Input layer: 512 units
  - Hidden layers: 256 and 128 units
  - Output layer: 3 units (softmax activation)
- **Hyperparameter Tuning**: Optuna was used to optimize hyperparameters, achieving a validation accuracy of **0.333**.

### 3. Long Short-Term Memory (LSTM)
- **Preprocessing**: Text data was tokenized and padded to a fixed length.
- **Architecture**: A bidirectional LSTM with:
  - Embedding layer: 128 dimensions
  - LSTM layers: 64 and 32 units
  - Output layer: 3 units (softmax activation)
- **Performance**: Achieved a test accuracy of **0.322**.

### 4. XLM-RoBERTa (XLM-R)
- **Preprocessing**: Text data was tokenized using the XLM-R tokenizer.
- **Status**: Under development. Preliminary results show promise, but further tuning is required.

---

## Steps to Run the Code

### Prerequisites
- Python 3.x
- Required libraries: `pandas`, `numpy`, `scikit-learn`, `tensorflow`, `transformers`, `nltk`, `matplotlib`, `seaborn`

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/sentence-contradiction-classification.git
   cd sentence-contradiction-classification
