# credit-card-fraud-detection
A machine learning project to detect fraudulent credit card transactions. This repository contains the code for data processing, model training (e.g., Logistic Regression, Random Forest), and evaluation.

# Credit Card Fraud Detection Project

### Internship Project | SPARKIIT powered by Wipro DICE ID

---

## 1. Objective

The primary goal of this project is to develop a high-performance machine learning model capable of accurately identifying fraudulent credit card transactions in near real-time. [cite_start]The model leverages pattern recognition and anomaly detection techniques on a highly imbalanced dataset to minimize financial losses for banking institutions by effectively flagging suspicious activities. [cite: 5, 6]

---

## 2. Tech Stack & Environment

- [cite_start]**Programming Language:** Python [cite: 8]
- **Libraries & Frameworks:**
    - [cite_start]**Data Manipulation:** Pandas, NumPy [cite: 10]
    - [cite_start]**Visualization:** Matplotlib, Seaborn [cite: 13]
    - [cite_start]**Machine Learning:** Scikit-learn (Logistic Regression, Random Forest), XGBoost [cite: 12]
    - [cite_start]**Imbalanced Data Handling:** Imbalanced-learn (SMOTE) [cite: 14]
- [cite_start]**Environment:** Jupyter Notebook / Google Colab [cite: 15]

---

## 3. Project Workflow

The project was structured into a series of logical steps, moving from data exploration to final model optimization.

### 🔹 Step 1: Exploratory Data Analysis (EDA)
[cite_start]The initial phase involved a deep dive into the Kaggle credit card fraud dataset. [cite: 20] Key activities included:
- Analyzing the severe class imbalance between fraudulent (Class 1) and non-fraudulent (Class 0) transactions.
- [cite_start]Visualizing transaction patterns and feature distributions using histograms and heatmaps. [cite: 22, 27]
- Studying the statistical differences in transaction `Amount` and `Time` between the two classes.

### 🔹 Step 2: Data Preprocessing
Before modeling, the data was prepared for machine learning algorithms:
- **Feature Scaling:** The `Time` and `Amount` columns were normalized using `StandardScaler` from Scikit-learn to ensure they had a comparable scale to the other PCA-transformed features.
- **Data Splitting:** The dataset was split into training (80%) and testing (20%) sets, using stratification to maintain the original class distribution in both splits.

### 🔹 Step 3: Handling Class Imbalance
The core challenge of this dataset is its severe imbalance. [cite_start]To address this, **SMOTE (Synthetic Minority Over-sampling Technique)** was applied *only* to the training data. [cite: 21] This process generates synthetic samples for the minority (fraud) class, creating a balanced dataset for the models to learn from without introducing data leakage from the test set.

### 🔹 Step 4: Baseline Model Training
[cite_start]Three different classification models were trained on the balanced (resampled) training data to establish a performance baseline: [cite: 24]
1.  Logistic Regression
2.  Random Forest Classifier
3.  XGBoost Classifier

### 🔹 Step 5: Hyperparameter Tuning
The best-performing baseline model (XGBoost) was selected for optimization. [cite_start]**`RandomizedSearchCV`** was used to efficiently search through a predefined grid of hyperparameters (like `n_estimators`, `max_depth`, `learning_rate`) to find the combination that yielded the best performance, measured by the ROC AUC score. [cite: 26]

---

## 4. Results & Evaluation

The models were rigorously evaluated on the original, untouched test set. The primary metrics focused on were **Recall** and **Precision** for the minority class (fraud), as correctly identifying fraud is the main objective.

- **Baseline Models:** All models performed reasonably well, with Random Forest and XGBoost showing superior performance, particularly in identifying fraudulent transactions.
- **Optimized Model:** After hyperparameter tuning, the final XGBoost model showed a significant improvement.

**Final Tuned XGBoost Model Performance:**
*(Note: Please replace these placeholders with your actual results from notebook 03)*
- **Precision (Class 1):** `0.XX`
- **Recall (Class 1):** `0.XX`
- **F1-Score (Class 1):** `0.XX`
- **ROC AUC Score:** `0.XXXX`

[cite_start]The outcome is a robust fraud detection model with high recall, ensuring that a vast majority of fraudulent transactions are caught, thereby minimizing false negatives and potential financial loss. [cite: 29, 30]

---

## 5. How to Run This Project

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/credit-card-fraud-detection.git](https://github.com/your-username/credit-card-fraud-detection.git)
    cd credit-card-fraud-detection
    ```

2.  **Create a virtual environment and install dependencies:**
    ```bash
    python -m venv venv
    venv\Scripts\activate  # On Windows
    pip install -r requirements.txt
    ```
    *(Note: You will need to create a `requirements.txt` file by running `pip freeze > requirements.txt` in your terminal)*

3.  **Launch Jupyter Notebook:**
    ```bash
    jupyter notebook
    ```

4.  **Run the notebooks:**
    - Open the `notebooks` directory.
    - Run the notebooks in sequential order:
      1.  `01_EDA_and_Preprocessing.ipynb`
      2.  `02_Model_Training_and_Evaluation.ipynb`
      3.  `03_Hyperparameter_Tuning.ipynb`
