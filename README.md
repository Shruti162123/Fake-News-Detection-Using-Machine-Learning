```markdown
# Fake News Detection Using Machine Learning

A beginner-friendly yet research-ready Python project for detecting **Fake News** vs **Real News** using:

- **TF-IDF Vectorization**
- **Logistic Regression**
- **NLTK-based text preprocessing**
- **Automatic graph generation for research papers**

---

## Project Overview

This project builds a complete fake news classification pipeline from scratch using two CSV files:

- `True.csv` (real news)
- `Fake.csv` (fake news)

Each file contains these columns:

- `title`
- `text`
- `subject`
- `date`

The model:
1. Loads both datasets
2. Adds labels (`1 = Real`, `0 = Fake`)
3. Combines data
4. Cleans text (`title + text`)
5. Extracts features using TF-IDF
6. Trains a Logistic Regression classifier
7. Evaluates with accuracy and confusion matrix
8. Generates multiple visualizations
9. Allows prediction from user input (CLI)
10. Supports optional Streamlit UI

---

## Features

- Clean and simple code in Python
- Beginner-friendly preprocessing pipeline
- High accuracy baseline model
- Research-focused graph generation (10 plots)
- Optional Streamlit interface for quick testing
- Ready for GitHub + report/paper usage

---

## Project Structure

```bash
.
├── app.py                  # Main training + evaluation + prediction script
├── visualizations.py       # Generates and saves all research graphs
├── True.csv                # Real news dataset (add manually)
├── Fake.csv                # Fake news dataset (add manually)
├── graphs/                 # Auto-generated output plots
│   ├── 01_class_distribution.png
│   ├── 02_content_length_distribution.png
│   ├── 03_content_length_boxplot.png
│   ├── 04_confusion_matrix_counts.png
│   ├── 05_confusion_matrix_normalized.png
│   ├── 06_roc_curve.png
│   ├── 07_precision_recall_curve.png
│   ├── 08_probability_distribution.png
│   ├── 09_top_features_real.png
│   └── 10_top_features_fake.png
└── README.md
```

---

## Installation

### 1) Clone repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2) Create virtual environment (recommended)

```bash
python -m venv .venv
```

Activate:

- **Windows (PowerShell):**
  ```bash
  .\.venv\Scripts\Activate.ps1
  ```
- **Windows (CMD):**
  ```bash
  .\.venv\Scripts\activate.bat
  ```
- **macOS/Linux:**
  ```bash
  source .venv/bin/activate
  ```

### 3) Install dependencies

```bash
pip install pandas scikit-learn nltk matplotlib seaborn streamlit
```

---

## Dataset Setup (Kaggle Download)

Due to large file size, the dataset is **not included** in this repository.

Please download it from Kaggle:  
[Fake News Detection Datasets (Kaggle)](https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets)

### Steps

1. Open the Kaggle link above and download the dataset zip.
2. Extract the zip file.
3. Copy these files into the project root (same folder as `app.py`):
   - `True.csv`
   - `Fake.csv`

Your folder should look like:

```bash
.
├── app.py
├── visualizations.py
├── True.csv
├── Fake.csv
└── README.md

---

## How to Run

## Option 1: Run in CLI mode (default)

```bash
python app.py
```

What happens:
- Model trains
- Accuracy + confusion matrix are printed
- Graphs are saved in `graphs/`
- You can enter custom news text for prediction

Exit CLI prediction loop by typing:

```text
exit
```

## Option 2: Run Streamlit UI (optional)

```bash
streamlit run app.py -- --streamlit
```

You’ll get:
- Text area input
- **Check News** button
- Real/Fake prediction output

---

## Output Graphs Generated

After running `python app.py`, the project automatically saves:

1. Class distribution  
2. Content length distribution by class  
3. Content length boxplot  
4. Confusion matrix (counts)  
5. Confusion matrix (normalized)  
6. ROC curve  
7. Precision-Recall curve  
8. Predicted probability distribution  
9. Top features supporting Real class  
10. Top features supporting Fake class  

These are useful for research reports and IEEE papers.

---

## Model Details

- **Vectorizer:** `TfidfVectorizer(max_features=5000)`
- **Classifier:** `LogisticRegression(max_iter=1000)`
- **Split:** 80% train, 20% test (`random_state=42`)
- **Labels:** Real = `1`, Fake = `0`

---

## Example Console Output

```text
Loading data and training model...

Model Evaluation
------------------------------
Accuracy Score: 0.9919
Confusion Matrix:
[[4598   52]
 [  21 4309]]

Saved 10 graphs successfully in: graphs
```

---

## Requirements

You can also create a `requirements.txt` file with:

```txt
pandas
scikit-learn
nltk
matplotlib
seaborn
streamlit
```

Install with:

```bash
pip install -r requirements.txt
```

---

## Troubleshooting

### 1) `True.csv` / `Fake.csv` not found
Make sure both files are in the same folder as `app.py`.

### 2) NLTK stopwords issue
The script already downloads stopwords automatically.  
If needed, run manually:

```python
import nltk
nltk.download("stopwords")
```

### 3) Streamlit import error
Install streamlit:

```bash
pip install streamlit
```

---

## Future Improvements

- Compare with SVM / Random Forest / XGBoost
- Add deep learning models (LSTM, BERT)
- Add model saving/loading (`joblib`)
- Build REST API for deployment
- Add multilingual fake news support

---

