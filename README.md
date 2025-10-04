# 📧 Spam Detection using Machine Learning
This project implements a **Spam Message Classifier** that predicts whether a message is **Spam** or **Ham (Not Spam)** using NLP techniques and various Machine Learning algorithms.
---
## 🚀 Features
- Preprocessing of SMS text messages (stopword removal, punctuation removal, etc.)
- Data visualization with **WordClouds**
- Feature extraction using **TF-IDF Vectorizer**
- Model training with multiple classifiers:
  - Naive Bayes
  - Logistic Regression
  - Support Vector Machine
  - Decision Tree
  - Random Forest
  - K-Nearest Neighbors
  - XGBoost
- Achieved accuracy of **~98%** with **Naive Bayes**
- Model saved using **Joblib** for future predictions
---
## 📂 Dataset
The project uses the **SMS Spam Collection Dataset** (`spam.csv`), which contains:
- **Ham messages:** 4825  
- **Spam messages:** 747  
- Total: **5572 messages**
---
## ⚙️ Installation
Clone this repository:
```bash
git clone https://github.com/your-username/SPAM-Detection.git
cd SPAM-Detection

V# 📧 Spam Detection using Machine Learning

This project implements a **Spam Message Classifier** that predicts whether a message is **Spam** or **Ham (Not Spam)** using NLP techniques and various Machine Learning algorithms.

---

## 🚀 Features
- Preprocessing of SMS text messages (stopword removal, punctuation removal, etc.)
- Data visualization with **WordClouds**
- Feature extraction using **TF-IDF Vectorizer**
- Model training with multiple classifiers:
  - Naive Bayes
  - Logistic Regression
  - Support Vector Machine
  - Decision Tree
  - Random Forest
  - K-Nearest Neighbors
  - XGBoost
- Achieved accuracy of **~98%** with **Naive Bayes**
- Model saved using **Joblib** for future predictions

---



## 📂 Dataset
The project uses the **SMS Spam Collection Dataset** (`spam.csv`), which contains:
- **Ham messages:** 4825  
- **Spam messages:** 747  
- Total: **5572 messages**

---

## ⚙️ Installation

Clone this repository:

```bash
git clone https://github.com/your-username/SPAM-Detection.git
cd SPAM-Detection

🧑‍💻 Usage
Run the Jupyter Notebook
jupyter notebook


Open spam_detection.ipynb and run all cells.
Predict New Messages
You can use the trained model best.pkl to predict:
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
# Load model
best = joblib.load('best.pkl')
vectorizer = joblib.load('vectorizer.pkl')  # Save & load TF-IDF vectorizer too
# Example predictions
texts = [
    "Free entry in a contest to win cash prizes",
    "Hey, are we still meeting for lunch?",
    "WINNER!! You just won a free trip to Bahamas. Send details."
]
features = vectorizer.transform(texts)
preds = best.predict(features)
for text, label in zip(texts, preds):
    print(f"{text} -> {'SPAM' if label==1 else 'HAM'}")

📊 Results
| Classifier                | Accuracy  |
| ------------------------- | --------- |
| Naive Bayes               | **98.8%** |
| Random Forest             | 97.9%     |
| Support Vector Classifier | 97.8%     |
| Decision Tree             | 96.0%     |
| Logistic Regression       | 95.3%     |
| K-Nearest Neighbors       | 93.3%     |


📌 Requirements
Python 3.x
scikit-learn
pandas
numpy
matplotlib
nltk
xgboost
joblib
wordcloud

Install via:
pip install scikit-learn pandas numpy matplotlib nltk xgboost joblib wordcloud

📜 License
This project is licensed under the MIT License.
🤝 Contributing
Pull requests are welcome! For major changes, open an issue first to discuss what you would like to change.
⭐ Acknowledgements
Dataset: [UCI SMS Spam Collection Dataset](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)
---
👉 Do you want me to also generate a **`requirements.txt`** file for your repo so that
