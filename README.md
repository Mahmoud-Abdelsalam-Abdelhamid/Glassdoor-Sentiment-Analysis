# 🧠 Glassdoor Sentiment Analysis  

This project analyzes **employee reviews from Glassdoor** to classify sentiment (Positive / Neutral / Negative) and extract insights about workplace culture, management, and job satisfaction.  
It demonstrates a **complete NLP workflow** — from raw text preprocessing to model training and Streamlit deployment — using both **classical and transformer-based deep learning architectures**.  

---

## 🚀 Project Overview  

The goal is to build an **intelligent system** that can automatically determine employee sentiment from textual reviews.  
The project is developed in **two phases** to showcase the transition from traditional NLP to **modern transformer models**.  

### 🔹 Phase 1 — LSTM + GloVe  
- Built an **LSTM model** using pre-trained **GloVe embeddings**.  
- Achieved **87% validation accuracy** on cleaned review data.  
- Implemented tokenization, padding, and dropout to stabilize training.  

### 🔹 Phase 2 — Fine-Tuned BERT  
- Leveraged **Hugging Face Transformers** with **PyTorch** for fine-tuning BERT.  
- Achieved **90% validation accuracy**, improving performance by **7%**.  
- Added **Dropout layers** and applied **hyperparameter tuning** to mitigate overfitting.  
- Deployed the model with **Streamlit** for real-time sentiment prediction.  

---

## ⚙️ Key Features  

✅ End-to-end NLP pipeline: Cleaning, preprocessing, embedding, model training & deployment.  
✅ Feature engineering: Compared **TF-IDF**, **Word2Vec (GloVe)**, and **BERT embeddings**.  
✅ Interactive web app: Streamlit-based real-time sentiment prediction.  
✅ Power BI dashboard: 3 dashboards (General, Jobs, Firms) for data-driven insights.  
✅ Modular architecture: Reusable scripts for preprocessing, feature engineering, and modeling.  

---

## 🧩 Tech Stack  

**Languages & Frameworks:**  
Python, PyTorch, TensorFlow, Hugging Face Transformers  

**Libraries:**  
Scikit-learn, Pandas, NumPy, NLTK, Gensim, Matplotlib, Seaborn  

**Visualization & Deployment:**  
Power BI, Streamlit  

---

## 📊 Model Performance  

| Model | Embedding | Accuracy | Framework |
|:------|:-----------|:----------|:------------|
| LSTM | GloVe | 87% | TensorFlow |
| BERT | Transformer (Fine-tuned) | 90% | PyTorch |

---

## 📈 Dashboards  

Developed **interactive Power BI dashboards** to visualize:  
- Sentiment distribution across companies and job titles.  
- Trends in management, culture, and compensation sentiment.  
- Comparison between firms and job roles.  

📊 File: `Dashboards/Glassdoor Dashboard.pbix`

---

## 📁 Project Structure  
```bash
Glassdoor-Sentiment-Analysis/
├── 📄 README.md
├── 📊 Dashboards/
│   └── Glassdoor Dashboard.pbix
├── 📁 Data/
│   ├── Analysis_data_.csv
│   ├── glassdoor_reviews.csv
│   └── Sentiment_data_2.csv
├── 🖼️ Images/
│   └── (Dashboard visuals, architecture diagrams, etc.)
├── 🧩 Phase 1/                # LSTM + GloVe (TensorFlow)
│   ├── 📂 Glove/
│   ├── 📂 Models/
│   ├── 📓 Notebooks/
│   └── ⚙️ Src/
├── 🤖 Phase 2/                # Fine-tuned BERT (PyTorch + Hugging Face)
│   ├── 📂 Models/
│   ├── 📓 Notebooks/
│   └── ⚙️ Src/
└── ⚠️ Note: Large files (datasets, models) are hosted externally.
```
## Download Pre-trained GloVe Embeddings

Download glove.6B.zip from the official GloVe website

Create a folder named Glove/.

Unzip the file and place glove.6B.100d.txt inside

## Download the Dataset

Download glassdoor job reviews.csv from Kaggle

## 💡 Insights & Learnings

- Combined NLP preprocessing with deep learning architectures.

- Observed the strong generalization power of BERT over LSTM.

- Enhanced model interpretability through sentiment visualization dashboards.

- Built a modular, production-ready pipeline for scalable NLP tasks.

## 🌐 Deployment

- Deployed via Streamlit, providing:

- Real-time sentiment classification.

- Visual analytics for review insights.

- Interactive interface for stakeholder demonstration.
