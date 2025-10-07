🧠 Glassdoor Sentiment Analysis

This project analyzes employee reviews from Glassdoor to classify overall sentiment (Positive / Neutral / Negative) and uncover insights about workplace culture, management, and job satisfaction. It demonstrates an end-to-end Natural Language Processing (NLP) workflow — from raw text preprocessing to model deployment — using modern deep learning architectures.

🚀 Project Overview

The goal is to build an intelligent system that can automatically determine employee sentiment based on textual reviews.
The project was implemented in two phases to showcase the evolution from traditional NLP methods to advanced transformer-based models.

🔹 Phase 1 — LSTM + GloVe

Built an LSTM model using pre-trained GloVe embeddings.

Achieved 87% validation accuracy on cleaned review text.

🔹 Phase 2 — Fine-Tuned BERT

Leveraged Hugging Face Transformers and PyTorch to fine-tune a BERT model.

Achieved 90% validation accuracy, improving performance by 7%.

Added Dropout layers and applied hyperparameter tuning to reduce overfitting.

⚙️ Key Features

End-to-end NLP pipeline: Data cleaning, preprocessing, embedding, model training, and deployment.

Feature engineering: Compared TF-IDF, Word2Vec (GloVe), and BERT embeddings.

Interactive deployment: Real-time prediction web app built with Streamlit.

Dashboard insights: Sentiment trends by company, job title, and location.

Scalable architecture: Modular file structure with reusable scripts for preprocessing, modeling, and visualization.

🧩 Tech Stack

Languages & Libraries:

Python, PyTorch, TensorFlow, Scikit-learn, Hugging Face Transformers

Pandas, NumPy, Matplotlib, Seaborn, NLTK, Gensim

Streamlit for deployment

📊 Results
Model	Embedding	Accuracy	Framework
LSTM	GloVe	87%	TensorFlow
BERT	Transformer (Fine-tuned)	90%	PyTorch
📁 Project Structure
Glassdoor-Sentiment-Analysis/
│
├── 📄 README.md
│
├── 📊 Dashboards/
│   └── Glassdoor Dashboard.pbix
│
├── 📁 Data/
│   ├── Analysis_data_.csv
│   ├── glassdoor_reviews.csv
│   └── Sentiment_data_2.csv
│
├── 🖼️ Images/
│   └── (dashboard template, icons, images)
│
├── 🧩 Phase 1/                # LSTM + GloVe (TensorFlow)
│   ├── 📂 Glove/
│   │   └── wiki_giga_2024_100_MFT20_vectors_seed_2024_alpha_0.75_eta_0.05.050_combined.txt
│   │
│   ├── 📂 Models/
│   │   ├── sentiment_model_0.87.keras
│   │   └── tokenizer.pkl
│   │
│   ├── 📓 Notebooks/
│   │   ├── Base_line_models.ipynb
│   │   └── Data_Preparation.ipynb
│   │
│   └── ⚙️ Src/
│       ├── data_preprocessing.py
│       ├── evaluate.py
│       ├── feature_engineering.py
│       ├── model.py
│       ├── sentiment_utils.py
│       └── train.py
│
├── 🤖 Phase 2/                # Fine-tuned BERT (PyTorch + Hugging Face)
│   ├── 📂 Models/
│   │   ├── BERT model/
│   │   └── distilbert BERT model/
│   │
│   ├── 📓 Notebooks/
│   │   └── model_preparation.ipynb
│   │
│   └── ⚙️ Src/
│       ├── data_preprocessing.py
│       ├── glassdoor_dataset_class.py
│       ├── model.py
│       ├── Streamlit web app.py
│       ├── test.py
│       └── train.py

💡 Insights & Learnings

Learned to combine NLP preprocessing with deep learning architectures.

Observed how transfer learning (BERT) significantly improves generalization.

Enhanced model interpretability through sentiment visualization.

Built modular, production-ready ML pipelines.

🌐 Deployment

The project is deployed using Streamlit, providing:

Real-time text sentiment classification.

Visual analytics for review trends.

Interactive UI for exploring model predictions.