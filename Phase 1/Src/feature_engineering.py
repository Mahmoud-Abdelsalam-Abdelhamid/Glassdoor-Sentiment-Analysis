import numpy as np
import gensim.downloader as api
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ------------------------------------------------------
# Compining text for better sentiment
# ------------------------------------------------------

def combine_sentiment(df):
    """
    Combine text from 'headline', 'pros', and 'cons' columns 
    into a single 'sentiment' column.
    """
    df['text'] = (
        df['headline'].fillna('') + "\n" +
        df['pros'].fillna('') + "\n" +
        df['cons'].fillna('')
    )
    return df


def combine_survey_result(df_text, df_survey):
    """
    Enrich 'sentiment' column by appending survey results 
    ('recommend_clean', 'ceo_approv_clean', 'outlook').
    """
    df_text['text'] = (
        df_text['text'] +
        "\nRecommended: " + df_survey['recommend_clean'].fillna('Unknown') +
        "\nCEO approval: " + df_survey['ceo_approv_clean'].fillna('Unknown') +
        "\nOutlook: " + df_survey['outlook_clean'].fillna('Unknown')
    )
    return df_text



# ------------------------------------------------------
#  Glove Embeddings
# ------------------------------------------------------

def load_glove_embeddings(glove_path, embedding_dim):
    embeddings_index = {}
    with open(glove_path, encoding="utf8") as f:
        for line_num, line in enumerate(f, 1):
            values = line.strip().split()
            
            # skip empty or malformed lines
            if len(values) < embedding_dim + 1:
                print(f"Skipping malformed line {line_num}: {line[:50]}...")
                continue

            word = values[0]
            try:
                vector = np.asarray(values[1:], dtype="float32")
            except ValueError:
                print(f"Skipping bad vector at line {line_num}: {word}")
                continue

            if len(vector) == embedding_dim:
                embeddings_index[word] = vector

    print(f"Loaded {len(embeddings_index)} word vectors.")
    return embeddings_index


def build_embedding_matrix(tokenizer, embeddings_index, vocab_size, embedding_dim=100):
    """
    Create embedding matrix aligned with tokenizer word_index.
    """
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    for word, i in tokenizer.word_index.items():
        if i < vocab_size:
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
    return embedding_matrix