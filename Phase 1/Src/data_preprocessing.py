import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# ------------------------------------------------------
# Data Splitting "Current column"
# ------------------------------------------------------

def parse_job_status(df, column):

    pattern = re.compile(
        r'^(?P<current_state>Current|Former)?\s*'
        r'(?P<emp_state>Employee|Temporary Employee|Contractor|Intern|Freelancer)?'
        r'(?:,\s*(?P<emp_working_years>less than 1 year|more than \d+ years?|more than \d+ year|\d+ years?|\d+ year))?$'
    )

    df_extracted = df[column].str.extract(pattern)
    df_extracted = df_extracted.fillna("Unknown")

    df = pd.concat([df, df_extracted], axis=1)

    return df



# ------------------------------------------------------
# Data mapping and preprocessing
# ------------------------------------------------------

def encode_label(df, label):
    le = LabelEncoder()
    return le.fit_transform(df[label])

# Mapping Survey Columns

def map_overall_rating(val: int) -> str:
    # if 1 <= val < 3:
    #     return 'Negative'
    
    if val < 3:
        return 'Negative'
    
    else:
        return 'Positive'


def map_recommend(val: str) -> str:
    rec_map = {
        "v": "Yes",
        "o": "Neutral",
        "x": "No"
    }
    return rec_map.get(val, "Unknown")


def map_ceo_approval(val: str) -> str:
    mapping = {
        "v": "Approve",
        "o": "Disapprove",
        "r": "No Opinion",
        "x": "Other"
    }
    return mapping.get(val, "Unknown")


def map_outlook(val: str) -> str:
    mapping = {
        "v": "Positive",
        "o": "Neutral",
        "r": "Negative",
        "x": "Other"
    }
    return mapping.get(val, "Unknown")


def preprocess_survey_columns(df):
    """Map numeric overall rating into categorical sentiment."""
    
    df["outlook_clean"] = df["outlook"].apply(map_outlook)
    df["recommend_clean"] = df["recommend"].apply(map_recommend)
    df["ceo_approv_clean"] = df["ceo_approv"].apply(map_ceo_approval)
    df["sentiment"] = df["overall_rating"].apply(map_overall_rating)
    return df


# ------------------------------------------------------
# NLP Functionalities
# ------------------------------------------------------

def clean_text(text_list):
    """Basic text cleaning: lowercase, remove special chars."""
    
    cleaned_text_list = []
    
    for text in text_list:
        if pd.isna(text):
            cleaned_text_list.append("")
            continue
        
        text = text.lower()
        text = re.sub(r"http\S+|www\S+|https\S+", '', text)
        text = re.sub(r"[^a-zA-Z0-9\s!?,']", "", text)
        
        cleaned_text_list.append(text.strip())
    
    return cleaned_text_list


def tokenize_text(text_list, vocab_size = 30000):
    """Fit a Keras Tokenizer on a list of texts."""
    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(text_list)
    return tokenizer


def embedding_sequence(X, tokenizer, max_len = 200):
    """Convert text data into padded integer sequences."""
    X_seq = tokenizer.texts_to_sequences(X)
    X_pad = pad_sequences(X_seq, maxlen=max_len, padding="post", truncating="post")
    return X_pad


def preprocess(X_train, X_test):
    """Clean, tokenize, and convert raw text into padded integer sequences 
    for training and testing."""
    X_train_cleaned = clean_text(X_train)
    X_test_cleaned = clean_text(X_test)
    
    tokenizer = tokenize_text(X_train_cleaned)
    
    X_train_embedding = embedding_sequence(X_train_cleaned, tokenizer)
    X_test_embedding = embedding_sequence(X_test_cleaned, tokenizer)
    
    return X_train_embedding, X_test_embedding, tokenizer








# def lemmatize_tokens(tokens):
#     """Lemmatize tokens and remove stopwords."""
#     lemmatized = [
#         lemmatizer.lemmatize(token)
#         for token in tokens
#         if token not in stop_words
#     ]
#     return lemmatized


# def tokens_to_sentence(tokens):
#     """Return the tokens to one sentence"""
#     return " ".join(tokens).strip()