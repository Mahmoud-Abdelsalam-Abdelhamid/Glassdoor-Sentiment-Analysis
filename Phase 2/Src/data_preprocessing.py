import os
import re
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder

def label_encoding(df, label):
    le = LabelEncoder()
    return le.fit_transform(df[label])


def clean_text(text):  
    """
    Cleans text: lowercase, remove non-alphabetic chars.
    The Transformer's tokenizer will handle stopwords and special tokens.
    """
    
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text, re.I|re.A)
    return text.strip()


def compine_sentiment_text(df):
    """
    Combine text from 'headline', 'pros', and 'cons' columns 
    into a single 'sentiment' column.
    """
    df['text'] = (
        df['headline'] + "\n" +
        df['pros'] + "\n" +
        df['cons']
    )
    return df['text']