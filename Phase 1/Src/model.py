from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout

def build_sentiment_model(vocab_size, embedding_dim, embedding_matrix, max_sequence_length, lstm_units):
    """
    Builds and returns the Bidirectional LSTM model architecture.

    Args:
        vocab_size (int): The size of the vocabulary.
        embedding_dim (int): The dimension of the word embeddings.
        max_sequence_length (int): The maximum length of input sequences.
        lstm_units (int): The number of units in the LSTM layer.

    Returns:
        A compiled Keras Sequential model.
    """
    model = Sequential([
        Embedding(input_dim=vocab_size,
                output_dim=embedding_dim,
                weights=[embedding_matrix],
                input_length=max_sequence_length,
                trainable=False),
        Bidirectional(LSTM(lstm_units, return_sequences=False)),
        Dropout(0.5),
        
        Dense(32, activation="relu"),
        Dropout(0.3),
        
        Dense(32, activation="relu"),
        Dropout(0.3),
        
        Dense(1, activation="sigmoid")  
    ])

    model.compile(optimizer="adam",
                loss="binary_crossentropy",
                metrics=["accuracy"])
    
    print("Model architecture built successfully.")
    model.summary()
    return model
