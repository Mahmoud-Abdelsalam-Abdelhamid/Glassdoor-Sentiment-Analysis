import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoModel

def build_sentiment_model(model_name, num_classes):
    """
    This function now returns an instance of our custom SentimentClassifier.
    """
    return SentimentClassifier(model_name, num_classes)

class SentimentClassifier(nn.Module):
    """
    A custom PyTorch model that wraps a pre-trained Transformer and adds
    a Dropout layer for regularization and a final classification layer.
    """
    def __init__(self, model_name, num_classes):
        super(SentimentClassifier, self).__init__()
        # Load the pre-trained BERT model
        self.bert = AutoModel.from_pretrained(model_name)
        
        # --- ADDED REGULARIZATION ---
        # A dropout layer to prevent overfitting. 0.3 means 30% of neurons
        # will be randomly zeroed out during training.
        self.dropout = nn.Dropout(p=0.3)
        
        # The final linear layer for classification. It takes the BERT hidden
        # state for the [CLS] token and maps it to the number of classes.
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Some models return pooler_output, others don't
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            pooled_output = outputs.pooler_output
        else:
            # fallback: use [CLS] token embedding
            pooled_output = outputs.last_hidden_state[:, 0, :]

        output_after_dropout = self.dropout(pooled_output)
        logits = self.classifier(output_after_dropout)

        return logits














# def build_sentiment_model(model_name, num_classes):
#     """
#     Builds a Transformer-based model for sequence classification using PyTorch.

#     Args:
#         model_name (str): The name of the pre-trained model from Hugging Face.
#         num_classes (int): The number of output classes.

#     Returns:
#         A PyTorch model.
#     """
#     # Load the pre-trained model from Hugging Face, configured for our number of classes.
#     # This returns a PyTorch `nn.Module`.
#     model = AutoModelForSequenceClassification.from_pretrained(
#         model_name,
#         num_labels=num_classes
#     )
    
#     print("PyTorch Transformer model loaded successfully.")
#     return model