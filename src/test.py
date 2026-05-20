from pathlib import Path
import tempfile

import numpy as np
import mlflow
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)



# This is the testing accuracy for all the models trained in the train.py file. This file is used to test the models on the test set and compute the metrics.


def ComputeMetrics(y_val, y_pred):
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_val, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_val, y_pred, average="weighted", zero_division=0)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")

    metrics_dict = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

    if(accuracy < 0.7):
        print("Warning: One or more accuracy metrics are below 0.7, indicating potential issues with the model's performance.")
        exit(1)

    # Log metrics to MLflow and save a metrics bar chart artifact
    if mlflow.active_run() is None:
        mlflow.start_run()
        started_run = True
    else:
        started_run = False

    try:
        mlflow.log_metrics(metrics_dict)

        labels = list(metrics_dict.keys())
        values = list(metrics_dict.values())

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(labels, values, color=["#4c72b0", "#55a868", "#c44e52", "#8172b3"])
        ax.set_ylim(0, 1)
        ax.set_title("Training Performance Metrics")
        ax.set_ylabel("Score")
        
        for i, v in enumerate(values):
            ax.text(i, v + 0.02, f"{v:.3f}", ha="center", va="bottom")
        fig.tight_layout()

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            chart_path = tmp.name
        fig.savefig(chart_path)
        plt.close(fig)

        mlflow.log_artifact(chart_path, artifact_path="metrics")
    finally:
        if started_run:
            mlflow.end_run()


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "dataset" / "dataset_preprocessed.npz"
MODEL_DIR = BASE_DIR / "models"

print("This is to compute the testing accuracy for all the models trained in the train.py file. This file is used to test the models on the test set and compute the metrics.")

with np.load(DATA_PATH, allow_pickle=True) as f:
    x_test = f["x_test"]
    y_test = f["y_test"]

x_test = pad_sequences(x_test, maxlen=200)


class TransformerBlock(keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
        self.att = keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [keras.layers.Dense(ff_dim, activation="relu"), keras.layers.Dense(embed_dim),]
        )
        self.layernorm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = keras.layers.Dropout(rate)
        self.dropout2 = keras.layers.Dropout(rate)

    def call(self, inputs):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "rate": self.rate,
        })
        return config


class TokenAndPositionEmbedding(keras.layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.maxlen = maxlen
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.token_emb = keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = keras.layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = keras.ops.shape(x)[-1]
        positions = keras.ops.arange(start=0, stop=maxlen, step=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

    def get_config(self):
        config = super().get_config()
        config.update({
            "maxlen": self.maxlen,
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
        })
        return config


custom_objects = {
    "TransformerBlock": TransformerBlock,
    "TokenAndPositionEmbedding": TokenAndPositionEmbedding,
}


# compute metrics for the 5 layer model

transformer_classifier_5_layers = keras.models.load_model(MODEL_DIR / "transformer_classifier_5_layers.h5", custom_objects=custom_objects)
y_pred_transformer_5_layers = transformer_classifier_5_layers.predict(x_test)
y_pred_transformer_5_layers = np.argmax(y_pred_transformer_5_layers, axis=1)
print("Metrics for Transformer Classifier with 5 layers:")
ComputeMetrics(y_test, y_pred_transformer_5_layers)

# compute metrics for the 3 layer model

transformer_classifier_3_layers = keras.models.load_model(MODEL_DIR / "transformer_classifier_3_layers.h5", custom_objects=custom_objects)
y_pred_transformer_3_layers = transformer_classifier_3_layers.predict(x_test)
y_pred_transformer_3_layers = np.argmax(y_pred_transformer_3_layers, axis=1)
print("Metrics for Transformer Classifier with 3 layers:")
ComputeMetrics(y_test, y_pred_transformer_3_layers)

# compute metrics for the 7 layer model

transformer_classifier_7_layers = keras.models.load_model(MODEL_DIR / "transformer_classifier_7_layers.h5", custom_objects=custom_objects)
y_pred_transformer_7_layers = transformer_classifier_7_layers.predict(x_test)
y_pred_transformer_7_layers = np.argmax(y_pred_transformer_7_layers, axis=1)
print("Metrics for Transformer Classifier with 7 layers:")
ComputeMetrics(y_test, y_pred_transformer_7_layers)
