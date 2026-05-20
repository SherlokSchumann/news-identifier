# Import the necessary libraries at the beginning

from pathlib import Path
import numpy as np
import pandas as pd
import os
import tempfile
import mlflow
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, losses, metrics
from tensorflow.keras.datasets import reuters
from keras import ops
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, SimpleRNN, LSTM, GRU, Bidirectional
from tensorflow.keras.utils import to_categorical

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

#--------------------Transformer block layer--------------------

# The code for creating the transformer block and calling its elements is the same as shared in the example code

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(out1 + ffn_output)

print("Successfully created the trnsformer block layer done")

#--------------------Positional encoding layer--------------------

class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super().__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = ops.shape(x)[-1]
        positions = ops.arange(start=0, stop=maxlen, step=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

print("Successfully created the token and positional embedding layer done")

#--------------------loading the preprocessed dataset for training only--------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "dataset"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

maxlen = 200
with np.load(DATA_DIR / "dataset_preprocessed.npz", allow_pickle=True) as f:
    x_train = f["x_train"]
    y_train = f["y_train"]
    x_val   = f["x_test"]
    y_val   = f["y_test"]


x_train = pad_sequences(x_train, maxlen=maxlen)
x_val = pad_sequences(x_val, maxlen=maxlen)
print(f"Loaded preprocessed dataset: x_train: {len(x_train)} sequences, x_val: {len(x_val)} sequences")

#--------------------create the classifier using transformer and encoder--------------------

def build_and_save_transformer_classifier(num_transformer_layers, save_dir='/models'):
    #default
    embed_dim = 32
    ff_dim2 = 64
    ff_dim1 = 32
    num_heads = 4
    vocab_size = 10000
    num_classes = 46


    inputs = layers.Input(shape=(maxlen,), dtype='int32')

    # position embedding
    embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
    x = embedding_layer(inputs)


    # For the first layer - Non Intermediate
    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim1)
    x = transformer_block(x)


    # The intermediate layers
    for index in range(num_transformer_layers - 2):
        x = TransformerBlock(embed_dim, num_heads, ff_dim2)(x)


    # The Final Layer
    x = transformer_block(x)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(20, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

   
    return model

#-----------------------------------------------------------

#--------------------function to compute training metrics--------------------

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

#--------------------fit the three layer transformer model and compute the training metrics--------------------

num_transformer_layers = 3
model = build_and_save_transformer_classifier(num_transformer_layers)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
history = model.fit(x_train, y_train, epochs=8, batch_size=64, validation_data=(x_val, y_val))

ComputeMetrics(y_train  , np.argmax(model.predict(x_train), axis=1))

model.save(MODEL_DIR / 'transformer_classifier_3_layers.h5')

#--------------------do for 5 layers--------------------
num_transformer_layers = 5
model = build_and_save_transformer_classifier(num_transformer_layers)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
history = model.fit(x_train, y_train, epochs=6, batch_size=64, validation_data=(x_val, y_val))

ComputeMetrics(y_train  , np.argmax(model.predict(x_train), axis=1))

model.save(MODEL_DIR / 'transformer_classifier_5_layers.h5')

#--------------------do for 7 layers--------------------
num_transformer_layers = 7
model = build_and_save_transformer_classifier(num_transformer_layers)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
history = model.fit(x_train, y_train, epochs=7, batch_size=64, validation_data=(x_val, y_val))

ComputeMetrics(y_train  , np.argmax(model.predict(x_train), axis=1))

model.save(MODEL_DIR / 'transformer_classifier_7_layers.h5')