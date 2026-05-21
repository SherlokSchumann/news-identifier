
import re
import json
from pathlib import Path
import streamlit as st
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import load_model

maxlen = 200
vocab_size = 10000

_word_index_path = Path(__file__).parent / "reuters_word_index.json"
with open(_word_index_path) as f:
    word_index = json.load(f)


class TransformerBlock(keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
        self.att = keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [keras.layers.Dense(ff_dim, activation="relu"), keras.layers.Dense(embed_dim)]
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
        maxlen = keras.backend.shape(x)[-1]
        positions = keras.backend.arange(start=0, stop=maxlen, step=1)
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


# Set the title of the Streamlit app

st.title("Transformer Classifier for news articles")

_orig_dense_from_config = keras.layers.Dense.from_config.__func__

# Patch the Dense layer's from_config method to ignore 'quantization_config' if it exists

@classmethod
def _patched_dense_from_config(cls, config):
    config.pop('quantization_config', None)
    return _orig_dense_from_config(cls, config)

keras.layers.Dense.from_config = _patched_dense_from_config

# Since this is a custome model, we must specify the architecture again here

custom_objects = {
    "TransformerBlock": TransformerBlock,
    "TokenAndPositionEmbedding": TokenAndPositionEmbedding,
}

# load the model

model_5 = load_model('models/transformer_classifier_5_layers.h5', custom_objects=custom_objects, compile=False)
model_3 = load_model('models/transformer_classifier_3_layers.h5', custom_objects=custom_objects, compile=False)
model_7 = load_model('models/transformer_classifier_7_layers.h5', custom_objects=custom_objects, compile=False)

user_input = st.text_input("Enter a text to classify:")


# Run preprocessing for input text

def preprocess(text):

    words = re.sub(r'[^a-z\s]', '', text.lower()).split()

    encoded = []

    for w in words:
        idx = word_index.get(w, 2)
        encoded.append(idx)

    for i in range(len(encoded)):
        if encoded[i] >= vocab_size:
            encoded[i] = 2

    padded  = keras.utils.pad_sequences([encoded], maxlen=maxlen)

    return padded

    



models_to_use = {
    "5-layer transformer": model_5,
    "3-layer transformer": model_3,
    "7-layer transformer": model_7
}

results = []
label_names = [
    'cocoa', 'grain', 'veg-oil', 'earn', 'acq', 'wheat', 'copper', 'housing',
    'money-supply', 'coffee', 'sugar', 'trade', 'reserves', 'ship', 'cotton',
    'carcass', 'crude', 'nat-gas', 'cpi', 'money-fx', 'interest', 'gnp',
    'meal-feed', 'alum', 'oilseed', 'gold', 'tin', 'strategic-metal', 'livestock',
    'retail', 'ipi', 'iron-steel', 'rubber', 'heat', 'jobs', 'lei', 'bop',
    'zinc', 'orange', 'pet-chem', 'dlr', 'gas', 'silver', 'wpi', 'hog', 'lead',
]


# Run classification through streamlit UI and show results

if st.button("Classify"):
    padded = preprocess(user_input)

    for name, model in models_to_use.items():
        prediction = model.predict(padded)
        predicted_label_index = np.argmax(prediction)
        predicted_label_name = label_names[predicted_label_index]
        results.append((name, predicted_label_name))

    st.write("Classification Results: ")
    for name, label in results:
        st.write(f"- {name}: {label}")
    
st.write("Note: The three layer transformer had the best performance on the test set among all of them")



    

