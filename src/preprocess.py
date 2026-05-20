from pathlib import Path
import numpy as np

vocab_size = 10000   # words beyond this are replaced with <UNK>

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "dataset"
INPUT_PATH = DATA_DIR / "reuters.npz"
OUTPUT_PATH = DATA_DIR / "dataset_preprocessed.npz"

with np.load(INPUT_PATH, allow_pickle=True) as f:
    x = f["x"]
    y = f["y"]

# Create a reproducible train/test split because the archived dataset contains only x and y.
indices = np.arange(len(x))
np.random.seed(42)
np.random.shuffle(indices)

split_idx = int(len(indices) * 0.8)
train_idx, test_idx = indices[:split_idx], indices[split_idx:]

x_train_raw = x[train_idx]
y_train = y[train_idx]
x_test_raw = x[test_idx]
y_test = y[test_idx]


def preprocess(sequences):
    result = []
    for seq in sequences:
        clipped = [idx if idx < vocab_size else 2 for idx in seq]
        result.append(np.array(clipped, dtype=np.int32))
    return np.array(result, dtype=object)

x_train = preprocess(x_train_raw)
x_test  = preprocess(x_test_raw)

np.savez(
    OUTPUT_PATH,
    x_train=x_train,
    y_train=y_train,
    x_test=x_test,
    y_test=y_test
)

print(f"x_train: {len(x_train)} sequences, x_test: {len(x_test)} sequences")