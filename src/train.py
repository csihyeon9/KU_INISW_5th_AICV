import os
import tensorflow as tf
from preprocess import load_data, tokenize_data
from model import build_model

# GPU 설정
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print("Using GPU:", gpus[0])
    except RuntimeError as e:
        print("GPU configuration error:", e)

# Ensure TensorBoard log directory exists
LOG_DIR = './logs'
if os.path.exists(LOG_DIR):
    if not os.path.isdir(LOG_DIR):
        os.remove(LOG_DIR)  # Remove if 'logs' exists as a file
os.makedirs(LOG_DIR, exist_ok=True)

# Constants
INPUT_SIZE = 500
VOCAB_SIZE = 10000
EPOCHS = 20
BATCH_SIZE = 128
DATA_PATHS = {
    "train": "../data/VDISC_train.hdf5",
    "validate": "../data/VDISC_validate.hdf5",
}

# Load and preprocess data
train_data = load_data(DATA_PATHS["train"])
validate_data = load_data(DATA_PATHS["validate"])

# Tokenizer setup
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=VOCAB_SIZE)
tokenizer.fit_on_texts(train_data['functionSource'])

x_train = tokenize_data(tokenizer, train_data, INPUT_SIZE)
x_validate = tokenize_data(tokenizer, validate_data, INPUT_SIZE)

y_train = train_data['combine'].values
y_validate = validate_data['combine'].values

# Build and train model
model = build_model(INPUT_SIZE, VOCAB_SIZE)
model.fit(
    x_train, y_train,
    validation_data=(x_validate, y_validate),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight={0: 1.0, 1: 5.0},
    callbacks=[
        tf.keras.callbacks.ModelCheckpoint('../saved_models/model.weights.h5', save_best_only=True, save_weights_only=True),
        tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR)
    ]
)
