import tensorflow as tf
from preprocess import load_data, tokenize_data
from model import build_model
from sklearn.metrics import confusion_matrix, classification_report

# GPU 설정
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print("Using GPU:", gpus[0])
    except RuntimeError as e:
        print("GPU configuration error:", e)

# Constants
INPUT_SIZE = 500
VOCAB_SIZE = 10000
MODEL_PATH = '../saved_models/model.weights.h5'
TEST_DATA_PATH = "../data/VDISC_test.hdf5"

# Load and preprocess test data
test_data = load_data(TEST_DATA_PATH)

tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=VOCAB_SIZE)
tokenizer.fit_on_texts(test_data['functionSource'])

x_test = tokenize_data(tokenizer, test_data, INPUT_SIZE)
y_test = test_data['combine'].values

# Rebuild model and load weights
model = build_model(INPUT_SIZE, VOCAB_SIZE)
model.load_weights(MODEL_PATH)

# Evaluate model
results = model.evaluate(x_test, y_test, batch_size=128)
print(f"Test Accuracy: {results[1]}")

# Additional metrics
y_pred = (model.predict(x_test) > 0.5).astype("int32")
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
