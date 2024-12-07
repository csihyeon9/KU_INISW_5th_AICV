import tensorflow as tf

def build_model(input_size, vocab_size, embedding_dim=13):
    """
    Build a CNN model for vulnerability detection.
    Args:
        input_size (int): Input sequence length.
        vocab_size (int): Size of the vocabulary.
        embedding_dim (int): Dimension of the embedding space.
    Returns:
        tf.keras.Model: Compiled Keras model.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=input_size),
        tf.keras.layers.Conv1D(filters=512, kernel_size=9, activation='relu'),
        tf.keras.layers.MaxPooling1D(pool_size=5),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.005),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model
