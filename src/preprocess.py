import h5py
import pandas as pd
import tensorflow as tf

def create_combine_label(data):
    """
    Create 'combine' column by applying a logical OR operation on CWE labels.
    Args:
        data (DataFrame): DataFrame containing CWE labels.
    Returns:
        DataFrame: Updated DataFrame with 'combine' column.
    """
    cwe_columns = ['CWE-120', 'CWE-119', 'CWE-469', 'CWE-476', 'CWE-other']
    data['combine'] = data[cwe_columns].any(axis=1).astype(int)
    return data

def load_data(file_path):
    """
    Load and preprocess data from HDF5 files using h5py.
    Args:
        file_path (str): Path to the HDF5 file.
    Returns:
        DataFrame: Preprocessed DataFrame.
    """
    with h5py.File(file_path, 'r') as f:
        function_source = [x.decode('utf-8') for x in f['functionSource']]
        cwe_labels = {
            'CWE-120': f['CWE-120'][:],
            'CWE-119': f['CWE-119'][:],
            'CWE-469': f['CWE-469'][:],
            'CWE-476': f['CWE-476'][:],
            'CWE-other': f['CWE-other'][:]
        }
    
    # Create a DataFrame from the loaded data
    data = pd.DataFrame({
        'functionSource': function_source,
        **cwe_labels
    })

    # Add 'combine' column
    data = create_combine_label(data)
    return data

def tokenize_data(tokenizer, data, input_size):
    """
    Tokenize and pad sequences from the source code.
    Args:
        tokenizer (Tokenizer): Keras tokenizer instance.
        data (DataFrame): Source code dataset.
        input_size (int): Maximum length of sequences.
    Returns:
        numpy.ndarray: Tokenized and padded sequences.
    """
    sequences = tokenizer.texts_to_sequences(data['functionSource'])
    return tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=input_size, padding='post')
