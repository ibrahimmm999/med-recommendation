import pandas as pd
from transformers import AutoTokenizer
import tensorflow as tf
import numpy as np

def preprocess_texts(texts, tokenizer, max_length=128):
    if isinstance(texts, pd.Series):
        texts = texts.tolist()
    encoding = tokenizer(texts, padding='max_length', truncation=True, max_length=max_length, return_tensors='tf')
    return encoding['input_ids'], encoding['attention_mask']

def encode_text(tokenizer, text, max_length):
    encoding = tokenizer(
        text,
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='tf'
    )
    return encoding['input_ids'], encoding['attention_mask']

def decode_single_text(input_ids, tokenizer):
    decoded_text = tokenizer.decode(input_ids, skip_special_tokens=True)
    return decoded_text

def get_similar_text(user_input, tokenizer, train_input_ids):
    encoded_text = encode_text(tokenizer, user_input, 128)[0]
    most_similar_text = get_most_similar_text(encoded_text, train_input_ids)
    decoded_text = decode_single_text(most_similar_text, tokenizer)
    real_word = standardize_output(decoded_text)
    return real_word

def get_most_similar_text(text_encoded, train_input_ids):
    embedding_matrix = tf.random.uniform((100000, 100))  # Example embedding matrix for demonstration
    max_jaccard_similarity = -1  # Initialize with lowest possible similarity
    most_similar_text = None

    embedding_text_encoded = tf.nn.embedding_lookup(embedding_matrix, text_encoded)
    embedding_text_encoded = tf.cast(embedding_text_encoded, dtype=tf.float32)

    for i in train_input_ids:
        embedding_text_train = tf.nn.embedding_lookup(embedding_matrix, i)
        embedding_text_train = tf.cast(embedding_text_train, dtype=tf.float32)
        jaccard_sim = jaccard_similarity(embedding_text_encoded, embedding_text_train)

        if jaccard_sim > max_jaccard_similarity:
            max_jaccard_similarity = jaccard_sim
            most_similar_text = i

    return most_similar_text

def jaccard_similarity(embedded_doc1, embedded_doc2):
    embedded_doc1 = tf.constant(embedded_doc1)
    embedded_doc2 = tf.constant(embedded_doc2)

    numpy_doc1 = embedded_doc1.numpy()
    numpy_doc2 = embedded_doc2.numpy()

    flattened_doc1 = numpy_doc1.reshape(-1, numpy_doc1.shape[-1])
    flattened_doc2 = numpy_doc2.reshape(-1, numpy_doc2.shape[-1])

    set1 = set(tuple(row) for row in flattened_doc1)
    set2 = set(tuple(row) for row in flattened_doc2)
    
    intersection = len(set1.intersection(set2))
    union = len(set1) + len(set2) - intersection
    jaccard_sim = intersection / union

    return jaccard_sim

def standardize_output(text):
    filter_1 = text.replace('(', '')
    filter_2 = filter_1.replace(')', '')
    filter_3 = filter_2.replace('-', '')
    filter_4 = filter_3.replace('   ', '')
    return filter_4.replace('  ', ' ')
