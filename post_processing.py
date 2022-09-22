import itertools

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support


def compute_metrics(dataset, prediction_output):
    all_labels = []
    all_binary_preds = []
    for example, token_probs in zip(dataset, prediction_output.predictions):
        num_chars = len(example["pn_history"])

        # Convert ground-truth location indexes into a binary vector
        labels = convert_location_indexes_to_binary_vector(example["location"], num_chars)
        all_labels.append(labels)

        # Convert predicted location indexes into a binary vector
        predicted_locations = char_probs_to_location_indexes(example, token_probs)
        binary_preds = convert_location_indexes_to_binary_vector(predicted_locations, num_chars)
        all_binary_preds.append(binary_preds)

    # Flatten the lists
    all_labels = list(itertools.chain(*all_labels))
    all_binary_preds = list(itertools.chain(*all_binary_preds))

    results = precision_recall_fscore_support(all_labels, all_binary_preds, average="binary")
    return {"precision": results[0], "recall": results[1], "f1": results[2]}


def token_probs_to_location_indexes(example, prediction_output):
    char_probs = token_probs_to_char_probs(example, prediction_output.predictions)
    predicted_locations = char_probs_to_location_indexes(example, char_probs)
    return predicted_locations


def token_probs_to_char_probs(token_probs, example):
    char_probs = np.zeros((len(example["pn_history"])))
    for i, (offsets, sequence_id) in enumerate(zip(example["offset_mapping"], example["sequence_ids"])):
        # Ignore special tokens and tokens representing features
        if sequence_id in (None, 0):
            continue
        # Some examinees use "yof" to represent "year old female". When the features are related to the age and the
        # gender of the patient, the correct annotations would be "yo" and "f" respectively. However, the
        # tokenization of "yof" by Deberta tokenizers is ["y", "of"]. So, we have to post-process the token "of"
        # depending on the feature. This tokenization problem does not exist for male.
        start_index, end_index = offsets
        text = example["pn_history"]
        if text[start_index:end_index] == "of" and start_index > 0 and text[start_index - 1 : end_index] == "yof":
            if example["feature_text"] == "Female":
                char_probs[end_index - 1] = 1
            elif example["feature_text"].endswith("year"):
                char_probs[start_index] = token_probs[i - 1]
            else:
                char_probs[start_index:end_index] = token_probs[i]
        else:
            char_probs[start_index:end_index] = token_probs[i]
    return char_probs


def char_probs_to_location_indexes(char_probs, example, threshold=0.5):
    """Get location indexs from probailities of being in the span an annotaion at character level."""
    location_indexes = []
    start_index = None
    end_index = None
    for i, (char, prob) in enumerate(zip(example["pn_history"], char_probs)):
        if prob >= threshold:
            # DeBERTa tokenizer includes space in offset_mapping, so do not start span if starting character is space.
            # See: https://www.kaggle.com/code/junkoda/be-aware-of-white-space-deberta-roberta
            if start_index is None and char != " ":
                start_index = i
                end_index = i
            elif start_index is not None:
                end_index = i
        elif start_index is not None and end_index is not None:
            # Previous span has eneded. Reset start_index and end_index.
            location_indexes.append((start_index, end_index + 1))
            start_index = None
            end_index = None
    # Final cleanup to take care of the edge cases where the end_index is the last character.
    # For example, if probs = [0, 0, 1, 1], end_index will never become None in the previous loop and hence
    # will not be included as part of the answer, so have to take care of them here.
    if start_index is not None and end_index is not None:
        location_indexes.append((start_index, end_index + 1))
    return location_indexes


def convert_location_indexes_to_binary_vector(location_indexes, num_chars):
    """Convert a list of location indexes into a binary vector.

    Example:
    >>> location_indexs = [[1, 2], [5, 8]]
    >>> num_chars = 10
    >>> convert_location_indexes_to_binary_vector(location_indexs, num_chars)
    np.array([0, 1, 1, 0, 0, 1, 1, 1, 1, 0])
    """
    binary_vector = np.zeros((num_chars))
    for start_index, end_index in location_indexes:
        binary_vector[start_index:end_index] = 1
    return list(binary_vector)
