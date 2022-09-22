from ast import literal_eval
from math import ceil

import numpy as np
import pandas as pd
from datasets import Dataset


class TrainDataModule:
    def __init__(
        self,
        tokenizer,
        feature_file,
        annotation_file,
        notes_file,
        test_size=0.1,
        max_length=512,
        seed=42,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.feature_file = feature_file
        self.annotation_file = annotation_file
        self.notes_file = notes_file
        self.test_size = test_size
        self.max_length = max_length
        self.seed = seed

    def setup(self):
        dataset = self.load_dataset()
        dataset = self.process_data(dataset)
        train_dataset, test_dataset = self.split_dataset(dataset)
        return train_dataset, test_dataset

    def load_dataset(self):
        feature_df = pd.read_csv(self.feature_file)
        annotation_df = pd.read_csv(self.annotation_file)
        notes_df = pd.read_csv(self.notes_file)
        df = pd.merge(annotation_df, feature_df, how="left", on=["feature_num", "case_num"])
        df = pd.merge(df, notes_df, how="left", on=["pn_num", "case_num"])
        dataset = Dataset.from_pandas(df)
        return dataset

    def process_data(self, dataset):
        dataset = dataset.map(self.process_location, batched=False)
        dataset = dataset.map(self.process_annotation, batched=False)
        dataset = dataset.map(self.process_feature_text, batched=False)
        dataset = dataset.map(self.tokenize, batched=True)
        dataset = dataset.map(self.align_labels, batched=False)
        cols_to_keep = ["input_ids", "attention_mask", "token_type_ids", "labels"]
        dataset.set_format("torch", columns=cols_to_keep)
        return dataset

    def process_location(self, example):
        locations = []
        location_list = literal_eval(example["location"])
        for location_str in location_list:
            if not location_str:
                continue
            for s in location_str.split(";"):
                start_index, end_index = s.split()
                locations.append((int(start_index), int(end_index)))
        example["location"] = locations
        return example

    def process_annotation(self, example):
        example["annotation"] = literal_eval(example["annotation"])
        return example

    def process_feature_text(self, example):
        example["feature_text"] = example["feature_text"].replace("-OR-", " or ")
        example["feature_text"] = example["feature_text"].replace("-", " ")
        return example

    def split_dataset(self, dataset):
        groups = np.array(dataset.unique("pn_num"))
        np.random.seed(self.seed)
        np.random.shuffle(groups)

        num_groups = len(groups)
        num_train = num_groups - ceil(self.test_size * num_groups)

        train_groups = set(groups[:num_train])
        test_groups = set(groups[num_train:])

        train_dataset = dataset.filter(lambda example: example["pn_num"] in train_groups)
        test_dataset = dataset.filter(lambda example: example["pn_num"] in test_groups)
        return train_dataset, test_dataset

    def tokenize(self, batch):
        tokenized_inputs = self.tokenizer(
            batch["feature_text"],
            batch["pn_history"],
            truncation="only_second",
            max_length=self.max_length,
            padding=False,
            return_offsets_mapping=True,
        )
        batch_size = len(batch["pn_history"])
        tokenized_inputs["sequence_ids"] = [tokenized_inputs.sequence_ids(i) for i in range(batch_size)]
        return tokenized_inputs

    def align_labels(self, example):
        num_tokens = len(example["input_ids"])
        labels = [0] * num_tokens
        for i in range(num_tokens):
            sequence_id = example["sequence_ids"][i]
            if sequence_id in (None, 0):
                labels[i] = -100
            token_start, token_end = example["offset_mapping"][i]
            for location_start, location_end in example["location"]:
                if (
                    token_start <= location_start < token_end
                    or token_start < location_end <= token_end
                    or location_start <= token_start < location_end
                ):
                    labels[i] = 1
        example["labels"] = labels
        return example


class PredictDataModule:
    def __init__(
        self,
        tokenizer=None,
        feature_file=None,
        notes_file=None,
        max_length=512,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.feature_file = feature_file
        self.notes_file = notes_file
        self.max_length = max_length

    def setup(self):
        dataset = self.load_dataset()
        dataset = self.process_data(dataset)
        return dataset

    def load_dataset(self):
        feature_df = pd.read_csv(self.feature_file)
        notes_df = pd.read_csv(self.notes_file)
        df = pd.merge(feature_df, notes_df, how="left", on="case_num")
        dataset = Dataset.from_pandas(df)
        return dataset

    def process_data(self, dataset):
        dataset = dataset.map(self.process_feature_text, batched=False)
        dataset = dataset.map(self.tokenize, batched=True)
        cols_to_keep = ["input_ids", "attention_mask", "token_type_ids", "labels"]
        dataset.set_format("torch", columns=cols_to_keep)
        return dataset

    def process_feature_text(self, example):
        example["feature_text"] = example["feature_text"].replace("-OR-", " or ")
        example["feature_text"] = example["feature_text"].replace("-", " ")
        return example

    def tokenize(self, batch):
        tokenized_inputs = self.tokenizer(
            batch["feature_text"],
            batch["pn_history"],
            truncation="only_second",
            max_length=self.max_length,
            padding=False,
            return_offsets_mapping=True,
        )
        batch_size = len(batch["pn_history"])
        tokenized_inputs["sequence_ids"] = [tokenized_inputs.sequence_ids(i) for i in range(batch_size)]
        return tokenized_inputs
