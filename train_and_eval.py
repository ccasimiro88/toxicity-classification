from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from datasets import load_dataset
import torch
import numpy as np
from sklearn.metrics import classification_report
import pandas as pd
import sys
import os

# for reproducibility
seed = 33

# Load data with the "datasets" library to create the model features


def load_data(data_files):
    dataset = load_dataset("csv", data_files=data_files)

    # extract labels to predict
    labels = dataset["train"].unique("label")
    return dataset, labels

# train the model and eval on the test set


def train_and_eval(dataset, labels, model_name, output_dir):
    # set labels configuration
    id2label = {0: 'Nulo', 1: 'Decibelios_suaves', 2: 'Decibelios_fuertes'}
    label2id = {'Nulo': 0, 'Decibelios_suaves': 1, 'Decibelios_fuertes': 2}
    config = AutoConfig.from_pretrained(model_name,
                                        num_labels=len(labels),
                                        id2label=id2label,
                                        label2id=label2id)

    model = AutoModelForSequenceClassification.from_pretrained(model_name,
                                                               config=config)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # tokenize examples to construct the model features
    def tokenize(examples):
        return tokenizer(examples['text'],
                         max_length=128,
                         truncation=True,
                         padding=True)

    # remove useless columns
    dataset_tokenized = dataset.map(tokenize,
                                    batched=True,
                                    remove_columns=['Unnamed: 0', 'twitterId'],
                                    desc="tokenizing dataset")

    train_dataset = dataset_tokenized['train']
    test_dataset = dataset_tokenized['test']

    # setup trainer to build the training loop
    # defines training arguments
    training_args = TrainingArguments(output_dir=output_dir,
                                      per_device_train_batch_size=6,
                                      per_device_eval_batch_size=6,
                                      gradient_accumulation_steps=2,
                                      num_train_epochs=5,
                                      save_strategy="epoch",
                                      load_best_model_at_end=True,
                                      logging_strategy="steps",
                                      evaluation_strategy="epoch",
                                      logging_steps=100,
                                      logging_dir=f"{output_dir}/tb",
                                      seed=seed
                                      )
    trainer = Trainer(model=model,
                      tokenizer=tokenizer,
                      args=training_args,
                      train_dataset=train_dataset,
                      eval_dataset=test_dataset,
                      )

    # start the training
    train_results = trainer.train()

    # save the model
    trainer.save_model()

    trainer.log_metrics("train", train_results.metrics)
    trainer.save_metrics("train", train_results.metrics)
    trainer.save_state()

    # evaluate best model on test set and compute metrics
    # store classification report in a table
    def metrics(model_preds):
        # evaluate the model with several metrics suitable for classification
        # and get predictions
        refs = model_preds.label_ids
        preds = model_preds.predictions
        preds = np.argmax(preds, axis=1)
        metrics = classification_report(refs, preds,
                                        output_dict=True)

        return metrics

    trainer.compute_metrics = metrics
    eval_metrics = trainer.evaluate()

    return eval_metrics


if __name__ == '__main__':

    model_name = sys.argv[1]

    # Load data
    train_file = "decibelios_train_data.csv"
    test_file = "decibelios_test_data.csv"

    data_files = {"train": train_file,
                  "test": test_file}

    dataset, labels = load_data(data_files)

    # Training Roberta-base-bne
    # model_name = "PlanTL-GOB-ES/roberta-base-bne"
    # model_name = "xlm-roberta-base"
    output_dir = f"classifiers/{model_name}"

    metrics = train_and_eval(dataset,
                             labels,
                             model_name,
                             output_dir)

    # store metrics in a table
    df_metrics = pd.DataFrame.from_dict(metrics)
    df_metrics.columns = df_metrics.columns.str.replace("eval_", "")
    df_metrics.to_csv(f"{output_dir}/eval_scores.csv")
