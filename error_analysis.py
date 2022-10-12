from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
import json
import sys
import os

# analyze incorrect predictions
def analyze_errors(dataset, model, tokenizer):
    # set pipeline for prediction on GPU
    clf = pipeline('text-classification', model=model,
                   tokenizer=tokenizer, device=0)

    texts = [ex['text'] for ex in dataset['test']]
    ids = [ex['twitterId'] for ex in dataset['test']]
    true_labels = [ex['label'] for ex in dataset['test']]
    pred_labels = [pred['label'] for pred in clf(texts)]
    pred_labels = [model.config.label2id[l] for l in pred_labels]

    # display  confusion matrix, from https://stackoverflow.com/questions/19233771/sklearn-plot-confusion-matrix-with-labels
    cm = confusion_matrix(true_labels, pred_labels)


    errors = {'strong': [], 'weak': []}
    for id, text, true_label, pred_label in zip(ids, texts, true_labels, pred_labels):
        if true_label != pred_label:
            if true_label in [1, 2] and pred_label in [1, 2]:
                errors['weak'].append(
                    {'id': id, 'text': text, 'true_label': true_label, 'pred_label': pred_label})
            else:
                errors['strong'].append(
                    {'id': id, 'text': text, 'true_label': true_label, 'pred_label': pred_label})
    return cm, errors


if __name__ == '__main__':
    model_dir = os.path.realpath(sys.argv[1])

    # Load data
    train_file = "decibelios_train_data.csv"
    test_file = "decibelios_test_data.csv"
    data_files = {"train": train_file,
                  "test": test_file}

    def load_data(data_files):
        dataset = load_dataset("csv", data_files=data_files)

        # extract labels to predict
        labels = dataset["train"].unique("label")
        return dataset, labels

    dataset, labels = load_data(data_files)

    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    cm, errors = analyze_errors(dataset, model, tokenizer)

    # display  confusion matrix, from https://stackoverflow.com/questions/19233771/sklearn-plot-confusion-matrix-with-labels#:~:text=labels%20%3D%20%5B%27business,)%0Aplt.show()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm)
    plt.title(f'{os.path.basename(model_dir)}')
    fig.colorbar(cax)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    plt.savefig(os.path.join(model_dir, 'confusion_matrix.png'))

    with open(os.path.join(model_dir, 'errors.json'), 'w') as fn:
        json.dump(errors, fn)
