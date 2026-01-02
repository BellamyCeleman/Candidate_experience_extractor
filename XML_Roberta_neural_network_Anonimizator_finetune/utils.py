import numpy as np
from seqeval.metrics import classification_report

def align_labels_with_tokens(examples, tokenizer, max_len=512):
    """
    Выравнивает NER-теги с токенами (учитывает разбивку на подслова).
    """
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        max_length=max_len
    )

    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []

        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def compute_metrics_factory(label_list):
    """
    Возвращает функцию метрик, "замкнутую" на список меток.
    Нужно, чтобы передать label_list внутрь Trainer.
    """

    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = classification_report(true_labels, true_predictions, output_dict=True)
        return {
            "precision": results["weighted avg"]["precision"],
            "recall": results["weighted avg"]["recall"],
            "f1": results["weighted avg"]["f1-score"],
            "accuracy": results["accuracy"],
        }

    return compute_metrics