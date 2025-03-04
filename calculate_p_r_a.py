import pandas as pd
from sklearn.metrics import precision_score, accuracy_score, f1_score, recall_score


def calculate_metrics(true_file, pred_file, output_file):

    true_labels = pd.read_csv(true_file, header=None).squeeze()
    pred_labels = pd.read_csv(pred_file, header=None).squeeze()

    print(len(true_labels))
    print(len(pred_labels))


    true_labels = true_labels.reset_index(drop=True)
    pred_labels = pred_labels.reset_index(drop=True)


    mask = true_labels != 0
    true_labels_filtered = true_labels[mask]
    pred_labels_filtered = pred_labels[mask]

    precision = precision_score(true_labels_filtered, pred_labels_filtered, average='macro')  # or 'macro' or 'weighted'
    recall = recall_score(true_labels_filtered, pred_labels_filtered, average='macro')
    f1 = f1_score(true_labels_filtered, pred_labels_filtered, average='macro')
    accuracy = accuracy_score(true_labels_filtered, pred_labels_filtered)


    with open(output_file, 'w') as f:
        f.write(f"Precision: {precision}\n")
        f.write(f"Recall: {recall}\n")
        f.write(f"F1 Score: {f1}\n")
        f.write(f"Accuracy: {accuracy}\n")

    print(f"Metrics saved to {output_file}")

import os
base_dir = os.getcwd()
folder = os.path.join(base_dir, "svm_loss")
val_file = os.path.join(folder, "val_predictions.csv")
test_file = os.path.join(folder, "test_predictions.csv")
val_file_true = os.path.join(folder, "val_true.csv")
test_file_true = os.path.join(folder, "test_true.csv")
output_file_val =  os.path.join(folder, "results_val.txt")
output_file_test =  os.path.join(folder, "results_test.txt")


calculate_metrics(val_file_true, val_file, output_file_val)


calculate_metrics(test_file_true, test_file, output_file_test)