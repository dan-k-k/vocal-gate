# ml_pipeline/check_labels.py
import csv
import os

vocab_path = "./FSD50K/FSD50K.ground_truth/vocabulary.csv"

if not os.path.exists(vocab_path):
    print("Could not find vocabulary.csv!")
else:
    print("=== All Available FSD50K Labels ===")
    with open(vocab_path, 'r') as f:
        # FSD50K vocabulary usually has ID, Label, and sometimes a MID
        reader = csv.reader(f)
        for row in reader:
            # Usually the label name is the second column (index 1)
            if len(row) > 1:
                print(row[1])

