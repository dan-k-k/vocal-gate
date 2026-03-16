# ml_pipeline/check_labels.py
import csv
import os

vocab_path = "./FSD50K/FSD50K.ground_truth/vocabulary.csv"

if not os.path.exists(vocab_path):
    print("Could not find vocabulary.csv!")
else:
    print("=== All Available FSD50K Labels ===")
    with open(vocab_path, 'r') as f:
        # FSD50K: ID, Label, sometimes a MID
        reader = csv.reader(f)
        for row in reader:
            if len(row) > 1: # Usually label in 2nd col
                print(row[1])

