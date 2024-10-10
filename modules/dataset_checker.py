import pandas as pd
import manage_datasets as md
import os

def check_balanced_db(file_name = "composite_db", threshold = 0.15):
    path = os.path.join(md.get_DB_path(), file_name)
    df = pd.read_csv(path)
    class_counts = df['Value'].value_counts()

    print(class_counts)

    total_instances = len(df)
    class_proportions = class_counts / total_instances

    print(class_proportions)

    is_balanced = abs(class_proportions[0] - class_proportions[1]) < threshold

    if is_balanced:
        print("✅\tDatabase: balanced\n")
    else:
        print("❌\tDatabase: unbalanced\n")

    return is_balanced
