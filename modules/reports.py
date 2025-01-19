
import os
import pandas as pd
import classifier

def check_balanced_db(folder_name, file_name, threshold = 0.15):
    RESET = '\033[0m'
    YELLOW = '\033[33m' 

    if file_name == "composite_db":
        path = os.path.join(folder_name, file_name)

    df = pd.read_csv(path)
    class_counts = df['Value'].value_counts()


    total_instances = len(df)
    class_proportions = class_counts / total_instances

    print(f"Name: {YELLOW}{file_name}{RESET}")
    print(class_counts)
    print(class_proportions)

    is_balanced = abs(class_proportions[0] - class_proportions[1]) < threshold

    if is_balanced:
        print(f"✅ Database: {YELLOW}BALANCED{RESET}\n")
    else:
        print(f"❌ Database: {YELLOW}UNBALANCED{RESET}\n")

    return is_balanced
def generate_markdown_table(row):
    text = '''
| Class | Number of examples | Proportion |
| ----- | ------------------ | ---------- |'''
    for i, clss in enumerate(row["classes"]):
        text += f'''
| `{clss}` | `{row["examples"][i]}` | `{row["proportion"][i]}` |'''
    
    text += f"|**Total** | `{row['total']}` | 1 |"   
    return text

def generate_balancing_info_markdown(row, threshold):
    value = "✅ **BALANCED**" if min(row['proportion']) < threshold else "❌ **UNBALANCED**"
    return f'''{value} - `{min(row['proportion']) * 100}%` for the minority class.'''

def generate_path_info_markdown(row):
    value = f"**Path**: `{row['path']}`."
    return f"{value}"


def generate_pie_chart(title, row):
    chart = f'''```mermaid
pie
    title {title}
'''

    for i, class_name in enumerate(row['classes']):
        count = row['examples'][i]
        
        chart += f'    "{class_name}": {count}\n'

    chart += '```'

    return chart



def generate_main_info_markdown(threshold):
    title = f"# Confidence Report\n"

    value = f'''**The Threshold for a well-balanced database is fixed to `{threshold * 100}%`***.
    The first confidence value refers to the confidence related to the individual transcription, which is the minimum confidence threshold for the individual timestamps within the transcription.*
The second confidence value refers to the minimum average confidence for the individual file to be considered in the final CSV.*
'''
    return title + value


def generate_confidence_report_markdown(row):
    value = f'''- **Trascription Confidence**: `{row["confidence"][0]}`
- **Files Confidence**: `{row['confidence'][1]}`
'''
    return value


def generate_threshold_report(rows, threshold):
    iterator = 0
    positive = 0
    result = ""
    lowest = ""
    for index, info in rows.items():
        min_proportion = min(info['proportion'])
        if min_proportion < threshold:
            file_name = info['name']
            file_path = info['path']
            result += f"- [{file_name}]({file_path}).\n"
            positive += 1
        lowest = min_proportion
        iterator += 1
    if iterator != 0:
        return f'''> [!WARNING]
> No one of the *{iterator}* `.csv` files reached the minimum threshold to appear inside this box. 
> The actual Treshold had been set to `{threshold * 100}%` and the highest balancing database can be found setting the threshold to **`{lowest * 100}%`**.
> {result}'''
    else:
        return f'''> [!TIP]
>  All `.csv` files checked and *{iterator}* reached the minimum threshold in this notes section. You can check the following files:
> {result}
'''

def get_file_name(path):
    parts = path.split(os.sep)
    folder_name, sub_folder_name, file_name = parts[-3], parts[-2], parts[-1]
    return file_name[:-3]


def get_confidences(path):
    parts = path.split(os.sep)
    file_name = parts[-1]
    confidences = file_name.split('_')
    first_confidence = confidences[-2] if len(confidences) > 1 and confidences[-2].replace('.', '', 1).isdigit() else 0
    second_confidence = confidences[-1].split('.csv')[0] 
    if not first_confidence and 'single' in file_name and 'files' not in file_name:
        first_confidence = second_confidence
        second_confidence = 0        
    elif not first_confidence and not 'single' in file_name and 'files' in file_name:
        first_confidence = 0

    return [float(first_confidence), float(second_confidence)]

def generate_row_markdown(row, threshold):
    return f'''## {row['name'].replace("_", " ").capitalize()}
- {generate_balancing_info_markdown(row, threshold)}
- **Confidence**:
{generate_path_info_markdown(row)}
{generate_confidence_report_markdown(row)}

{generate_markdown_table(row)}

{generate_pie_chart("Proportion between classes", row)}
'''

def create_or_clean_report_file(file_name="report_confidence.md"):
    if os.path.exists(file_name):
        with open(file_name, "w") as file:
            file.write("")  
    else:
        open(file_name, "w").close()

def populate_report_with_rows(rows, threshold, file_name="report_confidence.md"):
    content = generate_main_info_markdown(threshold) + f"\n\n{generate_threshold_report(rows, threshold)}\n\n"
    
    for index, row in rows.items():
        content += generate_row_markdown(row, threshold) + "\n\n"
    
    with open(file_name, "w", encoding="utf-8") as file:  
        file.write(content)



def collect_csv_info(base_dir):
    rows = {}
    index = 0

    for root, _, files in os.walk(base_dir):
        for file in sorted(files):  
            if file.endswith(".csv"):
                file_path = os.path.join(root, file)
                try:
                    df = pd.read_csv(file_path)
                    
                    class_counts = df['Value'].value_counts()
                    total = len(df)
                    classes = sorted(class_counts.index.tolist())  
                    examples = [class_counts.get(cls, 0) for cls in classes]
                    proportions = [round(count / total, 2) for count in examples]
                    file_name = get_file_name(file_path)
                    confidences = get_confidences(file_path)
                    confidences = [round(confidences[0], 2), round(confidences[1], 2)]
                    rows[index] = {
                        "classes": classes,
                        "examples": examples,
                        "proportion": proportions,
                        "total": len(df),
                        "confidence": confidences,
                        "name": f"{file_name}",
                        "path": file_path
                    }
                    index += 1

                except Exception as e:
                    print(f"Error reading file {file_path}: {e}")

    return rows

rows = collect_csv_info('data\\filtered_csv')
rows2 = collect_csv_info('data\\filtered_positive_csv')
rows.update(rows2)
create_or_clean_report_file()

populate_report_with_rows(rows, .10)

def write_to_markdown(epoch, loss, accuracy, precision, recall, file_name, learning_rate, preprocessor_url, encoder_url):
    if not os.path.exists(file_name):
        with open(file_name, 'w') as f:
            f.write(f"# {file_name}\n\n")
            f.write(f"**Learning Rate**: {learning_rate}\n")
            f.write(f"**Preprocessor URL**: {preprocessor_url}\n")
            f.write(f"**Encoder URL**: {encoder_url}\n\n")
            f.write("| Epoch | Loss | Accuracy | Precision | Recall |")
            f.write("|-------|------|----------|-----------|--------|")
    
    with open(file_name, 'a') as f:
        f.write(f"| {epoch} | {loss:.4f} | {accuracy:.4f} | {precision:.4f} | {recall:.4f} |\n")

def generate_final_report(loss, accuracy, precision, recall, file_name):
    with open(file_name, 'a') as f:
        f.write("\n## Final Evaluation\n")
        f.write("| Loss | Accuracy | Precision | Recall |\n")
        f.write("|------|----------|-----------|--------|\n")
        f.write(f"|{loss:.4f}|{accuracy:.4f}|{precision:.4f}|{recall:.4f}|\n")

def train_and_log_markdown(X_train, y_train, X_test, y_test, model, file_name, learning_rate, preprocessor_url, encoder_url):
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss="binary_crossentropy", metrics=["accuracy", "precision", "recall"])
    history = model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test), verbose=1)
    
    for epoch in range(len(history.history['loss'])):
        loss = history.history['loss'][epoch]
        accuracy = history.history['accuracy'][epoch]
        precision = history.history['precision'][epoch]
        recall = history.history['recall'][epoch]
        write_to_markdown(epoch + 1, loss, accuracy, precision, recall, file_name, learning_rate, preprocessor_url, encoder_url)

    final_loss = history.history['loss'][-1]
    final_accuracy = history.history['accuracy'][-1]
    final_precision = history.history['precision'][-1]
    final_recall = history.history['recall'][-1]
    generate_final_report(final_loss, final_accuracy, final_precision, final_recall, file_name, learning_rate, preprocessor_url, encoder_url)
