from processor_regex import classify_with_regex
from processor_llm import classify_with_llm
from processor_bert import classify_with_bert
import pandas as pd

def classify(log):
    labels = []
    for source, log_msg in log:
        label = classify_log(source, log_msg)
        labels.append(label)
    return labels

def classify_log(source, log_msg):
    if source == "LegacyCRM":
        label = classify_with_llm(log_msg)
    else:
        label = classify_with_regex(log_msg)
        if label is None:
            label = classify_with_bert(log_msg)
    return label

def classify_csv(input_file):
    df = pd.read_csv(input_file)
    df["target_label"] = classify(list(zip(df["source"], df["log_message"])))
    output_file = "resources/output.csv"
    df.to_csv(output_file, index=False)


if __name__ == "__main__":
    classify_csv("resources/test.csv")