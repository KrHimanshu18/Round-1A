import pandas as pd # type: ignore
import json

# Load your training data
with open('training_data/gemini_training_data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Use pandas to count the labels
df = pd.DataFrame(data)
label_counts = df['label'].value_counts()

print("Label counts in your training data:")
print(label_counts)