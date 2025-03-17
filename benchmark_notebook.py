import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from generated_text_detector.utils.model.roberta_classifier import RobertaClassifier
from generated_text_detector.utils.preprocessing import preprocessing_text
from transformers import AutoTokenizer
import torch.nn.functional as F


# Load the JSON dataset into a pandas DataFrame
data_dp = pd.read_json('/content/direct_prompt_test.json')  
data_mda = pd.read_json('/content/multi_domains_arxiv_test.json')  
data_mdwp = pd.read_json('/content/multi_domains_writing_prompt_test.json')  
data_mdyr = pd.read_json('/content/multi_domains_yelp_review_test.json')  

# Split eache data set into training and testing sets 
train_data_dp, test_data_dp = train_test_split(data_dp, test_size=0.2, random_state=42)
train_data_mda, test_data_mda = train_test_split(data_mda, test_size=0.2, random_state=42)
train_data_mdwp, test_data_mdwp = train_test_split(data_mdwp, test_size=0.2, random_state=42)
train_data_mdyr, test_data_mdyr = train_test_split(data_mdyr, test_size=0.2, random_state=42)


print("Training data shape:", test_data_mdyr.shape)
print("Testing data shape:", test_data_mda.shape)
print("Testing data shape:", test_data_mdwp.shape)
print("Testing data shape:", test_data_dp.shape)


model = RobertaClassifier.from_pretrained("SuperAnnotate/ai-detector")
tokenizer = AutoTokenizer.from_pretrained("SuperAnnotate/ai-detector")

model.eval()

def predict_text(input_text):
    text_example = preprocessing_text(input_text)

    tokens = tokenizer.encode_plus(
        text_example,
        add_special_tokens=True,
        max_length=512,
        padding='longest',
        truncation=True,
        return_token_type_ids=True,
        return_tensors="pt")

    _, logits = model(**tokens)

    proba = F.sigmoid(logits).squeeze(1).item()
    return proba

# Create a new column 'prediction' and the 'predicted_label'
test_data_mdyr['prediction'] = test_data_mdyr['text'].apply(predict_text)
test_data_mdyr['predicted_label'] = test_data_mdyr['prediction'].apply(lambda x: 'human' if x < 0.5 else 'llm')

test_data_mda['prediction'] = test_data_mda['text'].apply(predict_text)
test_data_mda['predicted_label'] = test_data_mda['prediction'].apply(lambda x: 'human' if x < 0.5 else 'llm')

test_data_mdwp['prediction'] = test_data_mdwp['text'].apply(predict_text)
test_data_mdwp['predicted_label'] = test_data_mdwp['prediction'].apply(lambda x: 'human' if x < 0.5 else 'llm')

test_data_dp['prediction'] = test_data_dp['text'].apply(predict_text)
test_data_dp['predicted_label'] = test_data_dp['prediction'].apply(lambda x: 'human' if x < 0.5 else 'llm')


cm_mdyr = confusion_matrix(test_data_mdyr['label'], test_data_mdyr['predicted_label'])

# Plotting the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm_mdyr, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['human', 'llm'], yticklabels=['human', 'llm'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


cm_mda = confusion_matrix(test_data_mda['label'], test_data_mda['predicted_label'])

# Plotting the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm_mda, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['human', 'llm'], yticklabels=['human', 'llm'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

cm_dp = confusion_matrix(test_data_dp['label'], test_data_dp['predicted_label'])

# Plotting the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm_dp, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['human', 'llm'], yticklabels=['human', 'llm'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

cm_mdwp = confusion_matrix(test_data_mdwp['label'], test_data_mdwp['predicted_label'])

# Plotting the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm_mdwp, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['human', 'llm'], yticklabels=['human', 'llm'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
