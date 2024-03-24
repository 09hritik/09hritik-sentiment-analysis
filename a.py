import torch
from transformers import DistilBertForSequenceClassification ,DistilBertTokenizer

# Define the path to the checkpoint file you want to load
checkpoint_path = "sentiment-analysis/checkpoint-2134"  # Replace "X" with the checkpoint number or name
id2label = {0: "Sadness", 1: "Joy",2: "Love", 3: "Anger", 4: "Fear", 5: "Surprise"}
label2id = { "Sadness" :0, "Joy" :1,"Love" :2, "Anger" :3, "Fear" :4, "Surprise" :5}

# Load the model from the checkpoint
model = DistilBertForSequenceClassification.from_pretrained(checkpoint_path, id2label=id2label, label2id=label2id)

# Ensure model is in evaluation mode
model.eval()

# Sample input text
input_text = "Today was a tough day. Everything seemed to go wrong, and I'm feeling really stressed out"

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Tokenize the input text
inputs = tokenizer(input_text, return_tensors="pt")

# Ensure model is in evaluation mode
model.eval()

# Make prediction
with torch.no_grad():
    outputs = model(**inputs)

# Interpret the output
predictions = outputs.logits.argmax(dim=1)
predicted_label = id2label[predictions.item()]

print("Predicted emotion:", predicted_label)



