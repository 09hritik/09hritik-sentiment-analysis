import json
import evaluate
import numpy as np
from datasets import load_dataset
from transformers import DistilBertTokenizer , DataCollatorWithPadding , DistilBertForSequenceClassification , TrainingArguments , Trainer , BertConfig 
data = load_dataset("dair-ai/emotion")
#print(data["test"][0])

#load a tokenizer (BertTokenizer in our case)
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

#preprocessing function to tokenize and truncate
def preprocess_data(examples):
    return tokenizer(examples["text"], trucation = True)

tokenized_data = data.map(preprocess_data, batched= True)        #True: to procoess multiple batches at ones
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


accuracy = evaluate.load("accuracy")
#print(accuracy.compute(references=[0,1,0,1] , predictions=[1,0,0,1]))

def compute_metrics(eval_pred):
    predictions , labels = eval_pred
    predictions = np.argmax(predictions, axis= 1)
    return accuracy.compute(predictions=predictions, references=labels)

id2label = {0: "Sadness", 1: "Joy",2: "Love", 3: "Anger", 4: "Fear", 5: "Surprise"}
label2id = { "Sadness" :0, "Joy" :1,"Love" :2, "Anger" :3, "Fear" :4, "Surprise" :5}

model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased" , id2label=id2label ,label2id=label2id)


trainingargs = TrainingArguments(
    output_dir= "sentiment-analysis",
    learning_rate = 0.00005 ,
    evaluation_strategy = "epoch" ,
    per_device_train_batch_size = 15,
    per_device_eval_batch_size=15,
    num_train_epochs=2,
    weight_decay=0.01,
    save_strategy= "epoch",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=trainingargs,
    data_collator=data_collator,
    train_dataset=tokenized_data["train"],
    eval_dataset=tokenized_data["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()

