from transformers import BertModel,BertTokenizer,BertForSequenceClassification,Trainer,TrainingArguments
from datasets import load_dataset

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased',num_labels=2)

dataset = load_dataset("imdb",split="train[:1%]")

def tokenize(batch):
    return tokenizer(batch["text"],truncation=True,padding="max_length",max_length=128)

dataset = dataset.map(tokenize,batched=True)
dataset = dataset.rename_column("label","labels")
dataset.set_format(type="torch",columns=["input_ids","attention_mask","labels"])


args = TrainingArguments(
    output_dir = "bert-sentiment",
    per_device_train_batch_size=8,
    num_train_epochs=1,
    logging_steps=10
)

trainer = Trainer(model=model,args = args,train_dataset = dataset)

trainer.train()