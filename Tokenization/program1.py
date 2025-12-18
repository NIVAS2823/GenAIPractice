from transformers import BertModel,BertTokenizer

tokenizer  = BertTokenizer.from_pretrained('bert-base-uncased')

model = BertModel.from_pretrained('bert-base-uncased')

text  = "The recent release from google was a hit"

encoded_input = tokenizer(text,return_tensors='pt')

print(encoded_input)

# output = model(**encoded_input)

# print(output)