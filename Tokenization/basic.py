from transformers import BertTokenizer,BertModel
# import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

text = "I Love Learnig about AI"

inputs  = tokenizer(text,return_tensors='pt')  #Words are assigned to numerical ids 

output = model(**inputs) #Based on id embeddings are generated 

# print(output)

# print(inputs)

# print("Token IDS",inputs['input_ids'])

# print("Embeddings shape",output.last_hidden_state.shape)

cls_embedding = output.last_hidden_state[:,0,:]
print(cls_embedding.shape)