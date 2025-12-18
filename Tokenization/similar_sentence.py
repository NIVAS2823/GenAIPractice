import torch
from transformers import BertTokenizer,BertModel
import torch.nn.functional as F

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def embed(sentence):
    inputs = tokenizer(sentence,return_tensors='pt')
    with torch.no_grad():
        output = model(**inputs)

    return output.last_hidden_state[:,0,:]

s1 = "AI is transforming the world"
s2 = "Humans are dogs"

v1 = embed(s1)
v2 = embed(s2)

similarity = F.cosine_similarity(v1,v2)
print("Similarity:",similarity.item())