from google import genai
from google.genai import types

client = genai.Client()


#basic prompt 
# response = client.models.generate_content(
#     model='gemini-2.5-flash',contents='What\'s the highest mountain in Africa?'
# )

# print(response.text)


#count tokens
# response = client.models.count_tokens(
#     model='gemini-2.5-flash',contents='Explain about India\'s glory like a freedom country'
# )

# print(response)

#Configure model 

response = client.models.generate_content(
    model='gemini-2.5-flash',
    contents='Explain about Compounding as a senior finance advisor + story teller to a 15 year old boy',
    config= types.GenerateContentConfig(
        temperature=0.7,
        top_p=0.95,
        top_k=20,
        candidate_count=1,
        seed=5,
        stop_sequences=["STOP!"],
        presence_penalty=0.0,
        frequency_penalty=0.0
    )
)

print(response.text)
