from openai import OpenAI

client = OpenAI()

file = client.files.create(
    file=open(r'C:\Users\NIVAS\Personal\Agentic_Ai\Gen_ai_practice\ReAct-Paper.pdf','rb'),
    purpose="assistants"
)

response=client.responses.create(
    model='gpt-4o-mini',
    input=[
        {"role":"user",
         "content":[
             {
                 "type":"input_file",
                 "file_id":file.id
             },
             {
                 "type":"input_text",
                 "text":"what is the main concept of the paper?"
             },
         ],

        },
    ]
)

print(response.output_text)