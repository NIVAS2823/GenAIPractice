from openai import OpenAI

client = OpenAI()

repsonse = client.responses.create(
    model='gpt-4o-mini',
    # reasoning={"effort":"low"},
    input=[
        {
            "role":"developer",
            "content":  "Talk like a tech beginner "
        },
        {
            "role":"user",
            "content":"Are semicolons optional in javascript?"
        }
    ]
)


print(repsonse.output_text)