import base64
from openai import OpenAI

client = OpenAI()

# Load + convert to base64
with open(r"C:\Users\NIVAS\Personal\Agentic_Ai\Gen_ai_practice\example.jpeg", "rb") as f:
    b64 = base64.b64encode(f.read()).decode("utf-8")

data_url = f"data:image/jpeg;base64,{b64}"

response = client.responses.create(
    model="gpt-4o-mini",
    input=[
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Describe about the image provided?"},
                {"type": "input_image", "image_url": data_url},
            ],
        }
    ],
)

print(response.output_text)
