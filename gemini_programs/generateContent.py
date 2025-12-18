from google import genai

client = genai.Client()

my_file = client.files.upload(file=r"C:\Users\NIVAS\Personal\Agentic_Ai\Gen_ai_practice\my_photo.jpg")

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=[my_file, "Caption this image."],
)

print(response.text)