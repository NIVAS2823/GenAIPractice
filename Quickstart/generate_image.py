from google import genai
from google.genai import types
from PIL import Image # Used to handle the image data
from io import BytesIO # Needed to read binary data from the response

client = genai.Client()

# Use the correct model ID for image generation
MODEL_ID = 'gemini-2.5-flash-image' 

prompt_text = 'Hi, can create a 3d rendered image of a pig with wings and a top hat flying over a happy futuristic scifi city with lots of greenery?'

response = client.models.generate_content(
    model=MODEL_ID,
    contents=prompt_text,
    config=types.GenerateContentConfig(
        # Requesting both text (optional) and image output
        response_modalities=['Text', 'Image'] 
    )
)

print(f"--- Processing Response for Model: {MODEL_ID} ---")

# Iterate through all parts of the model's response
for i, part in enumerate(response.candidates[0].content.parts):
    if part.text is not None:
        print("\n[Text Output]")
        print(part.text)
        
    elif part.inline_data is not None:
        print(f"\n[Image Output - Part {i+1}]")
        
        # 1. Check MIME type (optional, but good practice)
        mime_type = part.inline_data.mime_type
        print(f"MIME Type: {mime_type}")
        
        # 2. Extract and open the image using the built-in as_image() method
        try:
            generated_image = part.as_image()
            
            # 3. Save the image to a file in your project directory
            filename = f"generated_pig_with_wings_{i}.png"
            generated_image.save(filename)
            print(f"Image successfully saved as: {filename}")
            
            # Optional: Display the image (works best in environments like Jupyter, but may open a new window in VS Code)
            # generated_image.show()
            
        except Exception as e:
            print(f"Could not process or save image data: {e}")