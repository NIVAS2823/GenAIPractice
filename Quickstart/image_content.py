import os
from google import genai
from google.genai import types

# ⚠️ STEP 1: Specify the path to your image file
IMAGE_FILE_PATH = r"C:\Users\NIVAS\Personal\Agentic_Ai\Gen_ai_practice\my_photo.jpg" 

# --- Function to read image bytes ---
def read_image_bytes(image_path):
    """Reads an image file and returns the raw bytes."""
    # We just read the file as binary ('rb'). No manual Base64 encoding needed here.
    with open(image_path, "rb") as image_file:
        return image_file.read()

# --- Main script ---
try:
    # Initialize the client
    # Note: 'media_resolution' is removed as it caused the validation error.
    client = genai.Client(http_options={'api_version': 'v1alpha'})

    # STEP 2: Get the raw bytes of the image
    image_bytes = read_image_bytes(IMAGE_FILE_PATH)
    
    # STEP 3: Create the contents list for the API call
    response = client.models.generate_content(
        model="gemini-3-pro-preview",
        contents=[
            types.Content(
                parts=[
                    types.Part(text="What is in this image? Provide a detailed description."),
                    types.Part(
                        inline_data=types.Blob(
                            # ⚠️ Ensure the MIME type matches your file (jpeg/png)
                            mime_type="image/jpeg", 
                            # The SDK expects raw bytes here. It handles the encoding internally.
                            data=image_bytes
                        )
                    )
                ]
            )
        ]
    )
    
    # STEP 4: Print the model's response
    print("\n--- Model Response ---")
    print(response.text)

except FileNotFoundError:
    print(f"Error: The image file was not found at {IMAGE_FILE_PATH}. Please check the path.")
except Exception as e:
    print(f"An error occurred: {e}")