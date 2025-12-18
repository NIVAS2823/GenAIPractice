from langchain_google_genai import ChatGoogleGenerativeAI, Modality
from langchain_core.messages import AIMessage # Changed from langchain.messages
from IPython.display import Image, display

import base64

llm = ChatGoogleGenerativeAI(
    model='gemini-2.5-flash-image',
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2
)

message = {
    "role": "user",
    "content": "Generate a photorealistic image of a cuddly dog wearing a hat",
}

# 🐛 FIX: Change Modality.Image to Modality.image
response = llm.invoke(
    [message],
    response_modalities=[Modality.TEXT,"IMAGE"],
)

def get_image_base64(response: AIMessage) -> str: # Changed return type hint
    """Extracts the base64 part of the image URL from the AIMessage response."""
    # The image URL is part of a dict in the content list
    image_block = next(
        block
        for block in response.content
        if isinstance(block, dict) and block.get("image_url")
    )
    
    # Extract the base64 part which is after the comma (e.g., data:image/jpeg;base64,...)
    return image_block["image_url"].get("url").split(",")[-1]

image_base64 = get_image_base64(response)

# Display the image
display(Image(data=base64.b64decode(image_base64), width=300)) # Reduced width for common display