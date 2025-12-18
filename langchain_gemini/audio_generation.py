from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-tts")

response = llm.invoke(
    "Please say The quick brown fox jumps over the lazy dog",
    generation_config=dict(response_modalities=["AUDIO"]),
)

# Base64 encoded binary data of the audio
wav_data = response.additional_kwargs.get("audio")
with open("output.wav", "wb") as f:
    f.write(wav_data)