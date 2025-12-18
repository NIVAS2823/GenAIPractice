import os
from typing import List

class Settings:
    APP_NAME:str = os.getenv("APP_NAME","Sentence Similarity API")
    APP_ENV :str = os.getenv("APP_ENV","dev")


    ALLOWED_ORIGINS:List[str] = os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost,http:localhost:3000"
    ).split(",")

    MAX_SENTENCE_LENGTH :int = int(os.getenv("MAX_SENTENCE_LENGTH","512"))
    MAX_REQUEST_BYTES :int = int(os.getenv("MAX_REQUEST_BYTES",str(1024 * 16)))

    RATE_LIMIT :str = os.getenv("RATE_LIMIT","60/minute")

    LOG_LEVEL:str = os.getenv("LOG_LEVEL","INFO")

settings = Settings()