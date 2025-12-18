from typing import Annotated,List
from pydantic import BaseModel, StringConstraints,field_validator
from app.config import settings

class Textpair(BaseModel):
    sentence1: Annotated[
        str, 
        StringConstraints(strip_whitespace=True, min_length=1, max_length=settings.MAX_SENTENCE_LENGTH)
    ]
    sentence2: Annotated[
        str, 
        StringConstraints(strip_whitespace=True, min_length=1, max_length=settings.MAX_SENTENCE_LENGTH)
    ]

    @field_validator("sentence1","sentence2")
    def validate(cls,v:str)->str:
        if any(ord(c) < 32 and c not in ("\t","\n","\r") for c in v):
            raise ValueError("Sentence contains invalid control characters")
        
        return v
    


class SentenceEmbedding(BaseModel):
    text:str
    embedding:List[float]
    norm:float

class SimilarityResponse(BaseModel):
    sentence1 : SentenceEmbedding
    sentence2 : SentenceEmbedding
    cosine_similarity : float
    dot_product:float
    euclidean_distance:float
    manhatten_distance:float

class HealthResponse(BaseModel):
    status:str
    model_loaded : bool
    env : str

