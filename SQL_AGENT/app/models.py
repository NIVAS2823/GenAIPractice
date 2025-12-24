from pydantic import BaseModel,Field


class SQLQueryRequest(BaseModel):
    query:str = Field(...,min_length=5,max_length=500,description="Natural language question for sql agent")



class SQLAgentResponse(BaseModel):
    success : bool
    answer : str


class LoginRequest(BaseModel):
    username:str
    password:str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"