from datetime import timedelta
from pydantic_settings import BaseSettings

class Settings(BaseSettings):

    JWT_SECRET_KEY:str = 'CHANGE_ME_SUPER_SECRET'
    JWT_ALGORITHM : str = 'HS256'
    JWT_ACCESS_TOKEN_EXPIRE_TIME :int = 30
