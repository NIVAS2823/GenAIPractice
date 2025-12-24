from datetime import datetime,timedelta
from jose import jwt,JWTError
from passlib.context import CryptContext
from fastapi import HTTPException,status

from app.config import Settings

pwd_context = CryptContext(schemes=["bcrypt"],deprecated="auto")


def hash_password(password:str)->str:
    return pwd_context.hash(password)

def verify_password(plain:str,hashed:str)->bool:
    return pwd_context.verify(plain,hashed)


def create_access_token(data:dict)->str:
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(
        minutes=Settings.JWT_ACCESS_TOKEN_EXPIRE_TIME
    )

    to_encode.update({"exp":expire})

    return jwt.encode(
        to_encode,
        Settings.JWT_SECRET_KEY,
        algorithm=Settings.JWT_ALGORITHM
    )


def decode_token(token:str)->dict:
    try:
        payload = jwt.decode(token,Settings.JWT_SECRET_KEY,algorithms=Settings.JWT_ALGORITHM)
        return payload
    
    except JWTError:
        raise HTTPException(status_code.HTTP_401_UNAUTHORIZED,detail="Invalid or expired token")
    


    
