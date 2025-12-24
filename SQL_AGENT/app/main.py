from fastapi import FastAPI,HTTPException,Depends
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordBearer
from app.auth import verify_password,create_access_token,decode_token
from app.models import LoginRequest,TokenResponse
from app.users import fake_users_db
from app.agent import sql_agent_executor
from app.models import SQLAgentResponse,SQLQueryRequest
from app.logger import get_logger



logger = get_logger("API")


oauth2_scheme = OAuth2PasswordBearer(token_url="auth/login")

app = FastAPI(title = "SQL AGENT")

@app.post('/auth/login',response_model=TokenResponse)
def login(request:LoginRequest):
    user = fake_users_db(request.username)

    if not user or not verify_password(request.password,user['hashed_password']):
        raise HTTPException(status_code=401,detail='Invalid Credentials')
    
    token = create_access_token({"sub":request.username})

    return TokenResponse(access_token=token)

def get_current_user(token:str=Depends(oauth2_scheme)):
    payload = decode_token(token)
    return payload["sub"]

@app.post('/query',response_model=SQLAgentResponse)
async def query_sql_agent(request:SQLQueryRequest):
    logger.info(f"Received Query : {request.query}")
    # logger.info(f"User = {user}")

    try:
        result = sql_agent_executor.invoke({"input":request.query})

        answer = result.get("output")
        if not answer:
            raise ValueError("Empty response from agent")
        
        logger.info("Query processes successfully")


        return SQLAgentResponse(
            success= True,
            answer=answer
        )
    
    except ValueError as e:
        logger.info(f"Validation error : {e}")
        raise HTTPException(status_code=400,detail=str(e))
    
    except Exception as e:
        logger.info("Unhandled error in SQL Agent")
        raise HTTPException(status_code=500,
                            detail="Internal Server Error while processing query")
    
