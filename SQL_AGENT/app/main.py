from fastapi import FastAPI,HTTPException
from fastapi.responses import JSONResponse


from app.agent import sql_agent_executor
from app.models import SQLAgentResponse,SQLQueryRequest
from app.logger import get_logger


logger = get_logger("API")

app = FastAPI(title = "SQL AGENT")


@app.post('/query',response_model=SQLAgentResponse)
async def query_sql_agent(request:SQLQueryRequest):
    logger.info(f"Received Query : {request.query}")

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
    
    