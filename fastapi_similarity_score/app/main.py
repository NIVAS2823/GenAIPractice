import logging
from typing import Dict
from fastapi import FastAPI,HTTPException,Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter,_rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from starlette.concurrency import run_in_threadpool


from app.config import settings
from app.logging_config import setup_logging
from app.middleware     import BodySizeLimitMiddleware
from  app.models import Textpair,SentenceEmbedding,SimilarityResponse,HealthResponse
from app.similarity import (
    compute_embeddings,l2_norm,cosine_similarity,
    dot_product,euclidean_distance,manhattan_distance,MODEL_LOADED)



setup_logging()
logger = logging.getLogger(__name__)


limiter = Limiter(key_func=get_remote_address)

app = FastAPI(title=settings.APP_NAME)

app.state.limiter = limiter

app.add_middleware(
    CORSMiddleware,
    allow_origins = settings.ALLOWED_ORIGINS,
    allow_methods = ["*"],
    allow_headers = ["*"],
    allow_credentials = True
)



from slowapi.middleware import SlowAPIMiddleware

app.add_middleware(SlowAPIMiddleware)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request:Request,exc:RequestValidationError):
    logger.warning("Validation error",extra={"erros":exc.errors()})
    return JSONResponse(
        status_code=422,
        content={"detail":exc.errors()}
    )

@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request:Request,exc:RateLimitExceeded):
    return _rate_limit_exceeded_handler(request,exc)


@app.exception_handler(Exception)
async def generic_exception_handler(request:Request,exc:Exception):
    logger.exception("Unhandled Exception")
    return JSONResponse(
        status_code=500,
        content={"detail":"Internal Server Error"}
    )


@app.get("/health",response_model=HealthResponse,tags=["health"])
async def healthcheck():
    return HealthResponse(
        status="ok" if MODEL_LOADED else "degraded",
        model_loaded=MODEL_LOADED,
        env=settings.APP_ENV,
    )


@app.post("/similarity",response_model=SimilarityResponse,tags=["similarity"])
async def similarity(payload:Textpair,request:Request):
    """
    Compute semantic similarity and return:
    
    -sentence embeddings
    -norms
    -cosine similarity
    -dot product
    -euclidean & manhattan distance
    """

    if not MODEL_LOADED:
        raise HTTPException(status_code=503,detail="MODEL NOT LOADED")
    
    try:
        emb1,emb2 = await run_in_threadpool(
            compute_embeddings,payload.sentence1,payload.sentence2
        )

        norm1 = l2_norm(emb1)
        norm2 = l2_norm(emb2)

        cos_sim = cosine_similarity(emb1,emb2)
        dot = dot_product(emb1,emb2)
        euc  = euclidean_distance(emb1,emb2)
        man = manhattan_distance(emb1,emb2)

        logger.info(
            "similarity_computed",
            extra={
                "sentence1_len":len(payload.sentence1),
                "sentence2_len":len(payload.sentence2),
                "cosine_similarity":cos_sim,
                "client_ip":get_remote_address(request),
            },
        )


        return SimilarityResponse(
            sentence1=SentenceEmbedding(
                text=payload.sentence1,
                embedding=emb1.tolist(),
                norm=norm1
            ),
            sentence2=SentenceEmbedding(
                text=payload.sentence2,
                embedding=emb2.tolist(),
                norm=norm2,
            ),

            cosine_similarity=round(cos_sim,6),
            dot_product=round(dot,6),
            euclidean_distance=round(euc,6),
            manhatten_distance=round(man,6),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error Computing similarity")
        raise HTTPException(status_code=500,detail="Failed to compute similarity")
