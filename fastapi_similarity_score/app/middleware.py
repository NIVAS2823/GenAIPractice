from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from app.config import settings

class BodySizeLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app):
        super().__init__(app)
        self.max_body_size = settings.MAX_REQUEST_BYTES

    async def dispatch(self, request, call_next):
        body = await request.body()
        if len(body)> self.max_body_size:
            return JSONResponse(
                status_code=413,
                content={"detail":"Request Body to large"}
            )
        
        request._body = body
        return await call_next(request)