
from fastapi import FastAPI
from api.v1.routes import router as v1_router
from config import settings


import uvicorn


app = FastAPI(
    title=settings.project_name,
    docs_url="/api/docs",
    openapi_url="/api/openapi.json",
)

app.include_router(v1_router, prefix="/api/v1")


#
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, log_level="info")
