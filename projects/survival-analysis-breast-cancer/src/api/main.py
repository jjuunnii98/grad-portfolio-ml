from fastapi import FastAPI

from src.api.routes.predict import router as predict_router

app = FastAPI(
    title="Breast Cancer Survival Risk API",
    description="FastAPI service for survival-risk inference on breast cancer clinical features.",
    version="0.1.0",
)

app.include_router(predict_router)