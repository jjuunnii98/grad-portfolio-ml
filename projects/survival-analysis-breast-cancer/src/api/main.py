from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI

from src.api.routes.predict import get_inference_bundle, router as predict_router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    FastAPI lifespan hook.

    On startup:
    - loads the pre-trained penalized Cox inference bundle into memory
    - reduces first-request latency by warming the cache

    On shutdown:
    - emits a clean shutdown log
    """
    get_inference_bundle()
    logger.info("Inference bundle loaded successfully during startup.")
    yield
    logger.info("Breast Cancer Survival Risk API shutdown complete.")


app = FastAPI(
    title="Breast Cancer Survival Risk API",
    description=(
        "Production-oriented FastAPI service for breast cancer survival-risk "
        "inference using a penalized Cox proportional hazards model."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(predict_router, tags=["survival-risk"])