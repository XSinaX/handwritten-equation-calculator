import logging
import os
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI

from api.app.routers.calculation import router as calculation_router
from api.app.services.model_loader import loaded_model_count, preload_all_models

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

# Optional: cap PyTorch's CPU threads on small/shared machines (e.g. TORCH_THREADS=2)
if os.environ.get("TORCH_THREADS"):
    torch.set_num_threads(int(os.environ["TORCH_THREADS"]))


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Set PRELOAD_MODELS=0 to skip (e.g. for faster --reload during development)
    if os.environ.get("PRELOAD_MODELS", "1") != "0":
        preload_all_models()
    yield


app = FastAPI(lifespan=lifespan)


app.include_router(calculation_router)


@app.get("/")
def root():
    return {"message": "API is running"}


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": loaded_model_count()}
