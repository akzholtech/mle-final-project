from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
import logging
from dotenv import load_dotenv

from src.mlflow_loader import load_recommender_from_mlflow
from src.schemas import RecommendRequest, RecommendResponse
from src.service import RecommendationService

logger = logging.getLogger("uvicorn.info")
logger.setLevel(logging.INFO)

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_dotenv()
    logger.info("Starting!")
    recommender = load_recommender_from_mlflow()
    app.state.recommendation_service = RecommendationService(recommender)
    logger.info("Ready!")
    yield
    logger.info("Stopping!")


app = FastAPI(
    title="ALS Recommendation Service",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def heakth() -> dict:
    return {"status": "ok"}


@app.post("/recommend", response_model=RecommendResponse)
def recommend(payload: RecommendRequest):
    """
    test
    """
    try:
        service: RecommendationService = app.state.recommendation_service
        result = service.recommend(user_id=payload.user_id, top_k=payload.top_k)
        return RecommendResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
