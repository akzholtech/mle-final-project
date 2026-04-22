from pydantic import BaseModel, Field
from typing import List, Union

class RecommendRequest(BaseModel):
    user_id: Union[int, str]
    top_k: int = Field(default=10, ge=1, le=100)


class RecommendResponse(BaseModel):
    user_id: Union[int, str]
    recommendations: List[Union[int, str]]
    scores: List[float]
    source: str