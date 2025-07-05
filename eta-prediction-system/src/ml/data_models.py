from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

class LocationPoint(BaseModel):
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)

class ETARequest(BaseModel):
    origin: LocationPoint
    destination: LocationPoint
    vehicle_type: str = Field(..., pattern="^(car|bike|truck)$")
    departure_time: Optional[datetime] = None
    route_preferences: Optional[str] = "fastest"

class ETAResponse(BaseModel):
    estimated_time_seconds: int
    confidence_score: float
    route_distance_km: float
    prediction_timestamp: datetime
    model_version: str
