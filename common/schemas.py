from pydantic import BaseModel
from typing import Dict, Any, Optional, List

# Schema definitions for API events, feature vectors, ML/NLP scores, fusion scores, and risk decisions.
class ApiEvent(BaseModel):
    ts: str
    endpoint: str
    method: str
    headers: Dict[str, str] = {}
    payload: Dict[str, Any] = {}
    status_code: Optional[int] = None
    latency_ms: Optional[float] = None
    source_ip: Optional[str] = None
    user_id: Optional[str] = None
    hash_id: Optional[str] = None

class FeatureVector(BaseModel):
    ts: str
    endpoint: str
    numeric_feats: Dict[str, float]
    text_fields: Dict[str, str]
    hash_id: str

class MlScore(BaseModel):
    hash_id: str
    ml_score: float
    aux: Dict[str, Any] = {}

class NlpScore(BaseModel):
    hash_id: str
    nlp_score: float
    topics: List[str] = []
    aux: Dict[str, Any] = {}

class FusionScore(BaseModel):
    hash_id: str
    final_score: float
    alpha: float
    beta: float
    ml_score: float
    nlp_score: float

class RiskDecision(BaseModel):
    hash_id: str
    level: str
    final_score: float
    context: Dict[str, Any]
    explanation: Optional[Dict[str, Any]] = None
    message: Optional[str] = None