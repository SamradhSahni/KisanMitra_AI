from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


# ── Chat ──────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    query:      str             = Field(..., min_length=1, max_length=1000,
                                       description="Hindi farmer question")
    state:      str             = Field(default="UTTAR PRADESH",
                                       description="Farmer's state")
    crop:       str             = Field(default="others",
                                       description="Crop name")
    intent:     Optional[str]   = Field(default=None,
                                       description="Override intent detection")
    session_id: Optional[str]   = Field(default=None,
                                       description="Session ID for tracking")
    use_rag:    bool            = Field(default=True,
                                       description="Enable RAG retrieval")

    class Config:
        json_schema_extra = {
            "example": {
                "query":      "मक्का में फॉल आर्मी वर्म कीट का नियंत्रण कैसे करें?",
                "state":      "UTTAR PRADESH",
                "crop":       "maize (makka)",
                "session_id": "user_123",
                "use_rag":    True,
            }
        }


class PassageItem(BaseModel):
    answer:    str
    intent:    str
    crop:      str
    state:     str
    rrf_score: float


class ChatResponse(BaseModel):
    response:       str
    intent:         str
    rag_used:       bool
    passages:       List[PassageItem]
    latency_ms:     int
    retrieval_ms:   int
    generation_ms:  int
    session_id:     Optional[str]
    query:          str
    state:          str
    crop:           str
    timestamp:      datetime = Field(default_factory=datetime.utcnow)


# ── MSP ───────────────────────────────────────────────────────────────
class MSPResponse(BaseModel):
    crop:           str
    msp_price:      Optional[float]
    unit:           str
    season:         str
    year:           str
    source:         str
    found:          bool


# ── Feedback ──────────────────────────────────────────────────────────
class FeedbackRequest(BaseModel):
    session_id:  str
    query:       str
    response:    str
    rating:      int            = Field(..., ge=1, le=5,
                                       description="Rating 1-5")
    comment:     Optional[str]  = Field(default=None, max_length=500)
    intent:      Optional[str]  = None
    state:       Optional[str]  = None
    crop:        Optional[str]  = None

    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "user_123",
                "query":      "मक्का में कीट नियंत्रण",
                "response":   "इमामेक्टिन बेंज़ोइड का छिड़काव करें",
                "rating":     4,
                "comment":    "अच्छी जानकारी",
            }
        }


class FeedbackResponse(BaseModel):
    success:    bool
    message:    str
    feedback_id: Optional[int]


# ── Health ────────────────────────────────────────────────────────────
class HealthResponse(BaseModel):
    status:     str
    model:      str
    rag:        str
    version:    str
    uptime_s:   float