from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel, Field

class Entity(BaseModel):
    entity: str
    type: str
    start: Optional[int] = None
    end: Optional[int] = None
    confidence: Optional[float] = None

class DocPII(BaseModel):
    text: str
    entities: List[Entity] = Field(default_factory=list)
