from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class KnowledgeQueryResult(BaseModel):
    """
    Standardized result from any knowledge source.
    """
    source_id: str
    fact_id: str
    content: Any
    confidence: float
    metadata: Dict[str, Any] = Field(default_factory=dict)

class KnowledgeQueryEngine:
    """
    Interfaces with external structured knowledge sources.
    Requirement 4.1: Query external structured knowledge sources.
    Requirement 4.4: Separation between reasoning and knowledge.
    """
    
    def __init__(self, schema_version: str = "1.0"):
        self.schema_version = schema_version
        # In a real system, this would connect to databases/APIs
        self._knowledge_base = {
            "apple": {"category": "fruit", "edible": True, "color": ["red", "green"]},
            "run": {"category": "action", "energy_cost": "high"},
            "physics_gravity": {"value": 9.81, "unit": "m/s^2"}
        }
        self._cache: Dict[str, KnowledgeQueryResult] = {}
        self.query_log: List[Dict[str, Any]] = []

    def query(self, concept_key: str) -> Optional[KnowledgeQueryResult]:
        """
        Retrieves knowledge for a specific concept.
        Returns None if knowledge is missing (Knowledge Gap).
        """
        import time
        concept_key = concept_key.lower()
        self.query_log.append({"concept": concept_key, "timestamp": time.time()})
        
        if concept_key in self._cache:
            return self._cache[concept_key]

        data = self._knowledge_base.get(concept_key)
        if data:
            res = KnowledgeQueryResult(
                source_id="internal_kb_mock",
                fact_id=f"fact_{concept_key}",
                content=data,
                confidence=1.0,
                metadata={"schema": self.schema_version}
            )
            self._cache[concept_key] = res
            return res
        return None

    def update_knowledge(self, concept_key: str, data: Any):
        """Update or add a new fact to the knowledge base."""
        self._knowledge_base[concept_key.lower()] = data
        # Invalidate cache
        if concept_key.lower() in self._cache:
            del self._cache[concept_key.lower()]

    def query_concepts(self, concepts: List[str]) -> Dict[str, Optional[KnowledgeQueryResult]]:
        """Batch query for multiple concepts."""
        return {c: self.query(c) for c in concepts}
