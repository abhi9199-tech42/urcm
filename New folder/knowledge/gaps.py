from typing import List, Set
from ..models.reasoning import ReasoningDecision
from .engine import KnowledgeQueryEngine

class KnowledgeGapDetector:
    """
    Identifies missing information in reasoning decisions.
    Requirement 4.3: Explicitly identify knowledge gaps.
    """
    
    def __init__(self, query_engine: KnowledgeQueryEngine):
        self.engine = query_engine

    def detect_gaps(self, decision: ReasoningDecision) -> List[str]:
        """
        Scans the selected path for concepts that lack definition in the knowledge base.
        Returns a list of missing concept IDs/names.
        """
        missing_concepts = set()
        
        # 1. Extract all concepts from the selected path
        path_concepts = set()
        for step in decision.selected_path.steps:
            for primitive in step.semantic_payload:
                path_concepts.add(primitive.concept)

        # 2. Check against Knowledge Engine
        results = self.engine.query_concepts(list(path_concepts))
        
        for concept, result in results.items():
            if result is None:
                missing_concepts.add(concept)
                
        return list(missing_concepts)
