import time
import numpy as np
from typing import List, Optional, Dict, Any
import uuid

class Intent:
    """
    Represents a discrete Unit of Intent (Goal/Task) in Working Memory.
    """
    def __init__(self, 
                 description: str, 
                 target_concept_name: Optional[str] = None,
                 target_vector: Optional[np.ndarray] = None,
                 constraints: List[Any] = [],
                 logic_gates: List[Dict] = [],
                 priority: float = 1.0,
                 timeout_steps: int = 20):
        
        self.id = str(uuid.uuid4())[:8]
        self.description = description
        self.target_concept_name = target_concept_name
        self.target_vector = target_vector
        self.constraints = constraints
        self.logic_gates = logic_gates
        self.priority = priority
        self.created_at = time.time()
        self.steps_taken = 0
        self.timeout_steps = timeout_steps
        self.status = "active" # active, completed, failed, suspended

    def __repr__(self):
        return f"<Intent '{self.description}' ({self.status})>"

class WorkingMemory:
    """
    The Executive Workspace (Left Brain Core).
    Holds the 'Stack' of active intentions and manages context switching.
    """
    def __init__(self):
        self.intent_stack: List[Intent] = []
        self.completed_log: List[Intent] = []
        self.context_buffer: List[str] = []
        self.max_context = 64
        self.max_intents = 32
        self._freq: Dict[str, int] = {}

    def add_context(self, concept: str):
        self.context_buffer.append(concept)
        self._freq[concept] = self._freq.get(concept, 0) + 1
        if self._freq[concept] > 8:
            for i in range(len(self.context_buffer)-1, -1, -1):
                if self.context_buffer[i] == concept:
                    self.context_buffer.pop(i)
                    break
            self._freq[concept] = 8
        if len(self.context_buffer) > self.max_context:
            dropped = self.context_buffer.pop(0)
            self._freq[dropped] = max(0, self._freq.get(dropped, 0) - 1)
        print(f"[WM] 🧠 Context Update: {self.context_buffer[-8:]}")

    def clear_context(self):
        """Clears context buffer and intent stack."""
        self.context_buffer = []
        self.intent_stack = []
        print("[WM] 🧹 Context Cleared.")
        
    def get_context_vector(self, engine) -> Optional[np.ndarray]:
        """Computes a weighted vector of current context."""
        if not self.context_buffer:
            return None
        
        # Weighted towards recent
        vec = np.zeros(engine.l2_dim)
        total_weight = 0
        for i, word in enumerate(reversed(self.context_buffer)):
            v = engine.get_concept_vector(word)
            if v is not None:
                weight = 1.0 / (i + 1) # 1, 0.5, 0.33...
                vec += v * weight
                total_weight += weight
        
        if total_weight > 0:
            return vec / total_weight
        return None

    def add_intent(self, intent: Intent):
        """Pushes a new intent onto the stack (Focus Shift)."""
        # If there's an active intent, suspend it? 
        # For now, just simple stack behavior. Top is active.
        if len(self.intent_stack) >= self.max_intents:
            self.intent_stack.pop(0)
        self.intent_stack.append(intent)
        print(f"[WM] ➕ Added Intent: {intent.description}")
        
    def pop_intent(self) -> Optional[Intent]:
        """Removes the current intent (Completion/Failure)."""
        if self.intent_stack:
            intent = self.intent_stack.pop()
            print(f"[WM] ➖ Popped Intent: {intent.description}")
            return intent
        return None
        
    def get_current_intent(self) -> Optional[Intent]:
        """Peeks at the active intent."""
        if self.intent_stack:
            return self.intent_stack[-1]
        return None
        
    def complete_intent(self, intent: Intent, success: bool = True):
        """Marks intent as complete/failed and moves to log."""
        intent.status = "completed" if success else "failed"
        self.completed_log.append(intent)
        # Ensure it's removed from stack if it was top
        if self.intent_stack and self.intent_stack[-1].id == intent.id:
            self.pop_intent()
            
    def clear(self):
        self.intent_stack = []
        self.completed_log = []

class Planner:
    """
    Minimal Planner: Intent → Plan → Act loop with progress, retries, and logging.
    """
    def __init__(self, max_retries: int = 2):
        self.max_retries = max_retries
        self.logs: List[str] = []
    
    def decompose(self, goal: str) -> List[str]:
        """
        Simple rule-based decomposition for demonstration.
        """
        g = goal.lower()
        if "make tea" in g or ("tea" in g and "plan" in g):
            return ["boil water", "pour water", "steep leaves"]
        if "clean room" in g:
            return ["pick items", "sweep floor", "take trash"]
        # Fallback small steps
        return ["describe goal", "choose action", "act", "review"]
    
    def execute(self, steps: List[str], act_fn) -> bool:
        """
        Execute steps with retry policy and basic progress criteria.
        act_fn(step) should return True/False for success.
        """
        for step in steps:
            success = False
            tries = 0
            while not success and tries <= self.max_retries:
                self.logs.append(f"Executing: {step} (try {tries+1})")
                success = bool(act_fn(step))
                if not success:
                    self.logs.append(f"Retrying: {step}")
                tries += 1
            if not success:
                self.logs.append(f"Failed: {step}")
                return False
            self.logs.append(f"Completed: {step}")
        return True
