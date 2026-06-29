from typing import Any, Dict, Protocol

class DomainModule(Protocol):
    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        ...

class DomainLogicManager:
    """
    Manages pluggable domain-specific logic modules.
    Requirement 4.5: Use modular plug-in logic units.
    """
    
    def __init__(self):
        self._modules: Dict[str, DomainModule] = {}

    def register_module(self, name: str, module: DomainModule):
        self._modules[name] = module

    def execute_logic(self, domain_name: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if domain_name not in self._modules:
            raise ValueError(f"Domain module '{domain_name}' not found")
        return self._modules[domain_name].execute(inputs)
