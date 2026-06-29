from typing import Any, Callable, Dict, List, Tuple


class ProbabilisticLogic:
    """
    Minimal probabilistic combinators.
    """
    @staticmethod
    def bayes(prior: float, likelihood: float, evidence: float = None) -> float:
        prior = max(1e-9, min(1.0, prior))
        likelihood = max(1e-9, min(1.0, likelihood))
        if evidence is None:
            # Assume binary evidence space; normalize with law of total probability using 0.5 prior for not-hyp
            evidence = prior * likelihood + (1 - prior) * (1 - likelihood)
        return (prior * likelihood) / max(1e-9, evidence)

    @staticmethod
    def noisy_or(ps: List[float]) -> float:
        q = 1.0
        for p in ps:
            q *= (1 - max(0.0, min(1.0, p)))
        return 1 - q

    @staticmethod
    def and_prod(ps: List[float]) -> float:
        prod = 1.0
        for p in ps:
            prod *= max(0.0, min(1.0, p))
        return prod

class BayesianNetwork:
    def __init__(self) -> None:
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self._cache: Dict[Tuple[str, bool, Tuple[Tuple[str, bool], ...]], float] = {}

    def add_node(self, name: str, parents: List[str], cpt: Callable[[List[bool]], float]) -> None:
        self.nodes[name] = {"parents": parents, "cpt": cpt}

    def query(self, target: str, evidence: Dict[str, bool]) -> float:
        names = list(self.nodes.keys())
        unknowns = [n for n in names if n not in evidence]
        key_true: Tuple[str, bool, Tuple[Tuple[str, bool], ...]] = (target, True, tuple(sorted(evidence.items())))
        key_false: Tuple[str, bool, Tuple[Tuple[str, bool], ...]] = (target, False, tuple(sorted(evidence.items())))
        if key_true in self._cache and key_false in self._cache:
            pt = self._cache[key_true]
            pf = self._cache[key_false]
            return pt / max(1e-12, (pt + pf))
        def prob_assignment(assign: Dict[str, bool]) -> float:
            p = 1.0
            for n in names:
                node = self.nodes[n]
                parent_vals: List[bool] = [assign[pn] for pn in node["parents"]]
                p_true: float = node["cpt"](parent_vals)
                p *= p_true if assign[n] else (1 - p_true)
            return p
        def enumerate_sum(target_val: bool) -> float:
            total = 0.0
            import itertools
            if len(unknowns) == 0:
                assign = dict(evidence)
                assign[target] = target_val
                return prob_assignment(assign)
            for bits in itertools.product([False, True], repeat=len(unknowns)):
                assign = dict(evidence)
                for i, n in enumerate(unknowns):
                    assign[n] = bits[i]
                assign[target] = target_val
                total += prob_assignment(assign)
            return total
        p_true = enumerate_sum(True)
        p_false = enumerate_sum(False)
        self._cache[key_true] = p_true
        self._cache[key_false] = p_false
        return p_true / max(1e-12, (p_true + p_false))
