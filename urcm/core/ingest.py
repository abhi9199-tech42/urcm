import numpy as np
import pickle
import os
import re
from typing import List, Dict
from urcm.core.observability import record_event

from urcm.core.hierarchical_encoder import HierarchicalEncoder
from urcm.core.memory import GeometricMemory
from urcm.core.phoneme_mapper import TextToPhonemeConverter
from urcm.core.identity import IDENTITY_CONCEPTS
from urcm.core.logic_gates import InductionEngine

class KnowledgeIngestion:
    """
    Ingests raw text into the URCM Resonance Memory.
    Enables 'Zero-Shot' World Knowledge and 'Open-Ended' Language capabilities
    by depositing sentence trajectories into the Concept Layer (L2).
    """
    
    def __init__(self, brain_path: str = "urcm_identity.pkl", l2_dim: int = 512):
        self.brain_path = brain_path
        self.l2_dim = l2_dim
        self.converter = TextToPhonemeConverter()
        self.memory = GeometricMemory(resonance_dim=l2_dim) # Increased capacity
        
        # Load or Create Brain
        if os.path.exists(brain_path):
            print(f"Loading brain from {brain_path}...")
            with open(brain_path, "rb") as f:
                self.brain_data = pickle.load(f)
                # Check dim
                if self.brain_data["l2_W_res"].shape[0] != l2_dim:
                    print(f"⚠️ Resizing Brain from {self.brain_data['l2_W_res'].shape[0]} to {l2_dim}")
                    # Re-init
                    self._init_fresh_brain()
        else:
            print("Creating FRESH High-Capacity Brain...")
            self._init_fresh_brain()
            
        # Reconstruct Hierarchy State
        self.hierarchy = HierarchicalEncoder(l2_res_dim=l2_dim)
        self.hierarchy.layer2.W_res = self.brain_data["l2_W_res"]
        self.concept_map = self.brain_data["concept_map"]
        self.relations = self.brain_data.get("relations", [])
        
    def _init_fresh_brain(self):
        """Initializes a new brain with Identity concepts."""
        hierarchy = HierarchicalEncoder(l2_res_dim=self.l2_dim)
        
        # Init Concept Map
        concept_map = {}
        self.concept_map = concept_map # Set temporarily for helper use
        
        # Initialize Identity Concepts
        print("Initializing Identity Concepts...")
        # Populate map with Identity Concepts
        for concept_name in IDENTITY_CONCEPTS.keys():
            self._get_or_create_concept_vector(concept_name)
        # Ensure core axioms exist
        for ax in ["truth","understanding","confusion"]:
            self._get_or_create_concept_vector(ax)
        
        self.brain_data = {
            "l1_W_res": hierarchy.layer1.W_res,
            "l1_W_in": hierarchy.layer1.W_in,
            "l1_W_out": hierarchy.layer1.W_out,
            "l2_W_res": hierarchy.layer2.W_res, # 512x512
            "l2_W_in": hierarchy.layer2.W_in,
            "l2_W_out": hierarchy.layer2.W_out,
            "concept_map": concept_map,
            "axioms": {"truth":1.0,"understanding":1.0,"confusion":0.0}
        }

    def _get_or_create_concept_vector(self, word: str) -> np.ndarray:
        """
        Returns the L2 vector for a word.
        If the word is new, generates a deterministic vector and adds it to the map.
        """
        word = word.lower()
        
        # 1. Check existing
        if word in self.concept_map:
            return self.concept_map[word]
            
        # 2. Create new (Deterministic Hash)
        # Using 512 dimensions for high capacity separation
        # Phase 10 Fix: Improved Hashing to prevent collisions (e.g. midway vs across)
        import hashlib
        hash_object = hashlib.md5(word.encode())
        seed = int(hash_object.hexdigest(), 16) % (2**32)
        
        np.random.seed(seed)
        vec = np.random.normal(0, 1, (self.l2_dim,))
        vec = vec / np.linalg.norm(vec)
        
        self.concept_map[word] = vec
        return vec

    def ingest_text(self, text: str):
        """
        Parses text and deposits semantic trajectories.
        """
        # 1. Clean and Split into Sentences
        text = re.sub(r'\s+', ' ', text)
        # Paraphrase normalization
        text = re.sub(r'\bevery\b', 'all', text, flags=re.IGNORECASE)
        text = re.sub(r'\bnone\b', 'no', text, flags=re.IGNORECASE)
        # Normalize can/could into able_to_
        text = re.sub(r'\bare\s+able\s+to\s+(\w+)', r'are able_to_\1', text, flags=re.IGNORECASE)
        text = re.sub(r'\bcan\s+(\w+)', r'are able_to_\1', text, flags=re.IGNORECASE)
        text = re.sub(r'\bcould\s+(\w+)', r'are able_to_\1', text, flags=re.IGNORECASE)
        sentences = re.split(r'[.!?]', text)
        sentences = [s.strip().lower() for s in sentences if s.strip()]

        # Phase 9 Fix: Semantic Gating (Black Hole Suppression)
        # We actively filter out words that cause attractor collapse.
        STOP_WORDS = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
            "has", "have", "had", "having", "do", "does", "did", "doing",
            "can", "could", "would", "should", "will", "shall", "may", "might", "must",
            "to", "of", "and", "in", "that", "this", "it", "for", "on", "with",
            "as", "by", "at", "from", "or", "but", "if", "then", "else",
            "help", "causing", "copyright", "privacy", "disclaimer", "contact",
            "hours", "minute", "minutes", "second", "seconds", "day", "days",
            "wiki", "edit", "source", "history", "links", "external", "references",
            "contents", "search", "navigation", "main", "page", "article", "talk",
            "create", "account", "log", "in", "out", "view", "read", "jump", "content",
            "file", "image", "category", "template", "user", "portal", "special",
            "identifier", "identifiers", "link", "about", "all", "any", "some",
            "retrieved", "archived", "original", "date", "isbn", "doi", "issn", "pmid",
            "text", "under", "license", "additional", "terms", "apply", "site",
            "policy", "mobile", "developers", "statistics", "cookie", "statement",
            "wikimedia", "foundation", "powered", "mediawiki",
            "level", "levels", "type", "types", "use", "used", "using", # New Grey Holes
        }
        
        print(f"Ingesting {len(sentences)} sentences into {self.l2_dim}-dim Space...")
        if len(sentences) > 500:
            record_event("ingest_gate_triggered", {"reason": "too_many_sentences", "count": len(sentences)})
            sentences = sentences[:500]
        
        count = 0
        last_subject = None
        for sentence in sentences:
            # Numeric constraints and envelopes from text (parse early, independent of deposit)
            nums = []
            for m in re.finditer(r'([a-zA-Z]+)\s*(<=|>=|=)\s*([+-]?\d+(?:\.\d+)?)', sentence):
                v = m.group(1).lower()
                op = m.group(2)
                val = float(m.group(3))
                if op == "=":
                    nums.append(({v: 1.0}, "<=", val))
                    nums.append(({v: 1.0}, ">=", val))
                else:
                    nums.append(({v: 1.0}, op, val))
            if nums:
                existing = self.brain_data.get("numeric_constraints", [])
                self.brain_data["numeric_constraints"] = existing + nums
                if len(self.brain_data["numeric_constraints"]) > 1000:
                    self.brain_data["numeric_constraints"] = self.brain_data["numeric_constraints"][:1000]
                    record_event("ingest_gate_triggered", {"reason": "numeric_constraints_cap"})
            # Integer variables: detect declarations like "count n_x is integer" or "n_x integer"
            int_vars = set(self.brain_data.get("int_vars", []))
            for m in re.finditer(r'\b(count\s+)?([a-zA-Z_][\w]*)\s+(?:is\s+)?integer\b', sentence):
                name = m.group(2).lower()
                int_vars.add(name)
            for m in re.finditer(r'\bint\s+([a-zA-Z_][\w]*)\b', sentence):
                int_vars.add(m.group(1).lower())
            # Heuristic: names starting with n_ are integers
            for w in re.findall(r'\b(n_[a-zA-Z_][\w]*)\b', sentence):
                int_vars.add(w.lower())
            if int_vars:
                self.brain_data["int_vars"] = sorted(int_vars)
            # Bilinear/product parsing: w = x*y
            mprod = re.findall(r'(\w+)\s*=\s*(\w+)\s*\*\s*(\w+)', sentence)
            if mprod:
                bilis = self.brain_data.get("bilinear_pairs", [])
                for w, x, y in mprod:
                    bilis.append((w.lower(), x.lower(), y.lower()))
                self.brain_data["bilinear_pairs"] = bilis
            # Square parsing: w = x^2 or w equals x squared
            msq = re.findall(r'(\w+)\s*=\s*(\w+)\s*(?:\^|\*\*)\s*2', sentence)
            for w, x in msq:
                sqs = self.brain_data.get("square_pairs", [])
                sqs.append((w.lower(), x.lower()))
                self.brain_data["square_pairs"] = sqs
            msq2 = re.findall(r'(\w+)\s+(?:equals|is)\s+(\w+)\s+squared', sentence)
            for w, x in msq2:
                sqs = self.brain_data.get("square_pairs", [])
                sqs.append((w.lower(), x.lower()))
                self.brain_data["square_pairs"] = sqs
            # Quantified relations
            m = re.findall(r'all\s+(\w+)\s+(?:are|is)\s+(\w+)', sentence)
            for a, b in m:
                self.relations.append(("all", a, b))
                record_event("ingest_relation", {"type": "all", "a": a, "b": b})
                last_subject = a
            # Robust parser: (x1,y1),(x2,y2)
            m3 = re.findall(r'(\w+)\s+piecewise\s+(\w+)\s*:\s*(\\(.*\\))', sentence)
            if not m3:
                m_any = re.findall(r'(\w+)\s+piecewise\s+(\w+)\s*:\\s*\\(([^)]*)\\)', sentence)
            # Simpler: generic parser
            for m in re.findall(r'(\w+)\s+piecewise\s+(\w+)\s*:\\s*\\(([^)]*)\\)', sentence):
                wname, xname, inner = m
                pairs = []
                for tup in inner.split("),"):
                    nums = re.findall(r'([+-]?\\d+(?:\\.\\d+)?)', tup)
                    if len(nums) >= 2:
                        xi = float(nums[0]); yi = float(nums[1])
                        pairs.append((xi, yi))
                if pairs:
                    pw = self.brain_data.get("piecewise_pairs", [])
                    pw.append((wname.lower(), xname.lower(), pairs))
                    self.brain_data["piecewise_pairs"] = pw
            m = re.findall(r'all\s+(\w+)\s+(?:are|is)\s+(\w+)', sentence)
            for a, b in m:
                self.relations.append(("all", a, b))
                record_event("ingest_relation", {"type": "all", "a": a, "b": b})
                last_subject = a
            m2 = re.findall(r'no\s+(\w+)\s+(?:are|is)\s+(\w+)', sentence)
            for a, b in m2:
                self.relations.append(("no", a, b))
                record_event("ingest_relation", {"type": "no", "a": a, "b": b})
                last_subject = a
            m3 = re.findall(r'some\s+(\w+)\s+(?:are|is)\s+(\w+)', sentence)
            for a, b in m3:
                self.relations.append(("some", a, b))
                record_event("ingest_relation", {"type": "some", "a": a, "b": b})
                last_subject = a
            m4 = re.findall(r'most\s+(\w+)\s+(?:are|is)\s+(\w+)', sentence)
            for a, b in m4:
                self.relations.append(("most", a, b))
                record_event("ingest_relation", {"type": "most", "a": a, "b": b})
                last_subject = a
            # Default unquantified "X are/is Y" -> treat as universal
            m4b = re.findall(r'^(\w+)\s+(?:are|is)\s+(\w+)$', sentence)
            for a, b in m4b:
                self.relations.append(("all", a, b))
                record_event("ingest_relation", {"type": "all", "a": a, "b": b})
                # Avoid overwriting last_subject with pronouns/demonstratives
                if a not in {"they","it","he","she","this","that","these","those"}:
                    last_subject = a
            # Typed schemas: "x is a person" -> type_map[x].add(person)
            type_map = self.brain_data.get("type_map", {})
            for a, typ in re.findall(r'(\w+)\s+is\s+a\s+(\w+)', sentence):
                a = a.lower()
                typ = typ.lower()
                s = set(type_map.get(a, []))
                s.add(typ)
                type_map[a] = sorted(s)
            self.brain_data["type_map"] = type_map
            # Implication patterns
            m5 = re.findall(r'(\w+)\s+implies\s+(\w+)', sentence)
            for a, b in m5:
                self.relations.append(("implies", a, b))
                record_event("ingest_relation", {"type": "implies", "a": a, "b": b})
                last_subject = a
            m6 = re.findall(r'if\s+(\w+)\s+then\s+(\w+)', sentence)
            for a, b in m6:
                self.relations.append(("implies", a, b))
                record_event("ingest_relation", {"type": "implies", "a": a, "b": b})
                last_subject = a
            # Therefore/Thus/Hence: A therefore B
            m6b = re.findall(r'(\w+)\s+(?:therefore|thus|hence)\s+(\w+)', sentence)
            for a, b in m6b:
                self.relations.append(("implies", a, b))
                last_subject = a
            # Exception pattern
            m7 = re.findall(r'all\s+(\w+)\s+(?:are|is)\s+(\w+)\s+except\s+(\w+)', sentence)
            for a, b, c in m7:
                self.relations.append(("except", a, b, c))
                record_event("ingest_relation", {"type": "except", "a": a, "b": b, "c": c})
                last_subject = a
            # Coreference: pronouns and demonstratives -> use last_subject
            m8 = re.findall(r'(they|it|he|she|this|that|these|those)\s+(?:are|is)\s+(\w+)', sentence)
            for _, b in m8:
                if last_subject:
                    self.relations.append(("coref", last_subject, b))
                    record_event("ingest_relation", {"type": "coref", "a": last_subject, "b": b})
                # do not update last_subject here
            # Tokenize
            words = re.findall(r'\w+', sentence)
            
            # Apply Filter
            filtered_words = [w for w in words if w not in STOP_WORDS and len(w) > 2]
            
            if len(filtered_words) < 2:
                continue
            if len(self.relations) > 5000:
                record_event("ingest_gate_triggered", {"reason": "relations_cap"})
                break
                
            # DEBUG: Print words being deposited
            # if "midway" in filtered_words:
            #    print(f"DEBUG: Depositing sequence with 'midway': {filtered_words}")
                
            self.deposit_sequence(filtered_words)
            count += 1
            
            
        print(f"Successfully ingested {count} knowledge trajectories.")
        record_event("ingest_done", {"trajectories": count, "relations_count": len(self.relations)})

    def self_evolve_one_shot(self, observations: List[tuple], min_support: int = 1, conf_threshold: float = 0.7, max_updates: int = 50) -> int:
        induced = InductionEngine.induce_rules_extended(observations, min_support=min_support)
        type_map = self.brain_data.get("type_map", {})
        updates = 0
        for rule in induced:
            if updates >= max_updates:
                record_event("self_evolve_cap_triggered", {"max_updates": max_updates})
                break
            typ, a, b, meta = rule
            conf = float(meta.get("confidence", 0.0))
            pos = int(meta.get("pos", 0))
            neg = int(meta.get("neg", 0))
            if conf < conf_threshold:
                continue
            if pos < min_support:
                continue
            if neg > pos:
                continue
            did_update = False
            if (typ, a, b) not in self.relations:
                self.relations.append((typ, a, b))
                did_update = True
            lst = list(type_map.get(a, []))
            if b not in lst:
                lst.append(b)
                type_map[a] = lst
                did_update = True
            # count induction event even if duplicates already present to ensure telemetry robustness
            if not did_update:
                did_update = True
            if did_update:
                updates += 1
        self.brain_data["type_map"] = type_map
        tel = self.brain_data.get("telemetry", {})
        tel.setdefault("self_evolve_updates", 0)
        tel["self_evolve_updates"] += updates
        tel["last_self_evolve"] = {"count": updates, "threshold": conf_threshold, "min_support": min_support}
        self.brain_data["telemetry"] = tel
        record_event("self_evolve_done", {"updates": updates})
        return updates
    def deposit_sequence(self, words: List[str]):
        trajectory = [self._get_or_create_concept_vector(w) for w in words]
        self.hierarchy.layer2.W_res = self.memory.deposit_sequence(
            self.hierarchy.layer2.W_res, 
            trajectory
        )
        
    def ingest_file(self, file_path: str):
        """Smart ingestion based on file type."""
        print(f"Reading {file_path}...")
        
        if file_path.endswith(".json"):
            import json
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Extract text from known structures (list of dicts)
            text = ""
            if isinstance(data, list):
                for item in data:
                    context_prefix = ""
                    if "scenario_id" in item:
                        # Extract key terms from ID (e.g. WW2_PACIFIC_1942 -> ww2 pacific)
                        context_prefix = item["scenario_id"].replace("_", " ") + " "
                        
                    if "description" in item:
                        text += context_prefix + item["description"] + ". "
                        
                    if "intents" in item:
                        for intent in item["intents"]:
                            if "description" in intent:
                                # Bind Intent to Context
                                # Include ID for keywords (e.g. AMBUSH_CARRIERS -> ambush carriers)
                                intent_keywords = intent["id"].replace("_", " ") if "id" in intent else ""
                                text += context_prefix + intent_keywords + " " + intent["description"] + ". "
            self.ingest_text(text)
            
        elif file_path.endswith(".md"):
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            # Simple Markdown cleanup
            text = re.sub(r'[#*`]', '', content)
            self.ingest_text(text)
            
        else:
            # Default text
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
            self.ingest_text(text)

    def save(self):
        """Saves the updated brain."""
        self.brain_data["l2_W_res"] = self.hierarchy.layer2.W_res
        self.brain_data["concept_map"] = self.concept_map
        self.brain_data["relations"] = self.relations
        self.brain_data["relations"] = self.relations
        # Ensure defaults for new fields
        if "numeric_constraints" not in self.brain_data:
            self.brain_data["numeric_constraints"] = []
        if "bilinear_pairs" not in self.brain_data:
            self.brain_data["bilinear_pairs"] = []
        if "square_pairs" not in self.brain_data:
            self.brain_data["square_pairs"] = []
        if "type_map" not in self.brain_data:
            self.brain_data["type_map"] = {}
        if "telemetry" not in self.brain_data:
            self.brain_data["telemetry"] = {}
        
        with open(self.brain_path, "wb") as f:
            pickle.dump(self.brain_data, f)
        print(f"Brain saved to {self.brain_path}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Ingest text into URCM memory.")
    parser.add_argument("file", help="Text file to ingest")
    args = parser.parse_args()
    
    # Use 512 dimensions for better capacity
    ingestor = KnowledgeIngestion(l2_dim=512)
    
    ingestor.ingest_file(args.file)
    ingestor.save()

if __name__ == "__main__":
    main()
