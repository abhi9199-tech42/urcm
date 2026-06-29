
import sys
import os

# Add ISRE to path (set URCM_ISRE_PATH env var or edit this default)
isre_path = os.environ.get("URCM_ISRE_PATH", os.path.join(os.path.dirname(__file__), "..", "ISRE"))
sys.path.append(isre_path)

try:
    from isre.compression.text import ConceptMapper
    
    mapper = ConceptMapper()
    print("Successfully imported ConceptMapper")
    print("Semantic Map:", mapper._semantic_map)
    
except ImportError as e:
    print(f"Import failed: {e}")
except Exception as e:
    print(f"Error: {e}")
