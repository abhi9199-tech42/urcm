
import sys
import os

# Add ISRE to path
isre_path = r"C:\Users\kriti\OneDrive\Intentional Semantic Reasoning Engine (ISRE)"
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
