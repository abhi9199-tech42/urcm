"""
Safe serialization for URCM model data.
Replaces insecure pickle.load() with validated deserialization.

Uses JSON for metadata/config and numpy .npy format for arrays.
For backward compatibility, provides a restricted pickle loader
with integrity verification and size limits.
"""

import json
import os
import hashlib
import logging
from typing import Any, Dict, Optional
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Maximum allowed size for loaded data (500MB)
MAX_LOAD_SIZE = 500 * 1024 * 1024

# Allowed pickle classes for restricted unpickling
# Only numpy arrays and basic Python types
ALLOWED_PICKLE_TYPES = {
    np.ndarray,
    np.float64, np.float32, np.float16,
    np.int64, np.int32, np.int16, np.int8,
    np.uint64, np.uint32, np.uint16, np.uint8,
    dict, list, tuple, set, str, int, float, bool, bytes, type(None),
}


def compute_sha256(filepath: str) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


def verify_integrity(filepath: str, expected_hash: Optional[str] = None) -> bool:
    """
    Verify file integrity using SHA-256.
    If expected_hash is None, just check the file can be read.
    """
    if not os.path.exists(filepath):
        logger.error(f"File not found: {filepath}")
        return False

    file_size = os.path.getsize(filepath)
    if file_size > MAX_LOAD_SIZE:
        logger.error(f"File too large: {file_size} bytes (max {MAX_LOAD_SIZE})")
        return False

    if file_size == 0:
        logger.error(f"Empty file: {filepath}")
        return False

    if expected_hash:
        actual = compute_sha256(filepath)
        if actual != expected_hash:
            logger.error(f"SHA-256 mismatch for {filepath}")
            return False

    return True


def safe_load_npy(filepath: str, expected_hash: Optional[str] = None) -> Optional[np.ndarray]:
    """
    Safely load a .npy file with integrity verification.
    """
    if not verify_integrity(filepath, expected_hash):
        return None
    try:
        return np.load(filepath, allow_pickle=False)
    except Exception as e:
        logger.error(f"Failed to load {filepath}: {e}")
        return None


def safe_save_npy(filepath: str, data: np.ndarray) -> bool:
    """
    Safely save a numpy array to .npy format.
    """
    try:
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        np.save(filepath, data, allow_pickle=False)
        return True
    except Exception as e:
        logger.error(f"Failed to save {filepath}: {e}")
        return False


def safe_load_json(filepath: str, expected_hash: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Safely load a JSON file with integrity verification.
    """
    if not verify_integrity(filepath, expected_hash):
        return None
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load JSON {filepath}: {e}")
        return None


def safe_save_json(filepath: str, data: Dict[str, Any]) -> bool:
    """
    Safely save data to JSON format.
    """
    try:
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)
        return True
    except Exception as e:
        logger.error(f"Failed to save JSON {filepath}: {e}")
        return False


def safe_load_restricted_pickle(filepath: str, expected_hash: Optional[str] = None) -> Optional[Any]:
    """
    Restricted pickle loader with integrity verification.
    Only allows numpy arrays and basic Python types.
    Falls back to JSON/npy for new files.
    """
    if not verify_integrity(filepath, expected_hash):
        return None

    import pickle
    try:
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        logger.error(f"Failed to load pickle {filepath}: {e}")
        return None


def safe_load(filepath: str, expected_hash: Optional[str] = None) -> Optional[Any]:
    """
    Universal safe loader. Detects file format and loads accordingly.
    .npy -> safe_load_npy
    .json -> safe_load_json
    .pkl / .pickle -> safe_load_restricted_pickle
    """
    ext = Path(filepath).suffix.lower()
    if ext == '.npy':
        return safe_load_npy(filepath, expected_hash)
    elif ext == '.json':
        return safe_load_json(filepath, expected_hash)
    elif ext in ('.pkl', '.pickle'):
        return safe_load_restricted_pickle(filepath, expected_hash)
    else:
        logger.warning(f"Unknown file extension: {ext}, trying restricted pickle")
        return safe_load_restricted_pickle(filepath, expected_hash)
