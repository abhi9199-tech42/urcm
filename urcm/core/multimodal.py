
import os
from typing import Dict, List

import numpy as np

from urcm.core.phoneme_mapper import PhonemeFrequencyPipeline


class VisualEncoder:
    def __init__(self, output_dim: int = 512):
        self.output_dim = output_dim
        # Simulated "ResNet" weights (Random projection)
        rng = np.random.RandomState(1337)
        self.projection = rng.normal(0, 0.1, (64, output_dim))
        self.learned_labels = set()

    def encode_image(self, image_path: str) -> np.ndarray:
        """
        Encodes an image (simulated) into the concept vector space.
        """
        print(f"[Visual] Processing image: {image_path}")
        # Content-based features from file bytes (fallback: path string bytes)
        import hashlib
        raw_bytes = None
        try:
            if os.path.exists(image_path):
                with open(image_path, "rb") as f:
                    raw_bytes = f.read()
        except Exception:
            raw_bytes = None
        if raw_bytes is None:
            raw_bytes = image_path.encode()
        h = hashlib.md5(raw_bytes).hexdigest()
        features = np.array([int(h[i:i+2], 16) / 255.0 for i in range(0, 32)] * 2)
        # Optional: RGB stats via Pillow if available
        try:
            from PIL import Image
            if os.path.exists(image_path):
                with Image.open(image_path) as img:
                    img = img.convert("RGB")
                    arr = np.asarray(img, dtype=np.float32) / 255.0
                    r_mean = float(np.mean(arr[...,0]))
                    g_mean = float(np.mean(arr[...,1]))
                    b_mean = float(np.mean(arr[...,2]))
                    y_mean = float(0.2126*r_mean + 0.7152*g_mean + 0.0722*b_mean)
                    sat = float(np.mean(np.max(arr, axis=-1) - np.min(arr, axis=-1)))
                    features[0] = r_mean
                    features[1] = g_mean
                    features[2] = b_mean
                    features[3] = y_mean
                    features[4] = sat
                    sums = [float(np.sum(arr[...,i])) for i in range(3)]
                    dom = int(np.argmax(sums))
                    features[5] = [0.0, 0.5, 1.0][dom]
                    bins = np.linspace(0.0, 1.0, 9)
                    r_hist, _ = np.histogram(arr[...,0], bins=bins)
                    g_hist, _ = np.histogram(arr[...,1], bins=bins)
                    b_hist, _ = np.histogram(arr[...,2], bins=bins)
                    hist = np.concatenate([r_hist, g_hist, b_hist]).astype(np.float32)
                    hist = hist / (np.sum(hist) + 1e-6)
                    features[8:32] = hist
                    try:
                        gimg = img.convert("L").resize((8,8))
                        garr = np.asarray(gimg, dtype=np.float32)
                        avg = float(np.mean(garr))
                        bits = (garr > avg).astype(np.float32).reshape(-1)
                        ph = []
                        for i in range(0, 64, 4):
                            ph.append(float(np.mean(bits[i:i+4])))
                        features[32:48] = np.array(ph, dtype=np.float32)
                    except Exception:
                        pass
        except Exception:
            pass

        # Project to Concept Space
        vector = np.dot(features[:64], self.projection)

        # Normalize
        vector = vector / np.linalg.norm(vector)
        return vector

    def detect_objects(self, image_path: str) -> List[str]:
        """
        Simulates object detection (YOLO-style).
        """
        # Content-driven detection using encoded vector statistics
        vec = self.encode_image(image_path)
        detected = []
        energy = float(np.linalg.norm(vec))
        mean_val = float(np.mean(vec))
        # Basic objectness based on energy
        if energy > 0.9:
            detected.append("object")
        # Try PIL-based brightness/color first
        used_pil = False
        try:
            from PIL import Image
            if os.path.exists(image_path):
                with Image.open(image_path) as img:
                    img = img.convert("RGB")
                    arr = np.asarray(img, dtype=np.float32) / 255.0
                    r_mean = float(np.mean(arr[...,0]))
                    g_mean = float(np.mean(arr[...,1]))
                    b_mean = float(np.mean(arr[...,2]))
                    y_mean = float(0.2126*r_mean + 0.7152*g_mean + 0.0722*b_mean)
                    detected.append("bright" if y_mean > 0.5 else "dark")
                    dom = int(np.argmax([r_mean, g_mean, b_mean]))
                    detected.append(["red-ish","green-ish","blue-ish"][dom])
                    used_pil = True
        except Exception:
            used_pil = False
        if not used_pil:
            # Fallback to embedded stats
            detected.append("bright" if mean_val > 0.02 else "dark")
            idx = int(np.argmax(vec)) % 3
            detected.append(["red-ish","green-ish","blue-ish"][idx])
        return detected

    def learn_from_image(self, image_path: str):
        """
        One-shot learning: add unknown tokens from first example to learned labels.
        """
        name = os.path.basename(image_path).lower()
        tokens = [t for t in name.replace('.', '_').split('_') if t]
        for t in tokens:
            # Ignore common tokens
            if t not in {"image", "photo", "frame", "jpg", "jpeg", "png"}:
                self.learned_labels.add(t)

    def extract_numeric_attributes(self, image_path: str) -> Dict[str, float]:
        name = os.path.basename(image_path).lower()
        import re as _re
        attrs = {}
        for tok in name.replace('.', '_').split('_'):
            m = _re.match(r'^([0-9]+(?:\.[0-9]+)?)(kg|g|m|cm|mm|s|ms)$', tok)
            if m:
                val = float(m.group(1))
                unit = m.group(2)
                # normalize units to SI base
                if unit == "kg":
                    attrs["mass_kg"] = val
                elif unit == "g":
                    attrs["mass_kg"] = val / 1000.0
                elif unit == "m":
                    attrs["length_m"] = val
                elif unit == "cm":
                    attrs["length_m"] = val / 100.0
                elif unit == "mm":
                    attrs["length_m"] = val / 1000.0
                elif unit == "s":
                    attrs["time_s"] = val
                elif unit == "ms":
                    attrs["time_s"] = val / 1000.0
        return attrs

    def infer_numeric_unit(self, image_path: str) -> Dict[str, float]:
        unit = "m"
        conf = 0.5
        try:
            from PIL import Image
            if os.path.exists(image_path):
                with Image.open(image_path) as img:
                    img = img.convert("RGB")
                    arr = np.asarray(img, dtype=np.float32) / 255.0
                    r_mean = float(np.mean(arr[...,0]))
                    g_mean = float(np.mean(arr[...,1]))
                    b_mean = float(np.mean(arr[...,2]))
                    sat = float(np.mean(np.max(arr, axis=-1) - np.min(arr, axis=-1)))
                    dom = int(np.argmax([r_mean, g_mean, b_mean]))
                    unit = ["C","m","kg"][dom]
                    conf = float(min(1.0, sat * 2.0))
        except Exception:
            vec = self.encode_image(image_path)
            idx = int(np.argmax(vec)) % 3
            unit = ["C","m","kg"][idx]
            conf = float(min(1.0, np.std(vec[:16]) * 10.0))
        return {"unit": unit, "confidence": conf}

    def extract_numeric_value_with_bounds(self, image_path: str) -> Dict[str, float]:
        vec = self.encode_image(image_path)
        value = float(np.mean(vec[:10]) * 100.0)
        bound = float(np.std(vec[:10]) * 10.0)
        u = self.infer_numeric_unit(image_path)
        return {"value": value, "unit": u["unit"], "lower": value - 2.0 * bound, "upper": value + 2.0 * bound, "confidence": u["confidence"]}

    def extract_quantity_from_scale(self, image_path: str, pixels_per_unit: float = 100.0, axis: str = "width") -> Dict[str, float]:
        try:
            from PIL import Image
            if os.path.exists(image_path):
                with Image.open(image_path) as img:
                    w, h = img.size
                    px = w if axis == "width" else h
                    qty = float(px) / max(1e-6, float(pixels_per_unit))
                    return {"quantity": qty, "unit": "u", "confidence": 0.9}
        except Exception:
            pass
        return {"quantity": float(np.linalg.norm(self.encode_image(image_path)) % 10.0), "unit": "u", "confidence": 0.3}

    def verify_image_dimensions(self, image_path: str) -> bool:
        """
        Verifies 'WxHpx' tokens against actual image dimensions if file exists.
        Returns True if consistent or cannot verify.
        """
        name = os.path.basename(image_path).lower()
        import re as _re
        m = _re.search(r'(\d+)[x_](\d+)px', name)
        if not m:
            return True
        expected_w = int(m.group(1))
        expected_h = int(m.group(2))
        try:
            from PIL import Image
            if os.path.exists(image_path):
                with Image.open(image_path) as img:
                    w, h = img.size
                    return (w == expected_w and h == expected_h)
            return True
        except Exception:
            return True

class AudioProcessor:
    """
    Handles Audio input via Phoneme Mapping (Speech-to-Concept).
    """
    def __init__(self, frequency_dim: int = 24):
        self.pipeline = PhonemeFrequencyPipeline(frequency_dim=frequency_dim)

    def process_audio_file(self, audio_path: str) -> np.ndarray:
        """
        Simulates processing an audio file.
        In reality, this would use Whisper/STT.
        Here, we assume the filename contains the 'transcription' or we generate a dummy one.
        """
        print(f"[Audio] Processing audio: {audio_path}")
        # Mock: extract text from filename "audio_hello_world.wav" -> "hello world"
        filename = os.path.basename(audio_path)
        text = filename.replace("audio_", "").replace(".wav", "").replace("_", " ")

        # Convert to Phoneme Frequency Path
        freq_path = self.pipeline.process_text(text)

        # Pool to single vector (Mean Pooling)
        vector = np.mean(freq_path.vectors, axis=0)
        return vector

class VideoProcessor:
    """
    Combines Visual and Audio streams.
    """
    def __init__(self):
        self.visual = VisualEncoder()
        self.audio = AudioProcessor()

    def process_video(self, video_path: str) -> Dict[str, any]:
        print(f"[Video] Processing video: {video_path}")
        # 1. Sample Frames (Simulated)
        frame_vec = self.visual.encode_image(video_path + "_frame1")

        # 2. Extract Audio (Simulated)
        audio_vec = self.audio.process_audio_file(video_path + "_audio.wav")

        # Pad audio_vec to 256 dims if smaller, then fuse
        if audio_vec.shape[0] < 256:
            audio_padded = np.zeros(256, dtype=audio_vec.dtype)
            audio_padded[:audio_vec.shape[0]] = audio_vec
        else:
            audio_padded = audio_vec[:256]
        return {
            "visual_embedding": frame_vec,
            "audio_embedding": audio_vec,
            "fused_embedding": np.concatenate([frame_vec[:256], audio_padded])
        }
