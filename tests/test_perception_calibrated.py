from urcm.core.multimodal import VisualEncoder


def test_extract_quantity_from_scale_fallback():
    enc = VisualEncoder()
    out = enc.extract_quantity_from_scale("nonexistent.png", pixels_per_unit=50.0)
    assert "quantity" in out and "confidence" in out
    assert out["confidence"] <= 0.9
