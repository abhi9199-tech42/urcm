from urcm.core.multimodal import VisualEncoder


def test_infer_numeric_unit_and_bounds():
    v = VisualEncoder()
    u = v.infer_numeric_unit("nonexistent_image_123.png")
    assert u["unit"] in {"C","m","kg"}
    assert 0.0 <= u["confidence"] <= 1.0
    b = v.extract_numeric_value_with_bounds("nonexistent_image_123.png")
    assert b["lower"] < b["value"] < b["upper"]
    assert 0.0 <= b["confidence"] <= 1.0
