from urcm.core.multimodal import VisualEncoder


def test_extract_numeric_attributes_from_filename():
    v = VisualEncoder()
    attrs = v.extract_numeric_attributes("image_5kg_120cm_30ms.jpg")
    assert abs(attrs.get("mass_kg", 0.0) - 5.0) < 1e-6
    assert abs(attrs.get("length_m", 0.0) - 1.2) < 1e-6
    assert abs(attrs.get("time_s", 0.0) - 0.03) < 1e-6
