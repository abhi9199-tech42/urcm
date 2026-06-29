from urcm.core.multimodal import VisualEncoder

def test_verify_image_dimensions_token_without_file():
    v = VisualEncoder()
    # No actual file; function should return True (cannot verify)
    assert v.verify_image_dimensions("pic_800x600px.jpg") is True
