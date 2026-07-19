import os
import tempfile

from urcm.core.multimodal import VisualEncoder


def write_temp_file(content: bytes, suffix: str):
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        f.write(content)
    return path

def test_detection_independent_of_filename():
    v = VisualEncoder()
    data = b"\x89PNG\r\n\x1a\n" + b"\x00" * 1024
    path1 = write_temp_file(data, ".png")
    # same bytes, different misleading name
    path2 = write_temp_file(data, "_red_sports_car.jpg")
    det1 = v.detect_objects(path1)
    det2 = v.detect_objects(path2)
    assert det1 == det2
    os.remove(path1)
    os.remove(path2)

def test_detection_changes_with_content():
    v = VisualEncoder()
    data1 = b"A" * 2048
    data2 = b"B" * 2048
    p1 = write_temp_file(data1, ".bin")
    p2 = write_temp_file(data2, ".bin")
    d1 = v.detect_objects(p1)
    d2 = v.detect_objects(p2)
    assert d1 != d2 or d1.count("object") != d2.count("object")
    os.remove(p1)
    os.remove(p2)
