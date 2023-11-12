import os
import glob

import cv2
import numpy as np
import pytest
from ogura import solve

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CUR_DIR, "data")


def get_test_data():
    rng = np.random.RandomState(31415)

    levels = [1, 2, 3]
    data = []
    for level in levels:
        image_paths = glob.glob(os.path.join(DATA_DIR, "level{:d}/*.jpg".format(level)))
        if len(image_paths) > 10:
            idx = rng.choice(len(image_paths), 10)
            image_paths = [image_paths[i] for i in idx]
            image_paths = sorted(image_paths)
        data.extend([(path, level) for path in image_paths])

    return data


def check(path: str) -> None:
    base = os.path.splitext(path)[0]
    img_path = path
    txt_path = base + ".txt"
    assert os.path.exists(img_path), f"Image file not found: {img_path:s}"
    assert os.path.exists(txt_path), f"Text file not found: {txt_path:s}"

    dirname = os.path.basename(os.path.dirname(img_path))
    level = int(dirname[-1])

    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    poems = []
    expected = []
    with open(txt_path, mode="r", encoding="utf-8") as content:
        for ln in content:
            poem, ans = ln.strip().split()
            poems.append(poem)
            expected.append(int(ans))

    actual = solve(image, poems, level)

    act_np = np.array(actual, dtype="uint8")
    exp_np = np.array(expected, dtype="uint8")
    assert (act_np == exp_np).all(), "Your answer is wrong"


@pytest.mark.parametrize("image_path, level", get_test_data())
def test_solve(image_path: str, level: int):
    check(image_path)


@pytest.fixture(scope="session")
def path(pytestconfig):
    return pytestconfig.getoption("path")
