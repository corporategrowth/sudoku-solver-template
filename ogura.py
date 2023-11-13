from typing import List
from numpy.typing import NDArray
import numpy as np


def solve(image: NDArray[np.uint8], poems: List[str], level: int) -> List[int]:
    """
    Inputs:
      image: input image
      poems: list of ogura poems
      level: difficulty level of this problem (1-3)
    Outputs:
      answer: list of determination status
        0: specific poem does not exist in the image
        1: possible poem can exist in the image, but there remains other possible poems
        2: the specific poem exist in the card, and there is no other possible poems
    """
    return [0] * len(poems)