import numpy as np
import cv2
from typing import List

# 画像からカードを検出する関数
def detect_card(image: np.ndarray) -> np.ndarray:
    # カードの輪郭を見つけるための処理
    # ここでは単純化のため、最大の輪郭をカードと仮定
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 30, 150)
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        card_image = gray[y:y+h, x:x+w]
        return card_image
    else:
        return None

# カード画像から特徴を抽出する関数
def extract_features_from_card(card_image: np.ndarray) -> List[float]:
    # 特徴として、画像のピクセル値の統計を使用
    if card_image is not None:
        features = [np.mean(card_image), np.std(card_image)]
    else:
        features = []
    return features

# 特徴と詩をマッチングする関数
def match_features_with_poem(features: List[float], poem: str, level: int) -> bool:
    # この例では、詩の長さと特徴の平均値を比較
    if features and len(poem) > 0:
        return abs(features[0] - len(poem)) < 5  # ダミーの閾値
    return False

# 照合結果に基づいて答えのレベルを決定する関数
def determine_match_level(features: List[float], poem: str, level: int) -> int:
    return 1 if match_features_with_poem(features, poem, level) else 0

# 画像から詩を識別するメインの関数
def solve(image: np.ndarray, poems: List[str], level: int) -> List[int]:
    card_image = detect_card(image)
    features = extract_features_from_card(card_image)
    answer = [0] * len(poems)
    for i, poem in enumerate(poems):
        if card_image is not None and match_features_with_poem(features, poem, level):
            answer[i] = determine_match_level(features, poem, level)
    return answer
