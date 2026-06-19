"""
전처리 파이프라인 v3:
  - 오선 감지: 모폴로지
  - 코드 기호: PIL 전처리 + Tesseract (staff_height 기준 영역 수정)
  - 가사: PaddleOCR (한국어+영어 혼합 최강)
  - 오선 위: 코드 / 오선 아래: 가사
"""
import fitz
import sys, io, re
import numpy as np
import cv2
from PIL import Image, ImageFilter, ImageEnhance
import pytesseract
import easyocr

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

PDF_PATH = r"C:\Users\강우현\Desktop\Finale_Ref\pdfs\A Whole New World D (중등 음악2 교학사).pdf"

# ── 1. PDF → 이미지 ─────────────────────────────────────────────────
doc = fitz.open(PDF_PATH)
page = doc[0]
mat = fitz.Matrix(600 / 72, 600 / 72)
pix = page.get_pixmap(matrix=mat)
img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
h, w = img_gray.shape

# ── 2. 오선 감지 ────────────────────────────────────────────────────
_, binary = cv2.threshold(img_gray, 180, 255, cv2.THRESH_BINARY_INV)
kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (w // 6, 1))
staff_mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_h, iterations=2)
line_rows = np.where(staff_mask.sum(axis=1) > w * 0.3)[0]

def merge_close(rows, gap=20):
    if not len(rows):
        return []
    merged, cluster = [], [rows[0]]
    for r in rows[1:]:
        if r - cluster[-1] <= gap:
            cluster.append(r)
        else:
            merged.append(int(np.mean(cluster)))
            cluster = [r]
    merged.append(int(np.mean(cluster)))
    return merged

def find_staves(merged):
    staves = []
    i = 0
    while i + 4 < len(merged):
        five = merged[i:i+5]
        gaps = [five[j+1] - five[j] for j in range(4)]
        avg = np.mean(gaps)
        if max(gaps) < avg * 1.8:
            staves.append((five[0], five[-1]))   # (top_y, bot_y)
            i += 5
        else:
            i += 1
    return staves

staves = find_staves(merge_close(line_rows))
print(f"감지된 오선 수: {len(staves)}")

# ── 3. EasyOCR 초기화 ───────────────────────────────────────────────
print("EasyOCR 초기화 중...")
ocr = easyocr.Reader(['ko', 'en'], gpu=False)

# ── 4. 코드 기호 전처리 ─────────────────────────────────────────────
def preprocess_chord(crop_np):
    pil = Image.fromarray(crop_np)
    pil = ImageEnhance.Contrast(pil).enhance(2.5)
    pil = pil.filter(ImageFilter.SHARPEN)
    return pil

chord_pattern = re.compile(r'^[A-G][#b]?(m|M|maj|min|dim|aug|sus|add)?[0-9]?$')
chord_config  = r"--psm 6 -c tessedit_char_whitelist=ABCDEFGabcdefgmM#b1234567"

chord_results = []
lyric_results = []

for idx, (top_y, bot_y) in enumerate(staves):
    staff_height = bot_y - top_y          # 오선 전체 높이 (핵심 수정)
    next_top = staves[idx+1][0] if idx+1 < len(staves) else h

    # ── 코드 기호 영역: 오선 위로 staff_height 만큼 ─────────────────
    c_top = max(0, top_y - staff_height)
    c_bot = top_y
    if c_bot > c_top + 10:
        chord_img = preprocess_chord(img_gray[c_top:c_bot, :])
        data = pytesseract.image_to_data(
            chord_img, lang="eng", config=chord_config,
            output_type=pytesseract.Output.DICT
        )
        for i, text in enumerate(data["text"]):
            text = text.strip()
            conf = int(data["conf"][i])
            if chord_pattern.match(text) and conf > 55:
                chord_results.append((idx+1, data["left"][i], text, conf))

    # ── 가사 영역: 오선 아래 ~ 다음 오선 위 ────────────────────────
    l_top = bot_y + 5
    l_bot = min(next_top - 10, bot_y + staff_height * 2)
    if l_bot > l_top + 10:
        lyric_crop = img_np[l_top:l_bot, :]
        result = ocr.readtext(lyric_crop)
        if result:
            texts = []
            for (_, text, conf) in result:
                cleaned = re.sub(r'[^가-힣a-zA-Z\s\-]', ' ', text).strip()
                if cleaned and conf > 0.5:
                    texts.append(cleaned)
            if texts:
                lyric_results.append((idx+1, ' '.join(texts)))

# ── 5. 결과 출력 ────────────────────────────────────────────────────
print("\n=== 코드 기호 ===")
prev = None
for staff_num, x, text, conf in sorted(chord_results, key=lambda r: (r[0], r[1])):
    if staff_num != prev:
        print(f"\n  [오선 {staff_num}]")
        prev = staff_num
    print(f"    [{text:<6}] x={x:>5}  신뢰도: {conf}%")

print("\n\n=== 가사 ===")
for staff_num, text in lyric_results:
    print(f"\n  [오선 {staff_num}]")
    print(f"    {text}")
