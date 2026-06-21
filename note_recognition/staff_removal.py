"""
오선(staff line) 제거 모듈.

목표: 오선 픽셀만 지우고 음표머리/기둥/깃발 등 음악 기호는 그대로 보존.

핵심 난제 (CLAUDE.md "핵심 알고리즘 주의사항" 참고):
    단순 차집합(전체 이미지 - 가로 모폴로지로 검출한 오선)을 하면
    오선과 겹치는 음표머리 픽셀까지 같이 지워져 머리에 구멍이 뚫린다.
    마디선 검출(_detect_barlines)에서도 같은 부류의 문제로 "MORPH_OPEN이나
    수평선 제거 후 방식은 모두 실패"한 이력이 있다.

전략 (run-length 기반 가역 제거):
    오선은 "가로로 매우 길게(이미지 폭의 상당 부분) 이어지는 얇은(2~3px)
    수평선"이라는 기하학적 특징이 뚜렷하다. 반면 음표머리/기둥은 오선과
    겹치는 지점에서도 "세로로 길게 이어지는" 픽셀이거나 "가로 폭이
    음표머리 지름(보통 오선 간격의 1.0~1.3배) 정도로 짧은" 픽셀이다.

    1. 각 행(row)에서 연속된 검정 픽셀 구간(run)을 찾는다.
    2. 그 구간의 길이가 "오선이 끊기지 않고 이어지는 최소 길이" 기준을
       넘으면 오선 후보로 표시한다. 단, 음표머리가 오선 위에 정확히
       걸쳐 있으면 그 지점에서 run이 더 두꺼워지므로(여러 행에 걸쳐
       검정), 세로 방향 연속성도 함께 확인해 "두께가 oneline_thickness
       근방인 행"만 오선으로 인정한다.
    3. 음표머리/기둥이 차지하는 컬럼 구간은 오선 제거에서 제외(보존)한다 -
       이게 "가역(reversible)" 제거의 핵심: 일단 오선 후보를 전부 지운
       뒤, 연결성분 분석으로 "원래는 음표머리였는데 끊겨버린" 영역을
       복원하는 방식 대신, 애초에 음표가 있는 컬럼 구간은 건드리지 않는
       보수적 접근을 쓴다 (복원보다 예방이 안전 - 디지털 PDF라 오선
       두께가 일정하다는 전제를 활용).
"""

import numpy as np
import cv2


def detect_staff_line_thickness(img_gray: np.ndarray, staff_rows: list[tuple[int, int]]) -> int:
    """
    오선 한 줄의 실제 두께(픽셀)를 측정한다.

    staff_rows: _detect_staves()가 반환하는 (top_y, bot_y) 목록 중
                하나의 오선에 대해, 그 오선을 이루는 5개 가로선의
                평균 y좌표들. 여기서는 단순화를 위해 이미지에서 직접
                재탐지한다 (오선 중심 y 근방에서 검정 픽셀의 세로
                연속 길이를 측정).

    Returns:
        오선 한 줄의 두께 (픽셀). 측정 실패 시 기본값 2 반환.
    """
    if not staff_rows:
        return 2

    top_y, bot_y = staff_rows[0]
    h, w = img_gray.shape
    sample_cols = list(range(w // 4, w // 4 * 3, max(1, w // 40)))
    thicknesses = []

    _, binary = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY_INV)

    y_range_start = max(0, top_y - 5)
    y_range_end = min(h, bot_y + 5)

    for x in sample_cols:
        # run의 "시작점"에서만 두께를 측정해야 한다. run 중간 y에서 다시
        # 재면(예: 두께 3인 run의 두 번째 행에서 측정) 남은 길이(2)만
        # 잡혀 통계가 왜곡된다 (버그 이력: 이 때문에 최빈값이 실제 두께보다
        # 항상 작게 나왔음 - 두께 3인 오선을 1로 오판).
        y = y_range_start
        while y < y_range_end:
            if binary[y, x] > 0 and (y == y_range_start or binary[y - 1, x] == 0):
                run = 0
                yy = y
                while yy < y_range_end and binary[yy, x] > 0:
                    run += 1
                    yy += 1
                thicknesses.append(run)
                y = yy
            else:
                y += 1

    if not thicknesses:
        return 2
    # 최빈값(가장 흔한 두께)을 오선 두께로 채택 - 평균은 음표머리가 섞이면 왜곡됨
    values, counts = np.unique(thicknesses, return_counts=True)
    return int(values[np.argmax(counts)])


def remove_staff_lines(
    img_gray: np.ndarray,
    top_y: int,
    bot_y: int,
    line_thickness: int = 2,
    min_run_ratio: float = 0.6,
    min_horizontal_run: int | None = None,
) -> np.ndarray:
    """
    한 오선(top_y~bot_y 범위, 5줄) 영역에서 오선만 제거한 이미지를 반환.

    원본을 변경하지 않고 복사본을 반환한다.

    알고리즘 (2단계 검증 - 세로 run-length만으로는 부족함이 실험으로 확인됨):
        1차 시도(세로 run-length만 사용)는 빈 음표머리(2분/온음표)의 타원
        테두리가 오선과 두께가 비슷해(2~3px) 오선으로 오인되어 함께
        지워지는 회귀가 합성 이미지 테스트에서 발견됨 (온음표가 거의
        통째로 사라짐). 그래서 가로 연속성 기준을 추가한다.

        1단계(가로 연속성): 각 행에서 "충분히 길게(min_horizontal_run
        이상) 이어지는 검정 구간"만 오선 후보로 마스킹한다. 진짜 오선은
        악보 시스템 폭 전체에 걸쳐 끊김없이 이어지지만, 음표머리 테두리는
        음표머리 지름(대략 staff_gap의 1~1.3배) 정도로 짧게 이어지다 끊긴다.

        2단계(세로 run-length): 1단계를 통과한(=가로로 충분히 긴) 픽셀에
        대해서만, 세로 방향 연속 길이가 오선 두께 근방인지 추가로 확인해
        최종적으로 지운다. 이 순서(가로 먼저 -> 세로 나중)가 핵심: 가로
        조건 없이 세로만 보면 짧은 빈 머리 테두리도 통과해버린다.

    Args:
        img_gray:           그레이스케일 원본 이미지 (전체 페이지)
        top_y, bot_y:        오선 5줄의 최상단/최하단 y좌표 (_detect_staves 결과)
        line_thickness:      오선 한 줄의 두께 (detect_staff_line_thickness로 측정)
        min_run_ratio:       오선으로 판정할 세로 run 길이의 상한 배율.
                            run <= line_thickness * (1/min_run_ratio) 이면 오선 후보
                            (기본 0.6 -> 임계값 ≈ 1.67배)
        min_horizontal_run:  오선으로 판정할 가로 최소 길이(픽셀). None이면
                            이미지 폭의 5%로 자동 설정 (실제 음표머리 지름은
                            보통 이보다 훨씬 짧음). 오선 간격(staff_gap)을
                            안다면 호출부에서 staff_gap*2 정도로 더 타이트하게
                            지정하는 것을 권장.

    Returns:
        오선이 제거된 이미지 복사본 (uint8, 그레이스케일)
    """
    result = img_gray.copy()
    h, w = img_gray.shape
    _, binary = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY_INV)

    y_lo = max(0, top_y - 2)
    y_hi = min(h, bot_y + 2)
    crop = binary[y_lo:y_hi, :]
    crop_h = crop.shape[0]

    vertical_threshold = max(line_thickness * 2, int(line_thickness / min_run_ratio))
    if min_horizontal_run is None:
        min_horizontal_run = max(20, int(w * 0.05))

    # ── 1단계: 각 행에서 "가로로 충분히 긴 검정 구간"을 오선 후보 마스크로 표시 ──
    is_staff_candidate = np.zeros_like(crop, dtype=bool)
    for y in range(crop_h):
        row = crop[y]
        x = 0
        while x < w:
            if row[x] > 0:
                run_start = x
                while x < w and row[x] > 0:
                    x += 1
                run_len = x - run_start
                if run_len >= min_horizontal_run:
                    is_staff_candidate[y, run_start:x] = True
            else:
                x += 1

    # ── 2단계: 오선 후보 마스크 중에서, 세로 run-length가 짧은 픽셀만 최종 제거 ──
    for x in range(w):
        col = crop[:, x]
        col_candidate = is_staff_candidate[:, x]
        if not col.any():
            continue
        y = 0
        while y < crop_h:
            if col[y] > 0:
                run_start = y
                while y < crop_h and col[y] > 0:
                    y += 1
                run_len = y - run_start
                # 세로로 짧고(오선 두께 근방) AND 가로 후보 마스크에 포함된 경우만 제거
                if run_len <= vertical_threshold and col_candidate[run_start:y].all():
                    result[y_lo + run_start: y_lo + y, x] = 255
            else:
                y += 1

    return result
