# TODO: 조표 감지 개선

작성일: 2026-06-28

## 현재 상태

| 기준 | 정확도 |
|---|---|
| 5곡 대표 샘플 (C, G, D, A, Dm) | **5/5 (100%)** |
| 80개 PDF 광범위 테스트 (구 코드) | **33/80 (41.2%)** |
| 80개 PDF (신 코드, ts-bounded key_end) | 측정 중 (2026-06-28) |

### 구현 파일

- `note_recognition/header_detector.py` — `detect_header()`, `_detect_key_sig()`
- `tests/fixtures/scores/` — 5곡 샘플 PDF+MXL (커밋됨)

---

## ✅ 완료된 개선

### B. ts-bounded key_end (2026-06-28 완료)

`detect_header()`에서 `_locate_time_sig_blobs()`를 먼저 호출해
박자표 실제 x 시작 위치를 파악하고 `key_end`를 동적으로 클램프.

```python
# detect_header() 내부
ts_x0_rel, _ = _locate_time_sig_blobs(bin_ts_zone, gap)
if ts_x0_rel is not None:
    abs_ts_x0 = ts_start_approx + ts_x0_rel
    key_end_actual = max(clef_end + 1, abs_ts_x0 - int(gap * 0.3))
else:
    key_end_actual = key_end  # fallback
```

**효과:**
- C major_false 제거: C장조에서 key_end가 clef_end 이하로 좁혀져 박자표 숫자 획이 key 존에 들어오지 않음
- count_wrong 개선: 박자표가 조표 존에 포함되던 경우 제거

**이 방법이 안전한 이유:**
`_locate_time_sig_blobs`는 분자+분모 쌍(staff 상반부+하반부 블롭)을 요구하므로
조표 기호(샵·플랫)는 오선 상/하 쌍이 없어 무시됨. 박자표만 검출됨.

---

## ❌ 미해결 실패 유형 (구 코드 기준 80개 샘플)

### 1. flat_as_sharp (3건)

**증상**: `ks=-4 det=+3/+4` (Ab장조 곡들)

**원인**: 플랫의 세로 줄기(stem)이 tall-narrow 필터(`bh > gap*0.6 AND bw < gap*0.35`)를 통과
- 플랫 줄기: `bh ≈ gap (=0.6*gap 이상)`, `bw ≈ 2px (= 0.05*gap << 0.35*gap)` → 샵 막대로 오분류

**해결 방법**: 수평 투영 피크 2개 유무로 샵/플랫 구분 (아래 Plan A 참고)

---

### 2. C_major_false (3건)

**증상**: `ks=+0 det=+1` (C장조 곡에서 샵 1개 오감지)

**원인 (구 코드)**: C장조는 조표 없이 박자표가 바로 옴.
박자표 "4" 숫자의 가느다란 세로 획(bh≈1.85*gap, bw≈2-3px)이 key 존에 포함되어 샵 막대로 오분류.

**해결 방법**: ✅ **ts-bounded key_end(Plan B)로 해결** — key 존이 박자표 x 이전으로 좁혀짐

---

### 3. sharp_as_flat (3건)

**증상**: `ks=+1 or +2 det=-1` (My Favorite Things Em, A Whole New World D 등)

**원인**: 일부 출판사 폰트에서 샵 막대 bw가 gap*0.35를 초과해 tall-narrow 필터 미통과
→ has_sharp_bar = False → 플랫 분기로 떨어짐
→ excluded_run이 없으면 return 0, 있으면 return -n

**해결 방법**:
- has_sharp_bar=False일 때 excluded_run이 없으면 무조건 0 반환 (샵이 하나도 없는 줄기 없는 경우)
- 또는 Plan A로 샵/플랫을 픽셀 패턴으로 구분

---

### 4. sharp_missed (3건)

**증상**: `ks=+1 det=+0` (10주년 기념 G장조 곡들, gap=35px 계열)

**원인**: `key_end = x_start + gap*10.0` 고정값이 작아 샵이 존 바깥에 있거나,
또는 `n_sharps = round(span / (gap*0.80))` 반올림에서 0.5 아래로 떨어지는 경우.

**해결 방법**:
- ts-bounded key_end(완료) → 샵 존이 실제 박자표 직전까지 확장되어 missed 감소 기대
- n_sharps 계산에서 단일 샵의 경우 `max(1, ...)` 이미 있으므로 detected >= 1이어야 함
  → 실제로 key_run이 비어있는지 별도 조사 필요 (아래 TODO)

---

### 5. count_wrong (3건)

**증상**: `ks=+3 det=+1`, `ks=+2 det=+1`, `ks=+2 det=+6` 등 개수 오류

**원인**:
- span / (gap*0.80) 공식이 일부 폰트에서 오차가 큰 경우
- 샵 기호 사이 gap이 다른 폰트에서는 배수 계수(0.80)가 맞지 않음
- `ks=+2 det=+6` 케이스: key_end 바깥에 있는 다른 기호가 span에 포함됨 (ts-bounded로 해결 기대)

---

## TODO 목록

### 단기 (현재 코드로 가능)

- [x] **Plan B**: ts-bounded key_end — `detect_header()`에서 `_locate_time_sig_blobs`로
  박자표 x0 먼저 파악 후 key_end 동적 클램프 *(2026-06-28 완료)*

- [ ] **광범위 재테스트**: 신 코드로 80개 PDF 재측정 (진행 중, bumkmcg3n)

- [ ] **sharp_missed 원인 조사**: gap=35px 계열 G장조 곡 1개 골라 디버그
  - `detect_header(img, staves[0], debug=True)` 출력 추가해 key_runs, has_sharp_bar 확인
  - key_run이 비어 있으면 → x_start 오계산 또는 key_start > key_end 확인

- [ ] **sharp_as_flat 원인 조사**: My Favorite Things Em 한 곡 골라 디버그
  - 샵 막대 실제 bw 측정: gap=41.5px → 임계값 gap*0.35=14.5px, 실제 bw가 얼마인지
  - bw 임계값을 gap*0.45로 올리면 해결되는지 확인

### 중기

- [ ] **Plan A**: 수평 투영 피크 카운팅으로 샵/플랫 구분
  - 샵(#): 두 개의 가로 막대 → 행 합산 투영에서 피크 2개
  - 플랫(b) 줄기: 가로 막대 없음 → 투영 피크 없거나 1개
  - 박자표 숫자 "4": 피크 패턴이 다름 (이미 key 존에서 제외 예정이지만 fallback으로 유용)

  ```python
  # 각 tall-narrow blob에 대해 수평 투영 검사
  hproj = np.sum(blob_region > 0, axis=1)
  hproj_smooth = np.convolve(hproj, np.ones(3)/3, mode='same')
  peaks = count_peaks(hproj_smooth, threshold=0.3 * hproj.max())
  is_sharp_bar = (peaks >= 2)
  ```

- [ ] **n_sharps 개수 공식 보정**: 출판사별 샵 간격 다름
  - 현재: `n_sharps = max(1, round(span / (gap * 0.80)))`
  - 대안: blob 개수 카운팅 → 샵 막대 쌍(수직선 2개) = 샵 1개

### 장기

- [ ] **자동 조표 감지 통합**: `detect_header()` 결과를 `opencv_runner.py`에 연동
  - 현재: config.ini `key_sig` 수동 입력
  - 목표: `result.key_sig` 자동 사용 (정확도 ≥80% 달성 후)

- [ ] **플랫 개수 감지 개선**: 현재 width 기반 추정 (`excl_w / (gap*1.7)`)이 불안정
  - 대안: 플랫 기호의 볼록한 머리(round head) blob 개수 직접 카운팅

---

## 디버그 명령

```bash
# 5곡 대표 샘플 테스트
python -c "
import numpy as np, cv2
from pathlib import Path
from system_slicer import slice_pdf_to_systems
from note_detector import _detect_staff_lines
from note_recognition.header_detector import detect_header
# ... (see tests/test_header_detection.py)
"

# 광범위 80개 PDF 테스트
python tests/run_header_accuracy.py  # (TODO: 스크립트화)

# pytest (134개 단위 테스트)
python -m pytest tests/ -v
```

---

## 관련 파일

| 파일 | 역할 |
|---|---|
| `note_recognition/header_detector.py` | 클레프/조표/박자표 감지 핵심 로직 |
| `tests/test_arc_detection.py` | 이음줄 감지 테스트 (참고: 유사 구조) |
| `tests/fixtures/scores/` | 5곡 샘플 PDF+MXL |
