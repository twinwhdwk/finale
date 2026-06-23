"""
자체 음표 인식기 (OpenCV 기반).

Audiveris를 대체하는 독자 OMR 파이프라인. CLAUDE.md "입력 데이터 특성"의
조건(디지털 PDF, 노이즈 거의 없음, 2성부 이하, 중고등학생 수준 악보)에
특화해 정밀도를 높이는 것이 목표.

## 전체 파이프라인 (5단계)

    1. staff_removal  - 오선 제거 (2단계 run-length)
    2. note_detector  - 음가 분류 + 빔 분리 + 쉼표/점음표 탐지
    3. note_pitcher   - 음높이 판정 (head_y → Pitch)
    4. key_signature  - 조표 적용 (Pitch에 accidental 반영)
    5. xml_builder    - MusicXML 생성 (마디선 기반 마디 분리)

## 빠른 시작

    from note_recognition import detect_notes, save_musicxml

    result = detect_notes(img_gray, staff_top_y, staff_bot_y,
                          staff_gap=20, line_thickness=3,
                          x_start=60)          # 헤더 영역 제외
    save_musicxml(result, "output.musicxml",
                  barlines=[500, 1000],         # StaffZone.barlines
                  key_sig=1)                    # G장조
"""

from note_recognition.note_detector import (
    DetectedNote,
    DetectedRest,
    NoteDetectionResult,
    detect_notes,
)
from note_recognition.note_pitcher import Pitch, head_y_to_pitch, staff_step_to_pitch
from note_recognition.key_signature import (
    apply_key_signature, get_accidental_map, MeasureAccidentalState,
)
from note_recognition.xml_builder import notes_to_score, save_musicxml
from note_recognition.staff_removal import (
    detect_staff_line_thickness,
    remove_staff_lines,
)

__all__ = [
    "DetectedNote", "DetectedRest", "NoteDetectionResult", "detect_notes",
    "Pitch", "head_y_to_pitch", "staff_step_to_pitch",
    "apply_key_signature", "get_accidental_map", "MeasureAccidentalState",
    "notes_to_score", "save_musicxml",
    "detect_staff_line_thickness", "remove_staff_lines",
]
