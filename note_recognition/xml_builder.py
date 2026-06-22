"""
MusicXML 생성 모듈.

파이프라인 5단계 (최종): note_detector.py의 NoteDetectionResult와
note_pitcher.py의 pitch 판정을 결합해 music21 Score를 구성하고
.musicxml 파일로 저장한다.

## 입력 → 출력 흐름

  NoteDetectionResult (오선별 DetectedNote 목록)
    ↓ head_y_to_pitch()로 각 음표의 pitch 판정
  music21 stream.Score
    ↓ Score.write('musicxml')
  .musicxml 파일 (Audiveris 결과물과 동일 형식 → xml_comparator.compare()에 바로 투입 가능)

## 마디 구성 방식

현재 구현은 마디선 위치를 알지 못하므로 음표를 박자 단위로만 순서
나열한다. 4/4박자 기준으로 누적 박자가 마디 길이를 넘으면 다음 마디로
이동하는 단순 방식 (barline 위치를 모르는 것이 한계 - 향후 pdf_parser의
barlines 정보와 연동 예정).

## 알려진 한계

- 임시표(accidentals) 미처리: Pitch.accidental이 항상 ""
  (조표/임시표 인식은 별도 단계 필요 - TODO)
- 쉼표(rest) 미인식: 현재 DetectedNote에 쉼표 케이스가 없음
- 점음표(dotted) 미처리: duration이 정확히 whole/half/quarter/eighth/
  sixteenth 중 하나여야 함
- 코드(chord) 미처리: 화음(동시에 울리는 여러 음표)은 현재 개별 음표로 처리
- 2성부(두 개의 Part) 미분리: 이성부 악보는 하나의 Part로 합쳐짐
  (pdf_parser.py의 성부 분리 로직과 연동 필요)
"""

import math
from pathlib import Path

from music21 import clef, duration, meter, note, stream

from note_recognition.note_detector import DetectedNote, NoteDetectionResult
from note_recognition.note_pitcher import head_y_to_pitch


# duration 문자열 → quarterLength 변환
_DURATION_TO_QUARTER = {
    "whole":      4.0,
    "half":       2.0,
    "quarter":    1.0,
    "eighth":     0.5,
    "sixteenth":  0.25,
}

# music21 duration type 이름
_DURATION_TYPE = {
    "whole":      "whole",
    "half":       "half",
    "quarter":    "quarter",
    "eighth":     "eighth",
    "sixteenth":  "16th",
}


def notes_to_score(
    detection_result: NoteDetectionResult,
    time_sig: str = "4/4",
    clef_type: str = "treble",
    part_name: str = "Part 1",
) -> stream.Score:
    """
    NoteDetectionResult → music21 Score.

    Args:
        detection_result: detect_notes()의 반환값
        time_sig:         박자표 문자열 (기본 "4/4")
        clef_type:        "treble" | "bass"
        part_name:        파트 이름

    Returns:
        music21 stream.Score (악보 전체)
    """
    ts = meter.TimeSignature(time_sig)
    measure_length = ts.barDuration.quarterLength  # 4/4 → 4.0

    s = stream.Score()
    p = stream.Part(id=part_name)

    current_measure = stream.Measure(number=1)
    current_measure.append(
        clef.TrebleClef() if clef_type == "treble" else clef.BassClef()
    )
    current_measure.append(ts)

    accumulated = 0.0  # 현재 마디 내 누적 박자

    for detected in detection_result.notes:
        ql = _DURATION_TO_QUARTER.get(detected.duration, 1.0)
        if detected.is_dotted:
            ql *= 1.5

        # 이 음표를 추가하면 마디가 넘치는지 확인 (허용 오차 1e-6)
        if accumulated + ql > measure_length + 1e-6:
            # 현재 마디 마감 후 새 마디 시작
            p.append(current_measure)
            current_measure = stream.Measure(number=current_measure.number + 1)
            accumulated = 0.0

        pitch = head_y_to_pitch(
            detected.head_y,
            detection_result.staff_top_y,
            detection_result.staff_gap,
            clef=clef_type,
        )

        n = note.Note(pitch.name_with_octave)
        if detected.is_dotted:
            n.duration = duration.Duration(_DURATION_TYPE[detected.duration])
            n.duration.dots = 1
        else:
            n.duration = duration.Duration(_DURATION_TYPE[detected.duration])

        current_measure.append(n)
        accumulated += ql

        # 마디가 딱 맞으면 마감
        if math.isclose(accumulated, measure_length, abs_tol=1e-6):
            p.append(current_measure)
            current_measure = stream.Measure(number=current_measure.number + 1)
            accumulated = 0.0

    # 마지막 마디가 남아있으면 추가
    if len(current_measure) > 0:
        p.append(current_measure)

    # 쉼표(DetectedRest) 처리
    # 현재 구현은 음표 목록 뒤에 별도 마디로 추가하는 단순 방식.
    # TODO: 음표와 쉼표를 x좌표 순서로 통합 정렬해 올바른 마디 구성
    if detection_result.rests:
        rest_measure = stream.Measure(number=len(p.getElementsByClass("Measure")) + 1)
        for detected_rest in detection_result.rests:
            ql = _DURATION_TO_QUARTER.get(detected_rest.duration, 1.0)
            r = note.Rest()
            r.duration = duration.Duration(_DURATION_TYPE.get(detected_rest.duration, "quarter"))
            rest_measure.append(r)
        p.append(rest_measure)

    s.append(p)
    return s


def save_musicxml(
    detection_result: NoteDetectionResult,
    output_path: str,
    time_sig: str = "4/4",
    clef_type: str = "treble",
) -> str:
    """
    NoteDetectionResult를 .musicxml 파일로 저장한다.

    Args:
        detection_result: detect_notes()의 반환값
        output_path:      저장할 .musicxml 경로
        time_sig:         박자표 (기본 "4/4")
        clef_type:        음자리표 (기본 "treble")

    Returns:
        저장된 파일 경로 (output_path 그대로)
    """
    score = notes_to_score(detection_result, time_sig=time_sig, clef_type=clef_type)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    score.write("musicxml", fp=output_path)
    return output_path
