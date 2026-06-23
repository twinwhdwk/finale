"""
MusicXML 생성 모듈.

파이프라인 5단계 (최종): note_detector.py의 NoteDetectionResult와
note_pitcher.py의 pitch 판정을 결합해 music21 Score를 구성하고
.musicxml 파일로 저장한다.

## 마디 구성 방식

1. barlines(마디선 x좌표 목록)가 주어지면 → 마디선 기반 정확한 분리
2. barlines가 없으면 → 박자 누산(fallback) 방식

음표와 쉼표는 head_x / center_x 기준으로 통합 정렬해 올바른 순서로 배치.

## 알려진 한계

- 임시표(in-measure accidental): MeasureAccidentalState로 마디 내 임시표
  전파 지원. 단 제자리표(natural, ♮) 미지원 (Pitch.accidental=""가
  '임시표 없음'과 구분 불가 — OpenCV 파이프라인에서 제자리표 기호 미인식)
- 코드(chord): 화음은 개별 음표로 처리
- 2성부: 이성부 악보는 하나의 Part로 합쳐짐
- 4분/8분쉼표: note_detector._classify_rest가 선형 쉼표 탐지 지원.
  단 합성 이미지에서 8분쉼표가 4분쉼표로 오분류될 수 있음 (실측 후 조정 필요)
"""

import math
from pathlib import Path
from dataclasses import dataclass
from typing import Union

from music21 import clef, duration, key, meter, note, stream

from note_recognition.note_detector import DetectedNote, DetectedRest, NoteDetectionResult
from note_recognition.note_pitcher import head_y_to_pitch


_DURATION_TO_QUARTER = {
    "whole": 4.0, "half": 2.0, "quarter": 1.0,
    "eighth": 0.5, "sixteenth": 0.25,
}

_DURATION_TYPE = {
    "whole": "whole", "half": "half", "quarter": "quarter",
    "eighth": "eighth", "sixteenth": "16th",
}


# ── 통합 이벤트 ───────────────────────────────────────────────────────

@dataclass
class _Event:
    """음표 또는 쉼표를 x좌표 기준으로 통합 정렬하기 위한 래퍼."""
    x: int
    item: Union[DetectedNote, DetectedRest]


def _make_events(result: NoteDetectionResult) -> list[_Event]:
    """음표와 쉼표를 x좌표 기준으로 통합 정렬한 이벤트 목록을 반환."""
    events = []
    for n in result.notes:
        events.append(_Event(x=n.head_x, item=n))
    for r in result.rests:
        events.append(_Event(x=r.center_x, item=r))
    events.sort(key=lambda e: e.x)
    return events


def _event_to_music21(
    event: _Event,
    result: NoteDetectionResult,
    clef_type: str,
    key_sig: int = 0,
    acc_state=None,   # MeasureAccidentalState | None
) -> note.GeneralNote:
    """_Event → music21 Note 또는 Rest.

    acc_state가 주어지면 마디 내 임시표 상태 기계를 통해 pitch를 보정한다.
    None이면 기존 방식(key_sig만 적용)으로 fallback.
    """
    from note_recognition.key_signature import apply_key_signature
    item = event.item
    if isinstance(item, DetectedNote):
        ql = _DURATION_TO_QUARTER.get(item.duration, 1.0)
        if item.is_dotted:
            ql *= 1.5
        pitch = head_y_to_pitch(
            item.head_y, result.staff_top_y, result.staff_gap, clef=clef_type
        )
        if acc_state is not None:
            pitch = acc_state.apply(pitch)
        elif key_sig:
            pitch = apply_key_signature(pitch, key_sig)
        n = note.Note(pitch.name_with_octave)
        n.duration = duration.Duration(_DURATION_TYPE[item.duration])
        if item.is_dotted:
            n.duration.dots = 1
        return n
    else:  # DetectedRest
        r = note.Rest()
        r.duration = duration.Duration(_DURATION_TYPE.get(item.duration, "quarter"))
        return r


def _ql_of_event(event: _Event) -> float:
    item = event.item
    if isinstance(item, DetectedNote):
        ql = _DURATION_TO_QUARTER.get(item.duration, 1.0)
        return ql * 1.5 if item.is_dotted else ql
    return _DURATION_TO_QUARTER.get(item.duration, 1.0)


# ── 마디 분리 전략 ────────────────────────────────────────────────────

def _assign_events_to_measures_by_barlines(
    events: list[_Event],
    barlines: list[int],
) -> dict[int, list[_Event]]:
    """
    마디선 x좌표 목록을 기준으로 각 이벤트를 마디 번호(1-based)에 배정.

    barlines = [500, 1000] 이면 마디 경계:
      마디1: x < 500
      마디2: 500 ≤ x < 1000
      마디3: x ≥ 1000
    """
    measures: dict[int, list[_Event]] = {}
    for ev in events:
        m_idx = 0  # 0-based
        for bx in barlines:
            if ev.x >= bx:
                m_idx += 1
            else:
                break
        m_num = m_idx + 1  # 1-based
        measures.setdefault(m_num, []).append(ev)
    return measures


def _assign_events_to_measures_by_beat(
    events: list[_Event],
    measure_length: float,
) -> dict[int, list[_Event]]:
    """
    박자 누산 방식으로 각 이벤트를 마디 번호에 배정 (barlines 없을 때 fallback).
    """
    measures: dict[int, list[_Event]] = {}
    accumulated = 0.0
    m_num = 1
    for ev in events:
        ql = _ql_of_event(ev)
        if accumulated + ql > measure_length + 1e-6:
            m_num += 1
            accumulated = 0.0
        measures.setdefault(m_num, []).append(ev)
        accumulated += ql
        if math.isclose(accumulated, measure_length, abs_tol=1e-6):
            m_num += 1
            accumulated = 0.0
    return measures


# ── 공개 API ──────────────────────────────────────────────────────────

def notes_to_score(
    detection_result: NoteDetectionResult,
    time_sig: str = "4/4",
    clef_type: str = "treble",
    part_name: str = "Part 1",
    barlines: list[int] | None = None,
    key_sig: int = 0,
) -> stream.Score:
    """
    NoteDetectionResult → music21 Score.

    Args:
        detection_result: detect_notes()의 반환값
        time_sig:         박자표 문자열 (기본 "4/4")
        clef_type:        "treble" | "bass"
        part_name:        파트 이름
        barlines:         마디선 x좌표 목록 (StaffZone.barlines).
                          None이면 박자 누산 방식(fallback)으로 마디 분리.
        key_sig:          조표 (샵 개수, 음수=플랫. 기본 0=C장조).

    Returns:
        music21 stream.Score
    """
    ts = meter.TimeSignature(time_sig)
    measure_length = ts.barDuration.quarterLength

    events = _make_events(detection_result)

    if not events:
        s = stream.Score()
        s.append(stream.Part(id=part_name))
        return s

    # 마디별 이벤트 배정
    if barlines:
        measure_map = _assign_events_to_measures_by_barlines(events, barlines)
    else:
        measure_map = _assign_events_to_measures_by_beat(events, measure_length)

    from note_recognition.key_signature import MeasureAccidentalState
    acc_state = MeasureAccidentalState(key_sig=key_sig)

    s = stream.Score()
    p = stream.Part(id=part_name)

    for m_num in sorted(measure_map.keys()):
        m = stream.Measure(number=m_num)
        if m_num == 1:
            m.append(clef.TrebleClef() if clef_type == "treble" else clef.BassClef())
            if key_sig != 0:
                m.append(key.KeySignature(key_sig))
            m.append(ts)
        # 마디 시작: 임시표 상태 초기화 (조표는 유지)
        acc_state.reset()
        for ev in measure_map[m_num]:
            m.append(_event_to_music21(ev, detection_result, clef_type,
                                       acc_state=acc_state))
        p.append(m)

    s.append(p)
    return s


def save_musicxml(
    detection_result: NoteDetectionResult,
    output_path: str,
    time_sig: str = "4/4",
    clef_type: str = "treble",
    barlines: list[int] | None = None,
    key_sig: int = 0,
) -> str:
    """
    NoteDetectionResult를 .musicxml 파일로 저장한다.

    Args:
        detection_result: detect_notes()의 반환값
        output_path:      저장할 .musicxml 경로
        time_sig:         박자표
        clef_type:        음자리표
        barlines:         마디선 x좌표 목록 (StaffZone.barlines).
                          None이면 박자 누산 방식으로 마디 분리.
        key_sig:          조표 (샵 개수, 음수=플랫)

    Returns:
        저장된 파일 경로
    """
    score = notes_to_score(
        detection_result, time_sig=time_sig, clef_type=clef_type,
        barlines=barlines, key_sig=key_sig,
    )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    score.write("musicxml", fp=output_path)
    return output_path

