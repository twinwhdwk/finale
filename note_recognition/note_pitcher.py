"""
음높이(pitch) 판정 모듈.

파이프라인 4단계: detect_notes()가 반환한 DetectedNote의 head_y를
오선 좌표(staff_top_y, staff_gap)와 결합해 음이름(pitch name)을 결정한다.

## 좌표계 및 음 매핑 규칙

staff_step은 synthetic_score.py / pdf_parser.py와 동일한 규칙:
  - step 0 = 오선 맨 아래줄(line4) = E4 (높은음자리표 기준)
  - step +1 = 한 칸 위(F4), step +2 = G4, ...
  - step -1 = 아래로 한 칸(D4), step -2 = C4, ...
  - 7 step = 1 옥타브 상승

head_y → staff_step 변환:
  line4_y = staff_top_y + 4 * staff_gap
  staff_step = round((line4_y - head_y) * 2 / staff_gap)

  `round`가 중요: head_y에는 1~2픽셀 오차가 있으므로 반올림으로
  가장 가까운 줄/칸 위치로 스냅한다.

## 음자리표 지원

현재 높은음자리표(treble clef, G clef)만 지원.
낮은음자리표(bass clef)는 step 0 = G2로 기준이 다름 → 추후 확장.
베이스 클레프는 인자로 clef='bass' 넘기면 자동 보정.
"""

from dataclasses import dataclass


# ── 음자리표별 기준 매핑 ──────────────────────────────────────────────

# 높은음자리표: step 0 (맨 아래줄) = E4
# 음 이름은 온음계 순서 (EFGABCD 반복)
_TREBLE_STEP0_STEP   = 0       # 기준 step
_TREBLE_STEP0_NAME   = "E"
_TREBLE_STEP0_OCTAVE = 4

# 낮은음자리표: step 0 (맨 아래줄) = G2
_BASS_STEP0_STEP   = 0
_BASS_STEP0_NAME   = "G"
_BASS_STEP0_OCTAVE = 2

# 음 이름 순서 (온음계, 반음계 무시)
_DIATONIC = ["C", "D", "E", "F", "G", "A", "B"]
_DIATONIC_INDEX = {n: i for i, n in enumerate(_DIATONIC)}


@dataclass(frozen=True)
class Pitch:
    """음높이 정보."""
    step: str           # 음이름 (C~B)
    octave: int         # 옥타브 (4=중간 C 근방)
    staff_step: int     # 오선 기준 스텝 (디버그/pitch 매핑 검증용)
    accidental: str = ""  # "", "#", "b" (현재는 항상 "" - 임시표 미구현)

    @property
    def name_with_octave(self) -> str:
        """music21 호환 형식: 'C4', 'F#5' 등."""
        return f"{self.step}{self.accidental}{self.octave}"

    @property
    def midi_note(self) -> int:
        """MIDI 노트 번호 (C4=60 기준). 반음계(accidental) 반영."""
        semitones = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
        acc_offset = {"#": 1, "b": -1, "": 0}
        return (self.octave + 1) * 12 + semitones[self.step] + acc_offset.get(self.accidental, 0)


def head_y_to_staff_step(head_y: float, staff_top_y: int, staff_gap: int) -> int:
    """
    음표머리 y좌표를 오선 기준 staff_step으로 변환한다.

    Args:
        head_y:       음표머리 중심 y좌표 (DetectedNote.head_y)
        staff_top_y:  오선 맨 위줄(line0) y좌표
        staff_gap:    인접 오선 줄 사이 간격 (픽셀)

    Returns:
        staff_step (0=맨 아래줄E4, 양수=위, 음수=아래)
    """
    line4_y = staff_top_y + 4 * staff_gap
    raw = (line4_y - head_y) * 2 / staff_gap
    return int(round(raw))


def staff_step_to_pitch(staff_step: int, clef: str = "treble") -> Pitch:
    """
    staff_step을 Pitch로 변환한다.

    Args:
        staff_step: 오선 기준 스텝 (head_y_to_staff_step 결과)
        clef:       "treble" (높은음자리표) | "bass" (낮은음자리표)

    Returns:
        Pitch 객체
    """
    if clef == "bass":
        base_name, base_octave = _BASS_STEP0_NAME, _BASS_STEP0_OCTAVE
    else:
        base_name, base_octave = _TREBLE_STEP0_NAME, _TREBLE_STEP0_OCTAVE

    base_idx = _DIATONIC_INDEX[base_name]  # E=4, G=6

    # step만큼 위로 이동하면 diatonic index가 증가
    total_idx = base_idx + staff_step      # 음의 절대 diatonic 인덱스

    note_name = _DIATONIC[total_idx % 7]
    # 옥타브: base_octave + (음의 C 기준 위치 변화)
    # C의 index = 0. total_idx가 0 이상이면 정상 범위,
    # C를 지날 때마다 옥타브가 올라감
    octave = base_octave + (total_idx // 7)

    return Pitch(step=note_name, octave=octave, staff_step=staff_step)


def head_y_to_pitch(
    head_y: float,
    staff_top_y: int,
    staff_gap: int,
    clef: str = "treble",
) -> Pitch:
    """head_y → Pitch 원스텝 변환 (head_y_to_staff_step + staff_step_to_pitch)."""
    step = head_y_to_staff_step(head_y, staff_top_y, staff_gap)
    return staff_step_to_pitch(step, clef=clef)
