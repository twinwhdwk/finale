"""
조표(key signature) 적용 모듈.

note_pitcher.py가 반환한 Pitch(accidental="")에 조표 정보를 반영해
실제 음이름(예: F# 조에서 F → F#)으로 보정한다.

## 조표 시스템 (Circle of Fifths)

샵(#) 조표: 붙는 순서 F → C → G → D → A → E → B
플랫(b) 조표: 붙는 순서 B → E → A → D → G → C → F

key_sig 값:
  양수 = 샵 개수 (예: 1=G장조, 2=D장조, ...)
  음수 = 플랫 개수 (예: -1=F장조, -2=Bb장조, ...)
  0 = C장조 (임시표 없음)

## 임시표(accidental) 처리 우선순위

1. 마디 내 임시표(in-measure accidental): 같은 마디 내 앞선 음표에서
   발생한 임시표는 해당 마디 끝까지 유효.
2. 조표(key signature): 임시표가 없으면 조표 적용.
3. 제자리표(natural): 조표를 무효화하는 기호 (현재 미인식, TODO).

현재 구현은 조표만 지원하고 임시표/제자리표는 미구현.
"""

from note_recognition.note_pitcher import Pitch

# 샵 붙는 순서 (5도권)
_SHARPS_ORDER = ["F", "C", "G", "D", "A", "E", "B"]
# 플랫 붙는 순서
_FLATS_ORDER  = ["B", "E", "A", "D", "G", "C", "F"]


def get_accidental_map(key_sig: int) -> dict[str, str]:
    """
    key_sig 값으로부터 음이름 → accidental 딕셔너리를 반환한다.

    Args:
        key_sig: 양수=샵 개수, 음수=플랫 개수, 0=없음

    Returns:
        {"F": "#", "C": "#"} 형태. 해당 없는 음이름은 포함되지 않음.
    """
    if key_sig > 0:
        affected = _SHARPS_ORDER[:min(key_sig, 7)]
        return {note: "#" for note in affected}
    elif key_sig < 0:
        affected = _FLATS_ORDER[:min(-key_sig, 7)]
        return {note: "b" for note in affected}
    return {}


def apply_key_signature(pitch: Pitch, key_sig: int) -> Pitch:
    """
    조표를 Pitch에 적용해 accidental이 반영된 새 Pitch를 반환한다.

    이미 accidental이 있으면(임시표) 조표를 무시한다.

    Args:
        pitch:   note_pitcher.py가 반환한 Pitch
        key_sig: 조표 (양수=샵, 음수=플랫, 0=없음)

    Returns:
        accidental이 반영된 Pitch (변경 없으면 입력 그대로)
    """
    if pitch.accidental:
        return pitch  # 이미 임시표가 있으면 유지
    acc_map = get_accidental_map(key_sig)
    acc = acc_map.get(pitch.step, "")
    if not acc:
        return pitch
    return Pitch(step=pitch.step, octave=pitch.octave,
                 staff_step=pitch.staff_step, accidental=acc)


def apply_key_signature_to_pitches(
    pitches: list[Pitch],
    key_sig: int,
) -> list[Pitch]:
    """음높이 목록 전체에 조표를 적용한다."""
    return [apply_key_signature(p, key_sig) for p in pitches]


class MeasureAccidentalState:
    """
    마디 내 임시표 상태 기계.

    같은 마디 안에서 임시표(# 또는 b)가 붙은 음표가 나오면 해당 음이름의
    상태를 기억해 그 마디가 끝날 때까지 유효하게 유지한다. 마디 경계에서
    초기화한다.

    ## 우선순위 (표준 음악 이론)

    1. 이 마디에서 이미 등장한 임시표 (in-measure state) → 최우선
    2. 조표(key signature)
    3. 임시표/조표 없음 → 그대로

    ## 사용법

    ```python
    state = MeasureAccidentalState(key_sig=2)  # D장조(F#, C#)

    # 새 마디 시작할 때
    state.reset()

    # 각 음표를 처리할 때
    pitch = state.apply(raw_pitch)
    ```

    ## 현재 한계

    제자리표(natural, ♮) 미인식: 조표가 있는 음에 제자리표가 붙으면
    해당 마디에서 조표가 무효화되어야 하는데, 현재 OpenCV 파이프라인은
    제자리표 기호 자체를 인식하지 못하므로 처리할 수 없음 (TODO).
    """

    def __init__(self, key_sig: int = 0):
        self._key_sig = key_sig
        self._key_map: dict[str, str] = get_accidental_map(key_sig)
        # 이 마디에서 이미 나온 임시표: {"C": "#", "B": "b", ...}
        self._measure_state: dict[str, str] = {}

    def reset(self) -> None:
        """마디 경계: 마디 내 임시표 기억을 지운다. 조표는 유지."""
        self._measure_state.clear()

    def apply(self, pitch: Pitch) -> Pitch:
        """
        pitch에 우선순위에 따라 임시표/조표를 적용한 새 Pitch를 반환한다.
        음표에 이미 임시표가 명시돼 있으면 그 값을 상태에 기록하고 그대로 사용.
        """
        note_name = pitch.step

        if pitch.accidental:
            # 이 음표 자체에 임시표가 명시돼 있음 → 상태 업데이트 후 그대로
            self._measure_state[note_name] = pitch.accidental
            return pitch

        # 마디 내 앞선 임시표가 있으면 우선 적용
        if note_name in self._measure_state:
            acc = self._measure_state[note_name]
            return Pitch(step=note_name, octave=pitch.octave,
                         staff_step=pitch.staff_step, accidental=acc)

        # 조표 적용
        acc = self._key_map.get(note_name, "")
        if acc:
            return Pitch(step=note_name, octave=pitch.octave,
                         staff_step=pitch.staff_step, accidental=acc)

        return pitch
