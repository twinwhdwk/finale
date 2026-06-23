"""
MusicXML에서 마디별 음표 음높이 추출 (music21 없이).

비교용 다이어토닉 음높이(조표·임시표 제외)만 추출합니다.
note_detector.py의 이미지 감지 결과와 동일한 포맷으로 반환.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path


def _read_mxl(mxl_path: str) -> str:
    p = Path(mxl_path)
    if p.suffix.lower() == '.mxl':
        with zipfile.ZipFile(mxl_path) as z:
            names = z.namelist()
            entry = next(
                (n for n in names
                 if n.endswith(('.musicxml', '.xml')) and 'META' not in n),
                None,
            )
            if not entry:
                raise RuntimeError(f"MXL 내부 악보 파일 없음: {mxl_path}")
            return z.read(entry).decode('utf-8')
    return p.read_text(encoding='utf-8')


def extract_score_info(
    mxl_path: str,
    part_idx: int = 0,
) -> tuple[str, dict[int, list[str]]]:
    """
    MusicXML 파일에서 음자리표와 마디별 다이어토닉 음높이를 추출합니다.

    Returns:
        clef: 'G' | 'F' | 'C'
        notes_by_measure: {마디번호: ['G4', 'A4', ...]}
            조표·임시표는 무시한 다이어토닉 음높이만 포함.
            쉼표 제외, 타이 연음 포함 (악보 이미지에 음표 머리가 보이므로).
    """
    xml_str = _read_mxl(mxl_path)
    root    = ET.fromstring(xml_str)

    parts = root.findall('part')
    if not parts:
        return 'G', {}
    part = parts[min(part_idx, len(parts) - 1)]

    clef: str = 'G'
    notes_by_measure: dict[int, list[str]] = {}

    for measure in part.findall('measure'):
        try:
            m_num = int(measure.get('number', 1))
        except ValueError:
            continue

        attrs = measure.find('attributes')
        if attrs is not None:
            c = attrs.find('clef')
            if c is not None:
                s = c.find('sign')
                if s is not None:
                    clef = s.text.strip()

        pitches: list[str] = []

        for elem in measure:
            if elem.tag != 'note':
                continue
            if elem.find('rest') is not None:
                continue

            pitch_elem = elem.find('pitch')
            if pitch_elem is None:
                continue

            step_elem = pitch_elem.find('step')
            oct_elem  = pitch_elem.find('octave')
            if step_elem is None or oct_elem is None:
                continue

            step   = step_elem.text.strip()
            octave = oct_elem.text.strip()
            pitches.append(f"{step}{octave}")

        if pitches:
            notes_by_measure[m_num] = pitches

    return clef, notes_by_measure


_HOLLOW_TYPES = {'whole', 'half'}


def extract_note_types(
    mxl_path: str,
    part_idx: int = 0,
) -> dict[int, list[bool]]:
    """
    MusicXML에서 마디별 음표 hollow 여부 추출.
    extract_score_info와 동일한 필터링(쉼표 제외, 타이 연음 포함).

    Returns:
        {마디번호: [is_hollow, ...]}
        is_hollow=True: 2분음표·온음표
    """
    xml_str = _read_mxl(mxl_path)
    root    = ET.fromstring(xml_str)

    parts = root.findall('part')
    if not parts:
        return {}
    part = parts[min(part_idx, len(parts) - 1)]

    hollows_by_measure: dict[int, list[bool]] = {}

    for measure in part.findall('measure'):
        try:
            m_num = int(measure.get('number', 1))
        except ValueError:
            continue

        types: list[bool] = []

        for elem in measure:
            if elem.tag != 'note':
                continue
            if elem.find('rest') is not None:
                continue
            if elem.find('pitch') is None:
                continue

            type_elem = elem.find('type')
            note_type = type_elem.text.strip() if type_elem is not None and type_elem.text else 'quarter'
            types.append(note_type in _HOLLOW_TYPES)

        if types:
            hollows_by_measure[m_num] = types

    return hollows_by_measure


def extract_ties(
    mxl_path: str,
    part_idx: int = 0,
) -> dict[int, list[str]]:
    """
    마디별 음표 타이 상태 추출.

    Returns:
        {마디번호: [tie_status, ...]}
        tie_status: '' | 'start' | 'stop' | 'both'
    """
    xml_str = _read_mxl(mxl_path)
    root    = ET.fromstring(xml_str)

    parts = root.findall('part')
    if not parts:
        return {}
    part = parts[min(part_idx, len(parts) - 1)]

    ties_by_measure: dict[int, list[str]] = {}

    for measure in part.findall('measure'):
        try:
            m_num = int(measure.get('number', 1))
        except ValueError:
            continue

        statuses: list[str] = []

        for elem in measure:
            if elem.tag != 'note':
                continue
            if elem.find('rest') is not None:
                continue
            if elem.find('pitch') is None:
                continue

            tie_types = {t.get('type', '') for t in elem.findall('tie')}
            if 'start' in tie_types and 'stop' in tie_types:
                statuses.append('both')
            elif 'start' in tie_types:
                statuses.append('start')
            elif 'stop' in tie_types:
                statuses.append('stop')
            else:
                statuses.append('')

        if statuses:
            ties_by_measure[m_num] = statuses

    return ties_by_measure


def is_accompaniment_xml(notes_by_measure: dict[int, list[str]], threshold: float = 0.55) -> bool:
    """
    반주 전용 XML 패턴 감지.
    마디당 음표 3개 이상인 마디에서 고유음/전체음 비율이 낮으면 반주 패턴으로 판정.
    """
    suspects = 0
    checked  = 0
    for notes in notes_by_measure.values():
        if len(notes) < 3:
            continue
        checked += 1
        variety = len(set(notes)) / len(notes)
        if variety < threshold:
            suspects += 1
    if checked == 0:
        return False
    return (suspects / checked) > 0.5


def _harmony_to_str(harmony_elem) -> str:
    """<harmony> 요소 → "Am", "G7" 등 코드 문자열."""
    root = harmony_elem.find('root')
    if root is None:
        return ''
    step_el  = root.find('root-step')
    alter_el = root.find('root-alter')
    root_str = (step_el.text or '').strip() if step_el is not None else ''
    if alter_el is not None and alter_el.text:
        try:
            a = int(float(alter_el.text))
            root_str += '#' if a == 1 else ('b' if a == -1 else '')
        except ValueError:
            pass

    kind_el  = harmony_elem.find('kind')
    kind_str = ''
    if kind_el is not None:
        kind_str = kind_el.get('text', '')
        if not kind_str:
            _KIND = {
                'major': '',          'minor': 'm',       'dominant': '7',
                'major-seventh': 'maj7', 'minor-seventh': 'm7',
                'diminished': 'dim',  'augmented': 'aug',
                'half-diminished': 'm7b5', 'diminished-seventh': 'dim7',
                'suspended-fourth': 'sus4', 'suspended-second': 'sus2',
                'major-minor': 'mM7',
            }
            kind_str = _KIND.get((kind_el.text or '').strip(), '')

    bass_el  = harmony_elem.find('bass')
    bass_str = ''
    if bass_el is not None:
        bs = bass_el.find('bass-step')
        ba = bass_el.find('bass-alter')
        if bs is not None and bs.text:
            bass_str = '/' + bs.text.strip()
            if ba is not None and ba.text:
                try:
                    a = int(float(ba.text))
                    bass_str += '#' if a == 1 else ('b' if a == -1 else '')
                except ValueError:
                    pass

    return root_str + kind_str + bass_str


def extract_lyrics(mxl_path: str, part_idx: int = 0) -> dict[int, list[str]]:
    """마디별 가사. {마디번호: ["1절 가사", "2절 가사"]}"""
    xml_str = _read_mxl(mxl_path)
    root    = ET.fromstring(xml_str)
    parts   = root.findall('part')
    if not parts:
        return {}
    part = parts[min(part_idx, len(parts) - 1)]

    result: dict[int, list[str]] = {}
    for measure in part.findall('measure'):
        try:
            m_num = int(measure.get('number', 1))
        except ValueError:
            continue

        lines: dict[str, list[str]] = {}
        for note in measure.findall('note'):
            if note.find('rest') is not None:
                continue
            for lyric in note.findall('lyric'):
                num     = lyric.get('number', '1')
                text_el = lyric.find('text')
                if text_el is None or not text_el.text:
                    continue
                syl_el = lyric.find('syllabic')
                syl    = (syl_el.text or 'single').strip() if syl_el is not None else 'single'
                text   = text_el.text.strip()
                lines.setdefault(num, []).append(text + '-' if syl in ('begin', 'middle') else text)

        if lines:
            sorted_keys = sorted(lines.keys(), key=lambda k: int(k) if k.isdigit() else 0)
            result[m_num] = [''.join(lines[k]) for k in sorted_keys]
    return result


def extract_chords(mxl_path: str) -> dict[int, list[str]]:
    """마디별 코드 기호 목록. {마디번호: ["Am", "G7", ...]}"""
    xml_str = _read_mxl(mxl_path)
    root    = ET.fromstring(xml_str)
    parts   = root.findall('part')
    if not parts:
        return {}
    part = parts[0]

    result: dict[int, list[str]] = {}
    for measure in part.findall('measure'):
        try:
            m_num = int(measure.get('number', 1))
        except ValueError:
            continue
        chords = [_harmony_to_str(h) for h in measure.findall('harmony')]
        chords = [c for c in chords if c]
        if chords:
            result[m_num] = chords
    return result


def extract_measure_count(mxl_path: str, part_idx: int = 0) -> int:
    """MXL 파일의 총 마디 수."""
    xml_str = _read_mxl(mxl_path)
    root    = ET.fromstring(xml_str)
    parts   = root.findall('part')
    if not parts:
        return 0
    return len(parts[min(part_idx, len(parts) - 1)].findall('measure'))
