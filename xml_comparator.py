"""
MusicXML 비교 모듈 (music21 기반)

트랙 1: 음표 (Note/Rest/Chord) - OMR 추출 XML vs 피날레 원본 XML
트랙 2: 코드 기호 (ChordSymbol) - PDF OCR vs 피날레 원본 XML
트랙 3: 가사 (Lyric) - PDF OCR vs 피날레 원본 XML
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from pathlib import Path
from music21 import converter, note, chord, stream
from music21.note import Rest
from music21.harmony import ChordSymbol
from music21.expressions import TextExpression

_VERSE_NUM = re.compile(r'^\d+\.\s*')   # Finale XML 절 번호 접두사 ("1.아" → "아")
_LYRIC_SIM_THRESHOLD = 0.30              # 한글 Jaccard 유사도 하한


# ── 불일치 타입 정의 ──────────────────────────────────────────────────

KIND_LABELS = {
    "pitch":        "[음높이 오류]",
    "duration":     "[음길이 오류]",
    "type":         "[형식 오류]",
    "tie_suspect":  "[붙임줄 의심]",
    "measure_miss": "[마디 누락]",
    "missing":      "[누락]",
    "noise":        "[노이즈]",
    "chord_miss":   "[코드 누락]",
    "chord_diff":   "[코드 불일치]",
    "lyric_miss":   "[가사 누락]",
    "lyric_diff":   "[가사 불일치]",
}


@dataclass
class Discrepancy:
    kind:    str
    measure: int
    offset:  float
    message: str
    track:   str = "note"   # "note" | "chord" | "lyric"

    def label(self) -> str:
        return KIND_LABELS.get(self.kind, f"[{self.kind}]")

    def __str__(self) -> str:
        if self.kind == "measure_miss":
            return f"{self.label()} [마디 {self.measure}] {self.message}"
        return f"{self.label()} [마디 {self.measure} | {self.offset:.2f}박자] {self.message}"


@dataclass
class CompareResult:
    pdf_xml:  str
    orig_xml: str
    total_measures: int
    discrepancies: list[Discrepancy] = field(default_factory=list)

    # 트랙별 집계
    def _by_kind(self, *kinds) -> int:
        return sum(1 for d in self.discrepancies if d.kind in kinds)

    @property
    def note_errors(self)      -> int: return self._by_kind("pitch", "duration", "type", "tie_suspect")
    @property
    def missing_count(self)    -> int: return self._by_kind("missing")
    @property
    def noise_count(self)      -> int: return self._by_kind("noise")
    @property
    def tie_suspect_count(self)-> int: return self._by_kind("tie_suspect")
    @property
    def measure_miss_count(self)-> int: return self._by_kind("measure_miss")
    @property
    def note_total(self)       -> int: return sum(1 for d in self.discrepancies if d.track == "note")
    @property
    def chord_errors(self)     -> int: return self._by_kind("chord_miss", "chord_diff")
    @property
    def lyric_errors(self)     -> int: return self._by_kind("lyric_miss", "lyric_diff")
    @property
    def is_perfect(self)       -> bool: return len(self.discrepancies) == 0

    def summary(self) -> str:
        return "\n".join([
            f"PDF 추출본: {self.pdf_xml}",
            f"원본:       {self.orig_xml}",
            f"총 마디:    {self.total_measures}",
            f"음표 오류:  {self.note_errors}  (누락 {self.missing_count} / 노이즈 {self.noise_count})",
            f"코드 오류:  {self.chord_errors}",
            f"가사 오류:  {self.lyric_errors}",
        ])


# ── 내부 유틸 ─────────────────────────────────────────────────────────

def _load(path: str) -> stream.Score:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"파일 없음: {path}")
    print(f"  로딩: {p.name}")
    return converter.parse(str(p))


def _element_name(el) -> str:
    if isinstance(el, note.Note):
        return el.nameWithOctave
    if isinstance(el, chord.Chord):
        return "[" + ", ".join(p.nameWithOctave for p in el.pitches) + "]"
    if isinstance(el, Rest):
        return f"쉼표({el.quarterLength}박)"
    return type(el).__name__


def _note_dict(measure, voice_index: int | None = None) -> dict[float, list]:
    """
    박자 offset → [Note/Rest/Chord] (ChordSymbol 명시적 제외).

    주의: ChordSymbol은 chord.Chord의 서브클래스이므로
    getElementsByClass(chord.Chord)에 포함됩니다. isinstance 필터로 제외합니다.

    voice_index: None=전체, 0=소프라노, 1=알토 (2성부 악보용)
    """
    d: dict[float, list] = {}
    voices = list(measure.getElementsByClass("Voice"))

    if voices and voice_index is not None and voice_index < len(voices):
        elements = voices[voice_index].getElementsByClass([note.Note, chord.Chord, Rest])
    else:
        elements = measure.flatten().getElementsByClass([note.Note, chord.Chord, Rest])

    for el in elements:
        if isinstance(el, ChordSymbol):   # ChordSymbol ⊂ Chord — 명시적으로 제외
            continue
        k = round(float(el.offset), 6)
        d.setdefault(k, []).append(el)
    return d


def _chord_symbol_list(measure) -> list[tuple[float, str]]:
    """마디 내 (offset, chord_label) 코드 기호 목록"""
    result = []
    for el in measure.flatten().getElementsByClass(ChordSymbol):
        try:
            # figure 속성 우선 ("Am", "G7", "Cmaj7" 등 전체 표기)
            label = el.figure if (hasattr(el, "figure") and el.figure) else None
            if not label:
                root = el.root().name
                label = root + ("m" if el.quality == "minor" else "")
        except Exception:
            label = str(el)
        result.append((round(float(el.offset), 6), label))
    return result


def _chord_matches(orig_ch: str, pdf_texts: list[str]) -> bool:
    """OCR 텍스트 목록에서 원본 코드와 매칭되는 항목 확인 (접두사 허용)."""
    for pdf_ch in pdf_texts:
        if orig_ch == pdf_ch:
            return True
        # "Am" vs "Am7" 또는 "G" vs "G7" 등 접두사 허용
        if pdf_ch.startswith(orig_ch) or orig_ch.startswith(pdf_ch):
            return True
    return False


def _lyric_list_by_verse(measure) -> dict[int, list[tuple[float, str]]]:
    """마디 내 절(verse)별 (offset, 가사) 목록 — 절 번호("1.아" → "아") 제거"""
    result: dict[int, list] = {}
    for el in measure.flatten().getElementsByClass(note.Note):
        for ly in el.lyrics:
            if ly.text:
                text = _VERSE_NUM.sub('', ly.text.strip())
                if text:
                    verse = ly.number if ly.number is not None else 1
                    result.setdefault(verse, []).append(
                        (round(float(el.offset), 6), text)
                    )
    return result


def _korean_chars(text: str) -> set:
    return {c for c in text if '가' <= c <= '힣'}


def _lyric_similarity(orig_text: str, pdf_text: str) -> float:
    """원본 한글 음절 중 PDF에서 감지된 비율 (recall, 0.0~1.0).
    Jaccard 대신 recall을 사용하는 이유: EasyOCR이 인접 마디 글자를 같이 검출해도
    원본 글자가 포함돼 있으면 올바른 감지로 판단."""
    ok = _korean_chars(orig_text)
    pk = _korean_chars(pdf_text)
    if not ok:
        return 1.0
    if not pk:
        return 0.0
    return len(ok & pk) / len(ok)


_DUR_TOLERANCE = 1e-3  # 부동소수점 오차 허용 (0.001 박 이내는 같은 길이로 간주)


def _dur_eq(a: float, b: float) -> bool:
    return abs(a - b) <= _DUR_TOLERANCE


def _is_tied_start(el) -> bool:
    """음표가 타이의 시작 또는 연속 노트인지 확인 (타이 길이 합산 필요 신호)"""
    return getattr(el, 'tie', None) is not None and el.tie.type in ('start', 'continue')


def _compare_notes(el_pdf, el_orig, m: int, offset: float) -> list[Discrepancy]:
    errors = []

    if type(el_pdf) != type(el_orig):
        errors.append(Discrepancy("type", m, offset,
            f"형식 불일치 - PDF: {type(el_pdf).__name__}, 원본: {type(el_orig).__name__}"))
        return errors

    if isinstance(el_orig, note.Note):
        if el_pdf.nameWithOctave != el_orig.nameWithOctave:
            errors.append(Discrepancy("pitch", m, offset,
                f"음높이 틀림 - PDF: {el_pdf.nameWithOctave}, 원본: {el_orig.nameWithOctave}"))

        pdf_dur  = round(float(el_pdf.quarterLength), 6)
        orig_dur = round(float(el_orig.quarterLength), 6)
        if not _dur_eq(pdf_dur, orig_dur):
            pitch_match = el_pdf.nameWithOctave == el_orig.nameWithOctave
            if pitch_match or _is_tied_start(el_orig):
                # 음높이가 같거나 원본에 타이가 있으면 붙임줄/이음줄 오인식 의심
                errors.append(Discrepancy("tie_suspect", m, offset,
                    f"길이 다름 (붙임줄/이음줄 오인식 가능성) "
                    f"- PDF: {pdf_dur}박, 원본: {orig_dur}박"
                    + (" [원본 타이 존재]" if _is_tied_start(el_orig) else "")))
            else:
                errors.append(Discrepancy("duration", m, offset,
                    f"음길이 틀림 - PDF: {pdf_dur}박, 원본: {orig_dur}박"))

    elif isinstance(el_orig, chord.Chord):
        pdf_p  = sorted(p.nameWithOctave for p in el_pdf.pitches)
        orig_p = sorted(p.nameWithOctave for p in el_orig.pitches)
        if pdf_p != orig_p:
            errors.append(Discrepancy("pitch", m, offset,
                f"화음 불일치 - PDF: {pdf_p}, 원본: {orig_p}"))
        pdf_dur  = round(float(el_pdf.quarterLength), 6)
        orig_dur = round(float(el_orig.quarterLength), 6)
        if not _dur_eq(pdf_dur, orig_dur):
            errors.append(Discrepancy("duration", m, offset,
                f"화음 길이 틀림 - PDF: {pdf_dur}박, 원본: {orig_dur}박"))

    elif isinstance(el_orig, Rest):
        pdf_dur  = round(float(el_pdf.quarterLength), 6)
        orig_dur = round(float(el_orig.quarterLength), 6)
        if not _dur_eq(pdf_dur, orig_dur):
            errors.append(Discrepancy("duration", m, offset,
                f"쉼표 길이 틀림 - PDF: {pdf_dur}박, 원본: {orig_dur}박"))

    return errors


# ── 공개 API ──────────────────────────────────────────────────────────

def compare(
    pdf_xml_path:  str,
    orig_xml_path: str,
    part_index:    int = 0,
    voice_index:   int | None = None,
    pdf_chords:    list[tuple[int, str]] | None = None,
    pdf_lyrics:    list[tuple[int, str]] | None = None,
) -> CompareResult:
    """
    두 MusicXML 파일을 마디·박자 단위로 비교합니다.

    Args:
        pdf_xml_path:  PDF OMR 변환 XML 경로
        orig_xml_path: 피날레 원본 XML 경로
        part_index:    비교할 파트 인덱스
        voice_index:   비교할 성부 인덱스 (None=전체, 0=소프라노, 1=알토)
        pdf_chords:    PDF에서 OCR로 추출한 [(measure, chord_text), ...] (옵션)
        pdf_lyrics:    PDF에서 OCR로 추출한 [(measure, lyric_text), ...] (옵션)
    """
    print("\n[악보 로딩]")
    score_pdf  = _load(pdf_xml_path)
    score_orig = _load(orig_xml_path)

    def get_part(score, idx):
        parts = score.parts
        if not parts:
            return score.flatten()
        return parts[min(idx, len(parts)-1)]

    part_pdf  = get_part(score_pdf,  part_index)
    part_orig = get_part(score_orig, part_index)

    measures_pdf  = list(part_pdf.getElementsByClass("Measure"))
    measures_orig = list(part_orig.getElementsByClass("Measure"))
    total = max(len(measures_pdf), len(measures_orig))

    result = CompareResult(pdf_xml_path, orig_xml_path, total)
    print(f"\n[트랙1: 음표 비교] PDF({len(measures_pdf)}마디) vs 원본({len(measures_orig)}마디)")

    # 마디 번호 조회 헬퍼 (pickup measure = 0 포함, .number is not None 체크 필수)
    def _m_num(idx: int) -> int:
        orig_n = measures_orig[idx].number if idx < len(measures_orig) else None
        pdf_n  = measures_pdf[idx].number  if idx < len(measures_pdf)  else None
        n = orig_n if orig_n is not None else pdf_n
        return n if n is not None else idx + 1

    # ── 트랙 1: 음표 비교 ──────────────────────────────────────────
    for idx in range(total):
        m_num = _m_num(idx)

        if idx >= len(measures_pdf):
            result.discrepancies.append(Discrepancy("measure_miss", m_num, 0.0,
                "PDF 추출본에 마디 전체 누락", "note"))
            continue
        if idx >= len(measures_orig):
            result.discrepancies.append(Discrepancy("measure_miss", m_num, 0.0,
                "원본에 없는 잉여 마디가 PDF에 존재", "note"))
            continue

        data_pdf  = _note_dict(measures_pdf[idx],  voice_index)
        data_orig = _note_dict(measures_orig[idx], voice_index)
        all_offsets = sorted(set(data_pdf) | set(data_orig))

        for offset in all_offsets:
            in_pdf  = offset in data_pdf
            in_orig = offset in data_orig

            if in_pdf and in_orig:
                el_pdf  = data_pdf[offset][0]
                el_orig = data_orig[offset][0]
                for d in _compare_notes(el_pdf, el_orig, m_num, offset):
                    d.track = "note"
                    result.discrepancies.append(d)

            elif in_orig and not in_pdf:
                name = _element_name(data_orig[offset][0])
                result.discrepancies.append(Discrepancy("missing", m_num, offset,
                    f"PDF 누락 - 원본의 '{name}' 인식 실패 (OMR 오류 가능)", "note"))

            elif in_pdf and not in_orig:
                name = _element_name(data_pdf[offset][0])
                result.discrepancies.append(Discrepancy("noise", m_num, offset,
                    f"PDF 노이즈 - 원본에 없는 '{name}' 오인식", "note"))

    # ── 트랙 2: 코드 기호 비교 ────────────────────────────────────
    print("[트랙2: 코드 기호 비교]")
    for idx in range(min(len(measures_orig), total)):
        m_num = _m_num(idx)
        orig_chords = _chord_symbol_list(measures_orig[idx])
        if not orig_chords:
            continue

        # PDF에서 추출한 코드 (staff 번호 기준이므로 마디와 1:1 매칭은 근사)
        if pdf_chords:
            pdf_chord_texts = [c for m, c in pdf_chords if m == m_num]
            for orig_offset, orig_ch in orig_chords:
                if not _chord_matches(orig_ch, pdf_chord_texts):
                    result.discrepancies.append(Discrepancy("chord_miss", m_num, orig_offset,
                        f"PDF에서 코드 기호 '{orig_ch}' 누락", "chord"))
        else:
            # PDF 코드 없으면 원본 코드만 목록 기록 (누락 경고)
            for orig_offset, orig_ch in orig_chords:
                result.discrepancies.append(Discrepancy("chord_miss", m_num, orig_offset,
                    f"코드 기호 '{orig_ch}' - PDF 미추출 (OCR 필요)", "chord"))

    # ── 트랙 3: 가사 비교 ─────────────────────────────────────────
    print("[트랙3: 가사 비교]")
    for idx in range(min(len(measures_orig), total)):
        m_num = _m_num(idx)
        orig_by_verse = _lyric_list_by_verse(measures_orig[idx])
        if not orig_by_verse:
            continue

        if pdf_lyrics:
            pdf_lyric_text = " ".join(t for m, t in pdf_lyrics if m == m_num)
            pdf_korean = _korean_chars(pdf_lyric_text)

            # 각 절(verse)별로 유사도를 계산하여 최고 매칭 절 선택
            best_sim: float = -1.0
            best_orig_text: str | None = None
            for verse_lyrics in orig_by_verse.values():
                verse_text   = " ".join(t for _, t in verse_lyrics)
                verse_korean = _korean_chars(verse_text)
                if len(verse_korean) < 2:
                    continue
                sim = _lyric_similarity(verse_text, pdf_lyric_text) if pdf_korean else 0.0
                if sim > best_sim:
                    best_sim       = sim
                    best_orig_text = verse_text

            if best_orig_text is None:
                continue

            if not pdf_korean:
                result.discrepancies.append(Discrepancy("lyric_miss", m_num, 0.0,
                    f"가사 누락 - 원본: '{best_orig_text[:30]}'", "lyric"))
            elif best_sim < _LYRIC_SIM_THRESHOLD:
                result.discrepancies.append(Discrepancy("lyric_diff", m_num, 0.0,
                    f"가사 불일치(유사도 {best_sim:.0%}) - 원본: '{best_orig_text[:20]}', "
                    f"PDF: '{pdf_lyric_text[:20]}'", "lyric"))

    return result
