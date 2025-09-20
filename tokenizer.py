from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import miditoolkit

# ============================================================
# Token dataclasses
# ============================================================
@dataclass
class MetaToken:
    bpm: int
    time_sig: str
    token_type: str = "META"

@dataclass
class BarToken:
    bar_index: int
    token_type: str = "BAR"

@dataclass
class PositionToken:
    slot: int               # discrete slot in bar
    token_type: str = "POS"

@dataclass
class LHToken:
    position: str           # e.g. "320003"
    token_type: str = "LH"

@dataclass
class RHToken:
    fingers: str            # e.g. "pim"
    strings: str            # e.g. "543"
    pitches: str            # comma-separated MIDI pitches "48,52,55"
    token_type: str = "RH"

@dataclass
class DurationToken:
    durations: str          # e.g. "qqh" (per note in same order as RH)
    token_type: str = "DUR"

# ============================================================
# Guitar tuning and helpers
# ============================================================
OPEN = {6: 40, 5: 45, 4: 50, 3: 55, 2: 59, 1: 64}  # MIDI pitches E2..E4

DUR_TABLE = [
    ("w", 4.0), ("dh", 3.0), ("h", 2.0),
    ("dq", 1.5), ("q", 1.0), ("de", 0.75),
    ("e", 0.5), ("s", 0.25)
]

def quantize_duration_ticks(ticks: int, ppq: int) -> str:
    q = max(0.0, ticks / ppq)
    sym, _ = min(DUR_TABLE, key=lambda kv: abs(q - kv[1]))
    return sym

def pitch_to_string_frets(pitch: int, max_fret: int = 20) -> List[Tuple[int,int]]:
    opts = []
    for s, open_pitch in OPEN.items():
        f = pitch - open_pitch
        if 0 <= f <= max_fret:
            opts.append((s, f))
    return opts

def empty_lh_shape() -> Dict[int, Optional[int]]:
    return {1: None, 2: None, 3: None, 4: None, 5: None, 6: None}

def lh_dict_to_shape(shape: Dict[int, Optional[int]]) -> str:
    out = []
    for s in (6,5,4,3,2,1):
        f = shape[s]
        out.append('x' if f is None else str(f))
    return ''.join(out)

def default_rh_for_string(s: int) -> str:
    if s >= 4: return 'p'
    return {3:'i',2:'m',1:'a'}[s]

# ============================================================
# Time helpers
# ============================================================
def tempo_map(midi: miditoolkit.MidiFile) -> List[Tuple[int, float]]:
    arr = sorted([(t.time, 60_000_000.0 / t.tempo) for t in midi.tempo_changes], key=lambda x: x[0])
    return arr or [(0, 120.0)]

def bpm_at(tmap: List[Tuple[int,float]], tick: int) -> int:
    cur = tmap[0][1]
    for tt, bpm in tmap:
        if tick >= tt: cur = bpm
        else: break
    return int(round(cur))

def timesig_at(midi: miditoolkit.MidiFile, tick: int) -> Tuple[int,int]:
    arr = sorted([(ts.time, (ts.numerator, ts.denominator)) for ts in midi.time_signature_changes], key=lambda x: x[0])
    cur = (4,4)
    for tt, sig in arr:
        if tick >= tt: cur = sig
        else: break
    return cur

def bar_and_slot(onset_tick: int, midi: miditoolkit.MidiFile, slots_per_bar: int) -> Tuple[int,int]:
    ppq = midi.ticks_per_beat
    num, den = timesig_at(midi, onset_tick)
    ticks_per_beat = ppq * (4/den)
    bar_ticks = int(round(num * ticks_per_beat))
    bar_idx = onset_tick // bar_ticks
    inbar = onset_tick - bar_idx * bar_ticks
    slot = int(round(inbar / bar_ticks * (slots_per_bar-1)))
    return bar_idx, slot

# ============================================================
# Core: MIDI → tokens
# ============================================================
def midi_to_pat_tokens(
    midi_path: str,
    track_index: int = 0,
    slots_per_bar: int = 16,
    max_fret: int = 20
) -> List[object]:
    midi = miditoolkit.MidiFile(midi_path)
    insts = [i for i in midi.instruments if not i.is_drum]
    if not insts:
        raise ValueError("No non-drum tracks found.")
    inst = insts[min(track_index, len(insts)-1)]
    ppq = midi.ticks_per_beat

    notes = sorted(inst.notes, key=lambda n: (n.start, -n.pitch))

    # group by onset tick
    groups: List[List[miditoolkit.Note]] = []
    buf = []; cur = None
    for n in notes:
        if cur is None or n.start == cur:
            buf.append(n); cur = n.start
        else:
            groups.append(buf); buf=[n]; cur=n.start
    if buf: groups.append(buf)

    # META token
    tmap = tempo_map(midi)
    ts_num, ts_den = timesig_at(midi, groups[0][0].start if groups else 0)
    tokens: List[object] = [
        MetaToken(bpm=bpm_at(tmap, groups[0][0].start if groups else 0),
                  time_sig=f"{ts_num}/{ts_den}")
    ]

    prev_lh = empty_lh_shape()
    prev_bar = -1

    for g in groups:
        onset = g[0].start
        bar, slot = bar_and_slot(onset, midi, slots_per_bar)

        # BAR token if bar changed
        if bar != prev_bar:
            tokens.append(BarToken(bar_index=bar))
            prev_bar = bar

        # POS token always
        tokens.append(PositionToken(slot=slot))

        # Build LH shape (greedy, naive for now)
        shape = empty_lh_shape()
        used_strings = set()
        for n in sorted(g, key=lambda x: x.pitch):
            opts = pitch_to_string_frets(n.pitch, max_fret=max_fret)
            if not opts: continue
            s, f = opts[0]
            if s in used_strings: continue
            shape[s] = f
            used_strings.add(s)

        if lh_dict_to_shape(shape) != lh_dict_to_shape(prev_lh):
            tokens.append(LHToken(position=lh_dict_to_shape(shape)))
            prev_lh = {k: shape[k] for k in shape}

        # Build per-note RH data
        played = []
        for n in sorted(g, key=lambda x: -x.pitch):  # high→low
            opts = pitch_to_string_frets(n.pitch, max_fret=max_fret)
            if not opts: continue
            s, f = opts[0]
            played.append((n, s, f))

        fingers_list, strings_list, pitches_list, durs_list = [], [], [], []
        for n, s, f in played:
            rh = default_rh_for_string(s)
            fingers_list.append(rh)
            strings_list.append(str(s))
            pitches_list.append(str(n.pitch))
            durs_list.append(quantize_duration_ticks(n.end - n.start, ppq))

        # Group: thumb vs treble
        def emit_group(idx_list):
            if not idx_list: return []
            fingers = ''.join(fingers_list[i] for i in idx_list)
            strings = ''.join(strings_list[i] for i in idx_list)
            pitches = ','.join(pitches_list[i] for i in idx_list)
            durs    = ''.join(durs_list[i]    for i in idx_list)
            return [RHToken(fingers=fingers, strings=strings, pitches=pitches),
                    DurationToken(durations=durs)]

        idx_thumb  = [i for i, r in enumerate(fingers_list) if r == 'p']
        idx_treble = [i for i, r in enumerate(fingers_list) if r in ('i','m','a')]

        tokens += emit_group(idx_thumb)   # bass first
        tokens += emit_group(idx_treble)  # treble next

    return tokens

# ============================================================
# CLI for quick testing
# ============================================================
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python tokenizer.py path/to/file.mid")
        sys.exit(1)
    toks = midi_to_pat_tokens(sys.argv[1])
    for t in toks[:40]:
        print(t)
    print(f"... total tokens: {len(toks)}")
