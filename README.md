# this is a work in progress... A toy test, developed over weekend.
I am not sure this will work at any point

# performer-aware-transformer
Performer-Aware-Transformer for Music This project explores a new way to represent and model symbolic music. Instead of treating notes as abstract MIDI events, we encode performer gestures.
This project explores a new way to represent and model symbolic music.
Instead of treating notes as abstract MIDI events, we encode performer gestures:
- Pitch — the actual sounding note
- Left hand (LH) shape — full chord grip or hand position (e.g. 320003)
- Right hand (RH) fingering — p i m a masks, strums, or pluck patterns
- Duration & tempo — musical timing
- Velocity — dynamics / articulation

The performer-Aware Transformer (PAT) trains on these tokens, aiming that everything it generates is:
Musically correct (valid pitches and durations)
Physically plausible (playable on a real instrument)
Compact (hundreds of tokens instead of tens of thousands)

Composer always take performer into account while composing! let's force AI to do the same with the token set!


# PAT Tokenizer Design Document
**Performance-Aware Tokenization for Guitar Music Generation**

## Overview

The PAT (Performance-Aware Token) system is a novel tokenization approach for guitar music that encodes **physical performance constraints** rather than abstract musical notation. This enables transformer models to generate music that is both musically coherent and physically playable on guitar.

## Core Philosophy

### Physical Constraints as Musical Foundation
- **Traditional approach**: Music theory → fingering (post-hoc playability check)
- **PAT approach**: Physical constraints → music (inherent playability)
- **Key insight**: Real composers think through their instrument - musical ideas emerge from physical exploration

### Embodied Musical Intelligence
- Encode the **biomechanics of guitar playing** as the primary vocabulary
- Let musical relationships emerge from physical relationships
- Mirror how human guitarists learn: through muscle memory and pattern recognition

## Token Architecture

### Hierarchical Token Types
PAT uses separate token types with consistent semantics, allowing transformers to learn specialized attention patterns for different aspects of performance:

```
[META] → [BAR] → [POS] → [LH] → [RH] → [DUR] → [POS] → [RH] → [DUR] → [BAR] → ...
```

### Token Specifications

#### MetaToken
- **Purpose**: Establish global musical context
- **Content**: BPM, time signature
- **Example**: `META_120_4/4`
- **Frequency**: Once per piece/section

#### BarToken  
- **Purpose**: Structural boundaries for phrase learning
- **Content**: Bar index (resets per piece)
- **Example**: `BAR_0`, `BAR_1`, `BAR_2`
- **Frequency**: Start of each measure

#### PositionToken
- **Purpose**: Rhythmic timing within measures
- **Content**: Discrete slot (0-15 in 4/4 time)
- **Example**: `POS_0` (downbeat), `POS_4` (beat 2), `POS_8` (beat 3)
- **Resolution**: 16 slots per bar = up to 64th note precision

#### LHToken (Left Hand)
- **Purpose**: Chord shapes and fret positions
- **Content**: 6-character string notation (low→high: strings 6,5,4,3,2,1)
- **Examples**: 
  - `LH_320003` (G major chord)
  - `LH_xx0232` (D major, muted low strings)
  - `LH_xxxxxx` (all strings muted/silent)
- **Optimization**: Only emitted when position changes from previous

#### RHToken (Right Hand)
- **Purpose**: Picking/strumming patterns and attack definition  
- **Content**: 
  - `fingers`: Classical notation (p=thumb, i=index, m=middle, a=ring)
  - `strings`: String numbers being played (high→low pitch order)
  - `pitches`: Comma-separated MIDI pitches for learning pitch relationships
- **Examples**:
  - `RH_pim_321_60,64,67` (fingerpicked arpeggio)
  - `RH_pppppp_654321_40,45,50,55,59,64` (full strum)

#### DurationToken
- **Purpose**: Note sustain length per string
- **Content**: Duration symbols per played string (same order as RH strings)
- **Symbols**: `w`=whole, `h`=half, `q`=quarter, `e`=eighth, `s`=sixteenth, `d`=dotted
- **Examples**:
  - `DUR_qqh` (3 notes: quarter, quarter, half)
  - `DUR_w` (single whole note)

## Key Design Decisions

### 1. Attack Time vs. Note Duration Separation
- **Attack Time**: When right hand strikes (encoded in POS tokens)  
- **Note Duration**: How long left hand sustains (encoded in DUR tokens)
- **Rationale**: These are independent physical actions in guitar playing
- **Enables**: Syncopated attacks, overlapping sustains, realistic guitar articulation

### 2. Event-Based vs. Time-Grid Tokenization
- **Choice**: Event-based tokenization
- **Rationale**: Matches guitarist mental model (think in "attacks" not clock ticks)
- **Benefits**: Natural rubato, efficient encoding, variable timing support

### 3. Bass/Melody Voice Separation
- **Rule**: Split fingerpicked passages into separate RH/DUR token pairs
- **Exception**: Keep strummed chords as single RH/DUR pair
- **Detection**: Different finger types (p vs. ima) + different durations = separate voices
- **Rationale**: Reflects musical structure - fingerpicking is polyphonic, strumming is harmonic

### 4. String Assignment Algorithm  
- **Method**: Greedy assignment with cost function
- **Cost factors**:
  - Finger movement from previous position
  - Hand span (fret range)  
  - Open string preference (slight bonus)
- **Constraint**: Prefer string uniqueness (one note per string)
- **Fallback**: Allow string reuse for complex passages

### 5. Pitch Information Inclusion
- **Decision**: Include MIDI pitches in RH tokens
- **Purpose**: Let model learn physical→pitch mapping
- **Benefit**: Handles alternate fingerings for same pitches
- **Example**: Same note playable on multiple string/fret combinations

## Vocabulary Characteristics

### Estimated Size
- **META tokens**: ~50 (common BPM/time signature combinations)
- **BAR tokens**: Unlimited (but typically small numbers per piece)
- **POS tokens**: 16 (fixed slots per bar)
- **LH tokens**: ~2,000-5,000 (physically possible chord shapes)
- **RH tokens**: ~500-1,000 (finger/string/pitch combinations)  
- **DUR tokens**: ~100-200 (common duration patterns)

**Total vocabulary**: ~3,000-7,000 tokens (much smaller than typical music tokenizers)

### Advantages of Compact Vocabulary
- **Computational efficiency**: Smaller embedding matrices, faster training
- **Sample efficiency**: Better learning with limited data
- **Physical boundedness**: No impossible combinations in vocabulary
- **Musical focus**: All model capacity dedicated to playable patterns

## Training Data Considerations

### Preferred Sources
1. **Guitar tablature**: Already encodes performance information
2. **Clean guitar MIDI**: From guitar-specific libraries  
3. **Classical guitar repertoire**: Well-documented fingering patterns
4. **Converted piano MIDI**: As fallback with adapted fingering algorithms

### Data Preprocessing
- **Tab → PAT conversion**: Direct mapping from fret positions to LH tokens
- **MIDI → PAT conversion**: Inferred fingering using cost-based assignment
- **Quality filtering**: Remove passages exceeding physical constraints (hand span, impossible stretches)

## Expected Model Behaviors

### Musical Intelligence Through Physical Intelligence
- **Chord progressions**: Emerge from natural finger movements between shapes  
- **Voice leading**: Follows physically comfortable fingering transitions
- **Rhythmic patterns**: Reflect natural right-hand coordination capabilities
- **Stylistic coherence**: Physical constraints naturally enforce genre-appropriate techniques

### Generation Capabilities
- **Unconditional**: Pure pattern-based composition following physical logic
- **Conditional**: Style-aware generation (classical, flamenco, fingerstyle)
- **Interactive**: Real-time continuation and variation of human input
- **Educational**: Technique-focused exercise generation

## Implementation Notes

### Token Sequence Example
```
[META_120_4/4]
[BAR_0] [POS_0] [LH_320003] [RH_pim_321_60,64,67] [DUR_qqq]
[POS_4] [RH_p_6_40] [DUR_w]  
[BAR_1] [POS_0] [LH_x32010] [RH_ima_321_62,65,69] [DUR_eee]
[POS_2] [RH_p_5_45] [DUR_h]
```

### Future Extensions
- **Technique tokens**: Vibrato, bends, slides, harmonics
- **Dynamic tokens**: Volume, accent, articulation markings
- **Multi-instrument**: Adaptation to other fretted instruments
- **Style conditioning**: Explicit genre/composer tokens

## Conclusion

The PAT tokenizer represents a fundamental shift from notation-based to performance-based music AI. By grounding musical generation in physical constraints, it promises to produce guitar music that is not only musically coherent but authentically playable, opening new possibilities for AI-assisted composition, education, and performance.