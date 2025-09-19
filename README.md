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
