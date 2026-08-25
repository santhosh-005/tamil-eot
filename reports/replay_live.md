# Live-path replay — `smart-turn-tamil-tiny` @ 0.5

500 labelled boundaries from `test`, replayed through the real Silero VAD (`min_silence_duration=0.25`) and the real `SmartTurnDetectorStream` in 20 ms frames. No STT, LLM or TTS — none of them feed the prediction.

## Coverage — what the live path never asks about

| | n | share |
|---|---|---|
| boundaries | 500 | |
| **VAD closed, model consulted** | **461** | 92.2% |
| VAD never closed — never asked | 39 | 7.8% |

- `complete`: asked on 306 of 325 (94%)
- `incomplete`: asked on 155 of 175 (89%)

A boundary the VAD never closes on is one the model is **never consulted about** — LiveKit will not request a prediction below `min_silence_duration + 50 ms`. Those cases are not errors; they are outside the product. The offline test set counts them, which is why its accuracy and this one are not the same quantity.

## Accuracy — the same boundaries, two windows

Identical rows (461), model and threshold. The only difference is where the window ends: the pre-cut clip stops 0.20 s after the labelled offset, the live one stops wherever Silero closed.

| | pre-cut clip | **live window** |
|---|---|---|
| accuracy | 86.12% | **83.51%** |
| true complete | 266 | 260 |
| true incomplete | 131 | 125 |
| **false complete** (talks over the user) | 24 | **30** |
| false incomplete (waits too long) | 40 | 46 |
| FP/N | 5.21% | **6.51%** |

**Delta: -2.60 points.** This is the cost of serving, isolated — same audio, same labels, same model, later window.

Neither column is the headline `RESULTS.md` number: that one is over the whole split, and this is over the subset the VAD surfaced.

## The window the model was handed

| | p10 | p50 | p90 |
|---|---|---|---|
| VAD close, relative to the labelled offset | +0.26 s | **+0.29 s** | +0.32 s |
| audio in the buffer | 8.00 s | **8.00 s** | 8.00 s |

Offline clips end **+0.20 s** past the labelled offset, by construction (`rules.py: TRAIL_S`). The gap between that and the p50 above is the train/serve skew this whole script exists to measure.

0 of 461 predictions (0%) ran on less than a full 8 s window, zero-padded by `features.py`.