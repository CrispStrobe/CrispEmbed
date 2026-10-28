# Unlimited-OCR-2 precision sweep

commit `d6fd934a16018489c54a727ce7121e18cde55515`, checkpoint revision `04647c7a6098d8d58915f003536a741c82eafb40`, reference revision `04647c7a6098d8d58915f003536a741c82eafb40`. CPU, greedy, 384-token cap.

| precision | size GB | case | ids exact | first diff | text= (no boxes) | max box d | CER | wall s | peak RSS MB |
|---|---|---|---|---|---|---|---|---|---|
| f32 | 13.38 | small-untiled | True | None | False | 0.0 | 0.267857 | 440.15 | 21055 |
| f32 | 13.38 | para-en | False | 7 | False | 3.0 | 0.216393 | 2701.87 | 21051 |
| f32 | 13.38 | table | False | 4 | False | 3.0 | 0.3 | 2678.74 | 21086 |
| f16 | 6.69 | small-untiled | True | None | False | 0.0 | 0.267857 | 498.4 | 12821 |
| f16 | 6.69 | para-en | False | 7 | False | 3.0 | 0.216393 | 2849.54 | 12821 |
| f16 | 6.69 | table | False | 4 | False | 3.0 | 0.3 | 2762.05 | 11853 |
| q8_0 | 3.57 | small-untiled | True | None | False | 0.0 | 0.267857 | 350.33 | 7084 |
| q8_0 | 3.57 | para-en | False | 7 | False | 1.0 | 0.209836 | 2571.35 | 7625 |
| q8_0 | 3.57 | table | False | 4 | False | 3.0 | 0.3 | 2520.31 | 7625 |
| q4_k | 2.27 | small-untiled | False | 7 | False | 5.0 | 0.327381 | 334.65 | 5835 |
| q4_k | 2.27 | para-en | False | 4 | False | 5.0 | 0.232787 | 2573.93 | 6380 |
| q4_k | 2.27 | table | False | 4 | False | 6.0 | 0.327273 | 2509.46 | 6380 |
| mix-q8attn-q4exp | 2.36 | small-untiled | False | 10 | False | 5.0 | 0.357143 | 336.65 | 5929 |
| mix-q8attn-q4exp | 2.36 | para-en | False | 13 | False | 3.0 | 0.213115 | 2582.09 | 6470 |
| mix-q8attn-q4exp | 2.36 | table | False | 7 | False | 3.0 | 0.3 | 2509.98 | 6470 |
| mix-f16attn-q4exp | 2.75 | small-untiled | False | 10 | False | 5.0 | 0.357143 | 337.51 | 6303 |
| mix-f16attn-q4exp | 2.75 | para-en | False | 13 | False | 3.0 | 0.213115 | 2560.75 | 6844 |
| mix-f16attn-q4exp | 2.75 | table | False | 7 | False | 3.0 | 0.3 | 2527.78 | 6844 |
