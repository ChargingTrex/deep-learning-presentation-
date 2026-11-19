# Tamil uyir GAN results (standard, random_60_40, resnet)

Train letters: 3714 | test letters: 2476 | GAN iterations: 10000 | seeds: 1

| Recogniser trained on | Test accuracy % | Macro-F1 % |
|---|---|---|
| real | 98.26 | 98.21 |
| real+forged | 98.22 | 98.17 |

| Forger / detector metric | Value |
|---|---|
| Forgery accuracy (real-only recogniser reads the intended vowel) | 91.6% |
| Fréchet distance, forged vs real test (recogniser features) | 22.5 |
| Fréchet distance, real train vs real test (floor) | 1.2 |
| Diversity of forged letters vs real (1.0 = as varied as real) | 1.61 |
| Detector ROC-AUC, real test vs forged (0.5 = fooled) | 1.000 |