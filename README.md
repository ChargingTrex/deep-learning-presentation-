# Deep learning presentation: GANs and YuNet

Sathya's Sai University deep learning presentation (10 minutes, 23 slides).

- **Group A, GAN:** a conditional GAN that forges the 12 handwritten Tamil uyir letters (அ to ஔ). The Forger draws a letter on request, and the Detector tries to tell it apart from real handwriting.
- **Group B, YuNet:** a face detector with 75,856 parameters that finds faces in about 1.6 ms on a CPU.

## What's here

| Folder | Contents |
|---|---|
| `slides/` | `GAN_YuNet_deck.html` is the animated deck, and `GAN_YuNet_backup_slides.pdf` is a static copy of every slide |
| `script/` | `speech_10min.md` is the 10-minute speech, one line per click. `viva_questions.md` has likely questions with short answers |
| `code/` | `Tamil_Uyir_GAN_Forger_vs_Detector.ipynb` is the full GAN project notebook (Colab) |
| `code/results/` | `results.json` and `results.md` hold the final numbers. `figures/` holds the notebook's output figures, and `forged_letters.zip` has sample forgeries for each letter |
| `code/warmup/` | `GAN_MNIST.ipynb` is the small MNIST GAN behind the digit images on slides 2 and 3 |

## Presenting the deck

Download `slides/GAN_YuNet_deck.html` and open it in Chrome or Edge. Everything is embedded, fonts included, so it works offline.

| Key | Action |
|---|---|
| Right arrow, Space, Page Down (most clickers) or click | Next animation step |
| Left arrow, Page Up or right-click | Back one step |
| N | Speaker notes, with the time each slide should be finished by |
| P | Presenter window with notes and timer |
| F | Fullscreen |

The PDF is a backup in case the browser fails.

## The GAN at a glance

| Part | Details |
|---|---|
| Data | HP Labs India isolated handwritten Tamil characters (12 uyir letters, 6,190 images), split 60/40 into 3,714 training and 2,476 test letters |
| Preprocessing | Invert, Otsu threshold, crop, thin to one pixel, redraw with a 5-pixel pen, centre on 128×128, scale to −1..+1 |
| Forger (4.20 M parameters) | 120-number noise split into 6 slices (skip-z), a 128-number letter embedding, 5 residual up-blocks with conditional batch norm and ReLU, then Tanh. Output: 128×128 |
| Detector (9.61 M parameters) | 6 residual down-blocks with spectral norm and ReLU, sum pooling, and a projection head. No sigmoid |
| Training | Hinge loss, DiffAugment, a mode-seeking term, TTUR (Detector 4e-4, Forger 1e-4) and an EMA Forger. 10,000 steps of 64, about 50 minutes on a Colab T4 |

| Result | Value |
|---|---|
| Forgery accuracy (forgeries a recogniser trained on real letters reads correctly) | 91.6% |
| Fréchet distance, forged vs real (real-vs-real floor: 1.2) | 22.5 |
| Diversity vs real handwriting (1.0 = as varied as real) | 1.61 |
| Detector ROC-AUC (0.5 = fully fooled) | 1.000 |
| Recogniser accuracy, real letters only | 98.26% |
| Recogniser accuracy, real + 1,800 forged letters | 98.22% |

The forgeries are readable, but the Detector can still tell every one apart from real handwriting. Adding forged letters to the training data did not help, because real data alone already reaches 98%.

## Running the notebook

Open `code/Tamil_Uyir_GAN_Forger_vs_Detector.ipynb` in Google Colab, set the runtime to a GPU and choose Run all. The notebook downloads the dataset itself. A full run takes about an hour, so don't train live during the talk; show the saved outputs instead. The trained weights (`forger_G_ema.pt` and `detector_D.pt`) are not in this repository, because the notebook writes them when it runs.

## References

The full reference list is on the last slide of the deck. The main sources are:

- Goodfellow et al., *Generative Adversarial Nets*, 2014
- Mirza and Osindero, *Conditional Generative Adversarial Nets*, 2014
- Miyato et al., *Spectral Normalization for GANs*, 2018, and Miyato and Koyama, *cGANs with Projection Discriminator*, 2018
- Zhao et al., *Differentiable Augmentation for Data-Efficient GAN Training*, 2020
- Wu, Peng and Yu, *YuNet: A Tiny Millisecond-level Face Detector*, Machine Intelligence Research, 2023
- HP Labs India isolated handwritten Tamil character dataset
