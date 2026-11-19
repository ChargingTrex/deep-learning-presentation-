## What I changed from a standard DCGAN, and why

A standard DCGAN (Radford et al., 2015) is unconditional, works at 64×64, and uses transposed convolutions, batch norm, LeakyReLU(0.2) and a sigmoid with binary cross-entropy, trained with Adam at 2e-4 for both networks. My first version was close to that, and it gave poor Tamil letters. Each change below answers a problem from that run.

| Standard DCGAN | My GAN | Why |
|---|---|---|
| Unconditional: you can't choose what it draws | Conditional: the letter's 128-number embedding sets every Forger layer (conditional batch norm) and enters the Detector's projection head | To forge the letter I ask for. Gluing the label onto the noise, the usual way, tends to lose variety |
| One noise vector (100 numbers) at the input | 120 numbers in 6 slices of 20, one slice per layer (skip-z, from BigGAN) | With noise only at the start, the Forger can ignore it and draw one fixed letter per class |
| Transposed convolutions to upsample | Nearest-neighbour upsample, then a 3×3 convolution | Transposed convolutions leave checkerboard marks |
| Plain conv stacks, 64×64 output | Residual blocks with shortcuts in both networks, 128×128 output | Smoother strokes at 128×128; shortcuts carry gradients through the deeper network |
| Batch norm in the discriminator | Spectral normalisation on every Detector layer | The first run's Detector overpowered the Forger (D loss ≈ 0.2, G loss jumping from 2 to 10). Spectral norm caps how much each layer can amplify |
| LeakyReLU(0.2) in the discriminator | ReLU (LeakyReLU 0.1 only in the plain-DCGAN option) | Spectral norm and the shortcuts already keep gradients flowing |
| Sigmoid output with binary cross-entropy | Raw score with hinge loss | A sigmoid flattens out when the Detector is confident, so the Forger gets almost no gradient. Hinge loss stops pushing once real is above +1 and forged below −1 |
| One realness output | Projection head: fc(h) + ⟨embed(y), h⟩ | Scores "is this handwriting?" and "is it this letter?" together |
| No augmentation | DiffAugment: the same random shift and cutout on real and forged letters | Only 3,714 training letters, so the Detector could memorise them |
| Nothing against mode collapse | Mode-seeking term (MSGAN) | Penalises two different noise vectors that give the same letter |
| Adam, learning rate 2e-4 for both, β = (0.5, 0.999) | Detector 4e-4 and Forger 1e-4 (TTUR), β = (0.0, 0.99) as in BigGAN | Keeps the Detector a step ahead, as a useful teacher |
| Samples from the latest weights | EMA copy of the Forger (decay 0.999), batch-norm statistics re-estimated | Smoother, steadier final letters |
| About 1,400 updates in my first run | 10,000 updates, with checkpoints | 1,400 was far too short for 128×128 |

Kept from DCGAN: ReLU and batch norm in the generator, Tanh at the output, and pixels scaled to −1..+1.

## Speech V3: 10 minutes

One numbered line = one click; "(just click)" means click and keep going. 110 clicks, about 1,114 words. The time after each slide title is when that slide should be finished. The GAN slides focus on what I changed from a standard DCGAN and why; the classical GAN gets a short intro, and YuNet moves fast.

### 1. Title (by 0:15)

1. Good morning, I'm Sathya. Part one: my GAN that forges handwritten Tamil letters, and how I changed a standard DCGAN to get there,
2. and part two: YuNet, a tiny face detector.

### 2. Can a computer imagine? (by 0:30)

1. Most models judge: picture in, label out.
2. A GAN creates: noise in, a new picture out, like this digit from my warm-up GAN.
3. (just click)

### 3. The Artist and the Detective (by 0:55)

1. A GAN is a game: a Generator draws fakes, and a Discriminator tries to catch them.
2. (just click)
3. Every rejection teaches the Generator.
4. DCGAN, from 2015, is the standard recipe. My first version was close to a standard DCGAN, and its Tamil letters were poor. So on each slide: what DCGAN does, what I changed, and why.

### 4. The whole GAN in one picture (by 1:55)

1. Two networks: my Forger, the generator, and my Detector, the discriminator.
2. First change: a standard DCGAN is unconditional, you can't choose what it draws. My Forger gets 120 random numbers plus which letter to write, like அ. The Detector gets the letter too.
3. The Forger draws the letter at 128 by 128, twice DCGAN's usual 64.
4. The Detector also sees real letters.
5. DCGAN's discriminator ends in a sigmoid, a probability. Mine gives a raw score: plus 1.1 for this real அ, minus 2.65 for the forgery.
6. Those scores go into a hinge loss, not DCGAN's cross-entropy. In my first run the Detector overpowered the Forger; hinge loss stops pushing once it's confident.

### 5. Inside the Forger (by 3:05)

1. Inside the Forger. DCGAN feeds the noise in once, at the start. I split 120 numbers into 6 slices, one per layer: skip-z, from BigGAN. Otherwise the Forger can ignore the noise and draw the same letter every time.
2. A linear layer makes the seed: 512 channels, 4 by 4. These are real maps from my trained model.
3. DCGAN stacks plain convolutions; I use residual blocks with shortcuts, for smoother strokes at 128. Each doubles the size: 8, then 16.
4. Then 32, 64 and 128. Each block gets its own noise slice and the letter.
5. The last layer is the same as DCGAN: a convolution and Tanh. You can see அ appear.
6. One block, two more changes. Conditional batch norm: the letter sets every layer's scale and shift, instead of just being glued to the noise. And upsample-then-convolve replaces DCGAN's transposed convolution, which leaves checkerboard marks.

### 6. Inside the Detector (by 3:55)

1. The Detector gets the image and the letter.
2. DCGAN's discriminator uses batch norm; mine uses spectral normalisation on every layer. It limits how much a layer can amplify, so the Detector can't overpower the Forger, and its gradients stay useful.
3. Residual blocks halve the size, down to 512 channels at 4 by 4.
4. Sum over the grid: one vector of 512 numbers.
5. DCGAN has one output. My projection head scores two things: does it look like handwriting, and like this letter? Plus 1.1: real.
6. For the forged அ, both drop: minus 2.65.

### 7. ReLU, LeakyReLU and Tanh: why and where (by 4:50)

1. Activation functions: where I used each one, and how that differs from DCGAN.
2. ReLU keeps positives and zeroes negatives. DCGAN uses it only in the generator; I use it in both networks, and so does YuNet.
3. LeakyReLU lets a little of the negatives through. DCGAN's discriminator needs it so gradients never die. Mine doesn't: spectral norm and the shortcuts keep them flowing. My plain DCGAN option still uses LeakyReLU, 0.1.
4. Tanh, only at the Forger's output, same as DCGAN: minus one for paper, plus one for ink, matching my real letters.
5. And DCGAN's final sigmoid is gone: hinge loss needs the raw score, and a sigmoid flattens out when the Detector is confident, starving the Forger of gradient.

### 8. Training: a two-step dance (by 5:35)

1. Training still takes turns like DCGAN, with four more changes.
2. Detector step: DiffAugment. I have only 3,700 training letters, so the Detector could memorise them. Real and forged letters get the same random shift and cutout.
3. Forger step: a mode-seeking term penalises two different noise vectors that give the same letter.
4. DCGAN uses one learning rate; my Detector learns 4 times faster: that's TTUR. An averaged copy of the Forger, EMA, draws the final letters. And 10,000 steps instead of my first run's 1,400.

### 9. My implementation: the data (by 6:00)

1. The data: HP Labs India handwritten Tamil, 12 uyir letters, 6,190 images, split 60/40.
2. Each scan is thresholded with Otsu and cropped,
3. then thinned and redrawn with a 5-pixel pen on 128 by 128, so the Forger learns shape, not pen thickness.
4. Pixels go from minus one to plus one, matching Tanh.

### 10. Results: what my Forger learned (by 6:45)

1. Results. Same noise, from iteration 1 to 10,000: noise becomes letters.
2. Forged letters on top, real ones below.
3. A recogniser trained on real letters reads 91.6 percent of my forgeries correctly. Hardest: இ and அ, the loopiest letters.
4. Diversity is 1.61 times real handwriting, so skip-z and mode-seeking worked. The Fréchet distance shows forgeries still differ from real ones.
5. And the Detector's AUC is 1.0: it still catches every forgery.
6. Adding 1,800 forgeries to training gave no gain; real data already reaches 98 percent.

### 11. Strengths and struggles (by 7:05)

1. (just click)
2. Strengths: readable letters,
3. any letter on request,
4. and a smooth latent space.
5. Struggles: the Detector still pulls ahead,
6. mode collapse, which skip-z and mode-seeking fixed,
7. and GANs are hard to measure.

### 12. Where GANs are used (by 7:15)

1. GANs translate images, like sketch to photo,
2. sharpen images, and create extra training data,
3. and make art.

### 13. Why a tiny face detector? (by 7:35)

1. Part two: YuNet. Face detection runs on phones, robots and cameras.
2. Top detectors are huge: MogFace has 711 million parameters.
3. (just click)
4. Most faces are tiny, and deep layers mostly find the easy big ones.
5. So YuNet stays tiny and focuses on small faces.

### 14. YuNet: find every face in 1.6 ms (by 7:55)

1. I ran the real YuNet model on this picture.
2. All 7 faces found, each with a box and 5 landmarks.
3. 75,856 parameters: 360 times fewer than RetinaFace.
4. 1.6 milliseconds per frame on a normal CPU.

### 15. How a detector reads a picture (by 8:10)

1. It reads the picture as grids of cells.
2. (just click)
3. (just click)
4. Fine cells catch small faces, coarse cells big ones.
5. Each cell makes one anchor-free guess: 16 numbers.

### 16. How YuNet sees a face (by 8:25)

1. Backbone, neck, head.
2. (just click)
3. Depthwise separable convolutions keep it light.
4. A tiny feature pyramid mixes deep and fine maps.
5. (just click)
6. NMS keeps one box per face.

### 17. Inside the backbone (by 8:45)

1. Inside the backbone: real maps again.
2. (just click)
3. (just click)
4. (just click)
5. Only 64 channels per stage.
6. One DWUnit: 1 by 1, then 3 by 3 per channel, then ReLU: 8 times fewer weights.

### 18. Inside the neck and the head (by 9:00)

1. (just click)
2. The neck mixes deep and fine maps.
3. Score maps light up on faces.
4. 60 candidate boxes,
5. NMS leaves 7.

### 19. How YuNet was trained (by 9:10)

1. Trained from scratch on WIDER FACE, 32,000 images,
2. (just click)
3. (just click)
4. with a 4-part loss,
5. for 560 epochs.

### 20. YuNet results (by 9:20)

1. (just click)
2. 81.1 percent on hard faces, faster than models 8 times bigger.
3. (just click)
4. (just click)
5. (just click)

### 21. YuNet: strengths, limits and uses (by 9:35)

1. (just click)
2. Tiny,
3. real time on a CPU,
4. with free landmarks.
5. But it misses tiny blurry faces,
6. big models are more accurate,
7. and small inputs hurt.
8. It's built into OpenCV.

### 22. What to remember (by 9:55)

1. To sum up:
2. my first, DCGAN-style version gave poor Tamil letters; conditioning, spectral norm, hinge loss and augmentation made them readable, though the Detector still wins.
3. YuNet finds faces in milliseconds on a CPU.
4. Thank you. Questions?

### 23. References (by Q&A)

1. (Leave this up during questions.)
