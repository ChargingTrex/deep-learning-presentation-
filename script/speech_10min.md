## What I changed from a standard DCGAN, and why

A standard DCGAN (Radford et al., 2015) is unconditional, works at 64×64, and uses transposed convolutions, batch norm, LeakyReLU(0.2) and a sigmoid with binary cross-entropy, trained with Adam at 2e-4 for both networks. My first version was close to that, and it gave poor Tamil letters. Each change below answers a problem from that run.

| Standard DCGAN | My GAN | Why |
|---|---|---|
| Unconditional: you can't choose what it draws | Conditional: the letter's 128-number embedding sets every Generator layer (conditional batch norm) and enters the Discriminator's projection head | To forge the letter I ask for. Gluing the label onto the noise, the usual way, tends to lose variety |
| One noise vector (100 numbers) at the input | 120 numbers in 6 slices of 20, one slice per layer (skip-z, from BigGAN) | With noise only at the start, the Generator can ignore it and draw one fixed letter per class |
| Transposed convolutions to upsample | Nearest-neighbour upsample, then a 3×3 convolution | Transposed convolutions leave checkerboard marks |
| Plain conv stacks, 64×64 output | Residual blocks with shortcuts in both networks, 128×128 output | Smoother strokes at 128×128; shortcuts carry gradients through the deeper network |
| Batch norm in the discriminator | Spectral normalisation on every Discriminator layer | The first run's Discriminator overpowered the Generator (D loss ≈ 0.2, G loss jumping from 2 to 10). Spectral norm caps how much each layer can amplify |
| LeakyReLU(0.2) in the discriminator | ReLU (LeakyReLU 0.1 only in the plain-DCGAN option) | Spectral norm and the shortcuts already keep gradients flowing |
| Sigmoid output with binary cross-entropy | Raw score with hinge loss | A sigmoid flattens out when the Discriminator is confident, so the Generator gets almost no gradient. Hinge loss stops pushing once real is above +1 and generated below −1 |
| One realness output | Projection head: fc(h) + ⟨embed(y), h⟩ | Scores "is this handwriting?" and "is it this letter?" together |
| No augmentation | DiffAugment: the same random shift and cutout on real and generated letters | Only 3,714 training letters, so the Discriminator could memorise them |
| Nothing against mode collapse | Mode-seeking term (MSGAN) | Penalises two different noise vectors that give the same letter |
| Adam, learning rate 2e-4 for both, β = (0.5, 0.999) | Discriminator 4e-4 and Generator 1e-4 (TTUR), β = (0.0, 0.99) as in BigGAN | Keeps the Discriminator a step ahead, as a useful teacher |
| Samples from the latest weights | EMA copy of the Generator (decay 0.999), batch-norm statistics re-estimated | Smoother, steadier final letters |
| About 1,400 updates in my first run | 10,000 updates, with checkpoints | 1,400 was far too short for 128×128 |

Kept from DCGAN: ReLU and batch norm in the generator, Tanh at the output, and pixels scaled to −1..+1.

## Key ideas in depth

**Conditional batch norm (CondBN).** Plain batch norm normalises each channel over the batch, x̂ = (x − μ) / σ, then applies one learned scale γ and shift β that are the same for every image. In CondBN, γ and β come from the condition c: the letter's 128-number embedding plus that block's 20-number slice of z (148 numbers). Two linear layers turn c into a scale and shift for every channel: y = x̂ · (1 + γ(c)) + β(c). It is used twice in each of the 5 Generator blocks. That is how one Generator draws 12 different letters: with the same z and only the letter changed, the handwriting style stays and the letter switches.

**Spectral normalisation.** A weight matrix W stretches some input directions more than others; the largest stretch is its spectral norm σ(W), the largest singular value. Spectral normalisation uses W / σ(W), so no layer stretches its input by more than ×1. PyTorch estimates σ with one power-iteration step per update, so it is cheap. In my trained Discriminator the raw weights of the blocks' first convolutions have σ from 0.59 to 7.31; after dividing, every one is about 1.0. The Discriminator's score then changes smoothly with the image, so it cannot overpower the Generator (my first run: D loss ≈ 0.2, G loss jumping from 2 to 10) and the Generator always gets a useful gradient. It also makes plain ReLU safe in the Discriminator.

**Mode collapse.** The Generator finds a few shapes that fool the Discriminator and draws only those: different noise z, same letter. It happens because the Generator is rewarded only for fooling the Discriminator, never for variety, and in a DCGAN the noise enters only at the input, so it can be ignored. Fix 1, skip-z: the 120 noise numbers are split into 6 slices, one per block, so the noise reaches every layer. Fix 2, mode-seeking loss (MSGAN): for two noise vectors z1, z2 with the same letter, div = mean|G(z1) − G(z2)| / mean|z1 − z2|, and the Generator's loss adds λ / min(div, target), with λ = 1 and target = 0.9 × the variety of real handwriting. If two different z give similar letters, div is small and the penalty is big.

**How I measured accuracy.** A GAN has no right answer to compare with, so I used a separate judge. (1) Train a ResNet-18 recogniser on the 3,714 real training letters only; it reads 98.3% of the 2,476 real test letters correctly. (2) Generate 2,476 letters, the same number as the test set, spread evenly over the 12 letters. (3) The recogniser reads every forgery; it has never seen a generated letter. (4) Generatory accuracy is the share it reads as the letter I asked for: 91.6% overall; the hardest are இ (69.1%) and அ (76.8%).

## Speech (formal): 10 minutes

One numbered line = one click; "(just click)" means click and keep going. 24 slides, 115 clicks, about 1,102 words. The time after each slide title is when that slide should be finished. The GAN part focuses on how the code and architecture work, with conditional batch norm, spectral normalisation and mode collapse explained in depth. YuNet moves fast.

### 1. Title (by 0:15)

1. Good morning. My name is Sathya. First, my conditional GAN for handwritten Tamil letters, and why each design choice was made.
2. Second, I present YuNet, a compact face detector.

### 2. Design changes from a standard DCGAN (by 0:50)

1. A GAN has a generator that produces samples and a discriminator that tells them from real data. DCGAN is the standard design; my first version followed it and produced poor letters.
2. So I changed the Generator: class conditioning, noise in every block, no checkerboard artefacts, residual blocks.
3. In the Discriminator: spectral normalisation, ReLU, hinge loss and a projection head.
4. The following slides explain each change.

### 3. System overview (by 1:25)

1. This is the overall system: the Generator and the Discriminator.
2. The Generator gets 120 random numbers plus the letter to write, like அ. The Discriminator gets the letter too.
3. The Generator draws a 128 by 128 letter.
4. The Discriminator also sees real letters.
5. It gives a raw score: plus 1.1 for this real அ, minus 2.65 for the generated one.
6. The hinge loss on those scores trains both networks.

### 4. Generator architecture (by 2:10)

1. Inside the Generator: z in 6 slices of 20, and the letter as 128 numbers.
2. A linear layer makes a 512 by 4 by 4 seed. These are real maps from my trained model.
3. Five residual blocks. Each one doubles the size and halves the channels: 8, 16,
4. 32, 64, 128. Each block gets its own z slice and the letter.
5. Batch norm, ReLU, a 3 by 3 convolution and Tanh give the letter.
6. One block: conditional batch norm, ReLU, upsample, convolution, twice, plus a shortcut.

### 5. Conditional batch normalisation (by 2:50)

1. Conditional batch norm is how the letter steers the Generator.
2. Plain batch norm uses one learned scale and shift for every image, so it cannot know which letter to draw.
3. Conditional batch norm computes the scale and shift from the letter and the z slice. These real values from my Generator differ for அ and இ.
4. In code: a batch norm plus two linear layers.
5. Same z, all 12 letters: the noise sets the style, the label sets the letter.

### 6. Discriminator architecture (by 3:25)

1. The Discriminator gets the image and the letter.
2. Six residual blocks, each with spectral normalisation, halve the size: 64, 32, 16,
3. down to 512 channels at 4 by 4.
4. A sum over the grid gives one vector of 512 numbers.
5. The projection head scores two things: handwriting? and this letter? Plus 1.1: real.
6. For the generated அ, both drop: minus 2.65.

### 7. Spectral normalisation (by 4:10)

1. Spectral normalisation keeps the two networks balanced.
2. In my first run the Discriminator dominated: its loss fell to 0.2, while the Generator's rose from 2 to 10.
3. A weight matrix stretches its input; the biggest stretch is sigma, its largest singular value. Dividing by sigma caps every layer at one times.
4. PyTorch estimates sigma with one cheap power-iteration step per update. I wrap every Discriminator layer with it.
5. My raw weights stretch up to 7.3 times; after dividing, every layer is 1, so the Generator always gets a useful gradient.

### 8. Activation functions: ReLU, LeakyReLU and Tanh (by 4:50)

1. Activation functions: where each one is, and why.
2. ReLU keeps positives and zeroes negatives. It is in every Generator and Discriminator block, and in YuNet.
3. LeakyReLU passes a little of the negatives so gradients never die. DCGAN's discriminator needs it; mine does not, thanks to spectral norm.
4. Tanh is only at the Generator's output: minus one for paper, plus one for ink, the same range as my real letters.
5. And no sigmoid in the Discriminator: hinge loss needs the raw score.

### 9. Training procedure (by 5:15)

1. Training takes turns every step.
2. Discriminator step: real above plus one, generated below minus one, with DiffAugment so it cannot memorise 3,714 letters.
3. Generator step: push the score up, plus the mode-seeking term.
4. 10,000 steps of 64. The Discriminator learns 4 times faster, and an averaged copy of the Generator draws the final letters.

### 10. Mode collapse and its mitigation (by 5:50)

1. Mode collapse occurs when the Generator repeats one output that works: different noise, same letter.
2. It is rewarded only for fooling the Discriminator, not for variety.
3. Fix one, skip-z: the noise goes into every block, so it cannot be ignored.
4. Fix two, a mode-seeking loss: if two different z give similar letters, the penalty grows.
5. The result: six noise vectors give six different handwriting styles of அ.

### 11. Dataset and preprocessing (by 6:20)

1. The data: HP Labs India handwritten Tamil, 12 uyir letters, 6,190 images, split 60/40.
2. Each scan is thresholded with Otsu and cropped,
3. then thinned and redrawn with a 5-pixel pen on 128 by 128, so the Generator learns shape, not pen thickness.
4. Pixels go from minus one to plus one, matching Tanh.

### 12. Results (by 6:50)

1. Results: with the same noise from step 1 to 10,000, the output becomes letters.
2. Generated letters are shown above, real letters below.
3. A ResNet-18 recogniser trained only on real letters, 98.3 percent accurate on real test letters, reads 2,476 generated letters.
4. It reads 91.6 percent as the intended letter. The loopy இ and அ are the hardest.

### 13. Advantages and limitations (by 7:05)

1. (just click)
2. The advantages are recognisable letters,
3. class-conditional output,
4. and a smooth latent space.
5. The limitations are training balance,
6. mode collapse, which I mitigated,
7. and the difficulty of evaluating GANs.

### 14. Applications (by 7:15)

1. Applications include image translation, such as sketch to photo,
2. super-resolution and training-data generation,
3. and art.

### 15. YuNet: motivation (by 7:40)

1. Part two is YuNet. Face detection runs on phones, robots and cameras.
2. Leading detectors are large: MogFace has 711 million parameters.
3. (just click)
4. Most faces are tiny, and deep layers mostly find the easy big ones.
5. So YuNet stays compact and focuses on small faces.

### 16. YuNet: overview (by 8:00)

1. I ran the real YuNet model on this picture.
2. It finds all 7 faces, each with a box and 5 landmarks.
3. 75,856 parameters: 360 times fewer than RetinaFace.
4. 1.6 milliseconds per frame on a normal CPU.

### 17. Anchor-free detection on a grid (by 8:15)

1. The detector divides the image into grids of cells.
2. (just click)
3. (just click)
4. Fine cells catch small faces, coarse cells big ones.
5. Each cell makes one anchor-free prediction of 16 numbers.

### 18. YuNet architecture (by 8:30)

1. Backbone, neck, head.
2. (just click)
3. Depthwise separable convolutions keep it light.
4. A tiny feature pyramid mixes deep and fine maps.
5. (just click)
6. NMS keeps one box per face.

### 19. Backbone (by 8:50)

1. These are real feature maps from the backbone.
2. (just click)
3. (just click)
4. (just click)
5. Only 64 channels per stage.
6. One DWUnit: 1 by 1, then 3 by 3 per channel, then ReLU: 8 times fewer weights.

### 20. Neck and detection head (by 9:05)

1. (just click)
2. The neck mixes deep and fine maps.
3. Score maps light up on faces.
4. 60 candidate boxes,
5. NMS leaves 7.

### 21. YuNet training (by 9:15)

1. Trained from scratch on WIDER FACE, 32,000 images,
2. (just click)
3. (just click)
4. with a 4-part loss,
5. for 560 epochs.

### 22. YuNet: advantages, limitations and applications (by 9:35)

1. (just click)
2. It is compact,
3. real time on a CPU,
4. and provides landmarks.
5. However, it misses tiny, blurry faces,
6. larger models are more accurate,
7. and small inputs reduce accuracy.
8. It is available in OpenCV.

### 23. Conclusions (by 9:55)

1. To conclude:
2. my GAN extends DCGAN with conditional batch normalisation, spectral normalisation, hinge loss and mode-seeking; 91.6 percent of its letters are recognised.
3. YuNet finds faces in milliseconds on a CPU.
4. Thank you. I welcome your questions.

### 24. References (by Q&A)

1. (Leave this up during questions.)
