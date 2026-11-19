## Speech V2: 10 minutes

One numbered line = one click; "(just click)" means click and keep going. 110 clicks, about 1,098 words. The time after each slide title is when that slide should be finished. GAN architecture and activation functions get the most time; YuNet moves fast.

### 1. Title (by 0:15)

1. Good morning. I'm Sathya. My topics: GANs, with my own GAN that forges handwritten Tamil letters,
2. and YuNet, a tiny face detector.

### 2. Can a computer imagine? (by 0:35)

1. Most models judge: picture in, label out.
2. A GAN creates: random noise in, a brand-new picture out. This digit is from a small warm-up GAN I trained.
3. Generative means: learn what the data looks like, then make more of it.

### 3. The Artist and the Detective (by 1:00)

1. Think of two players: an Artist, the Generator, and a Detective, the Discriminator.
2. At first the fakes are terrible, and the Detective says: fake.
3. The Artist learns from every rejection.
4. The goal: the Detective can only guess, fifty-fifty. That's Goodfellow's idea from 2014.

### 4. The whole GAN in one picture (by 1:45)

1. My project uses the same idea. Two networks: a Forger and a Detector.
2. The Forger gets 120 random numbers, plus which letter to write, for example அ. The letter also goes to the Detector. That makes it a conditional GAN.
3. The Forger draws a forged letter.
4. The Detector also sees real handwritten letters.
5. It gives each image a realness score. For this real அ it says plus 1.1; for this forgery, minus 2.65.
6. The mistakes become the hinge loss, which trains both networks.

### 5. Inside the Forger (by 2:55)

1. Inside the Forger. The 120 noise numbers are split into 6 slices of 20. The letter becomes a vector of 128 numbers.
2. A linear layer turns the first slice and the letter into a small seed: 512 channels, 4 by 4. These are real maps from my trained model.
3. Each block doubles the size and halves the channels: 8 by 8, then 16 by 16.
4. Then 32, 64, and 128 by 128. Each block gets its own noise slice and the letter. That's skip-z, and it stops the Forger from ignoring the noise.
5. A last convolution and Tanh give a 128 by 128 letter. You can see அ appear.
6. One block: conditional batch norm, ReLU, upsample, convolution, twice, plus a shortcut. Conditional batch norm is how the letter steers every layer. I upsample then convolve instead of a transposed convolution, which avoids checkerboard patterns.

### 6. Inside the Detector (by 3:55)

1. The Detector works the other way. It gets the image and the letter.
2. Each residual block halves the size: 64, 32, 16. Spectral normalisation keeps every layer from over-reacting, so its gradient stays useful to the Forger.
3. Down to 512 channels at 4 by 4.
4. We sum over the 4 by 4 grid to get one feature vector of 512 numbers.
5. The score has two parts: does it look like handwriting, and does it look like this letter. Together: plus 1.1, real.
6. For the forged அ, both parts drop: minus 2.65, forged.

### 7. ReLU, LeakyReLU and Tanh: why and where (by 4:55)

1. Now the activation functions: where I used each one, and why.
2. ReLU keeps positives and zeroes negatives. It's fast, and the gradient is 1 for positive inputs. It's in every Forger block and every Detector block, and in YuNet. Its risk: a neuron stuck below zero passes no gradient.
3. LeakyReLU lets a small part of negatives through. Classic DCGAN discriminators use it so the gradient never dies. In my ResNet Detector, spectral norm already keeps gradients smooth, so plain ReLU works. My DCGAN option uses LeakyReLU, 0.1.
4. Tanh is only at the Forger's output. It gives minus one for paper and plus one for ink, the same range as my real letters.
5. And no sigmoid in my Detector: the hinge loss uses the raw score.

### 8. Training: a two-step dance (by 5:40)

1. Training takes turns every step.
2. Detector step, Forger frozen: push real letters above plus one and forgeries below minus one. That's hinge loss. Both get the same random shift and cutout, DiffAugment, so it can't memorise 3,700 letters.
3. Forger step, Detector frozen: push the score up. A mode-seeking term makes two noise vectors give different letters.
4. 10,000 steps of 64. The Detector learns 4 times faster, that's TTUR, and an averaged copy of the Forger draws the final letters. 50 minutes on a Colab GPU.

### 9. My implementation: the data (by 6:10)

1. The data: HP Labs India handwritten Tamil, 12 uyir letters, 6,190 images, split 60/40.
2. Each scan is thresholded with Otsu and cropped.
3. Then thinned to one pixel and redrawn with a 5-pixel pen, centred on 128 by 128. So the Forger learns shape, not pen thickness.
4. Pixels become minus one to plus one, matching Tanh.

### 10. Results: what my Forger learned (by 6:55)

1. Results. Same noise, from iteration 1 to 10,000: noise becomes letters.
2. Forged letters on top, real ones below.
3. A recogniser trained on real letters reads 91.6 percent of my forgeries correctly. The hardest are இ and அ, the loopiest letters.
4. Diversity is 1.61 times real handwriting, so no mode collapse. But the Fréchet distance shows forgeries still differ from real ones.
5. And the Detector's AUC is 1.0: it still catches every forgery.
6. Adding 1,800 forged letters to training gave no gain, because real data already reaches 98 percent.

### 11. Strengths and struggles (by 7:20)

1. (just click)
2. Strengths: readable letters,
3. any letter on request, in one consistent handwriting,
4. and a smooth latent space.
5. Struggles: keeping the balance, my Detector pulled ahead,
6. mode collapse, which I hit and fixed,
7. and GANs are hard to measure.

### 12. Where GANs are used (by 7:35)

1. GANs translate images, like horse to zebra or sketch to photo,
2. sharpen images, and create extra training data,
3. and make art and text-to-image.

### 13. Why a tiny face detector? (by 7:55)

1. Part two: YuNet. Face detection runs on phones, robots and cameras.
2. Top detectors are huge: MogFace has 711 million parameters.
3. (just click)
4. Most faces are tiny, and deep layers mostly find the easy big ones.
5. So YuNet stays tiny and focuses on small faces.

### 14. YuNet: find every face in 1.6 ms (by 8:10)

1. I ran the real YuNet model on this picture.
2. All 7 faces found, each with a box and 5 landmarks.
3. 75,856 parameters: 360 times fewer than RetinaFace.
4. 1.6 milliseconds per frame on a normal CPU.

### 15. How a detector reads a picture (by 8:22)

1. It reads the picture as grids of cells.
2. (just click)
3. (just click)
4. Fine cells catch small faces, coarse cells big ones.
5. Each cell makes one anchor-free guess: 16 numbers.

### 16. How YuNet sees a face (by 8:35)

1. Backbone, neck, head.
2. (just click)
3. Depthwise separable convolutions keep it light.
4. A tiny feature pyramid mixes deep and fine maps.
5. (just click)
6. NMS keeps one box per face.

### 17. Inside the backbone (by 8:47)

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

### 20. YuNet results (by 9:22)

1. (just click)
2. 81.1 percent on hard faces, faster than models 8 times bigger.
3. (just click)
4. (just click)
5. (just click)

### 21. YuNet: strengths, limits and uses (by 9:36)

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
2. a GAN learns by competition; mine forges readable Tamil letters, but the Detector still wins.
3. YuNet finds faces in milliseconds on a CPU.
4. Thank you. Questions?

### 23. References (by Q&A)

1. (Leave this up during questions.)
