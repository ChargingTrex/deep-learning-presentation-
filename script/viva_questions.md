## Likely questions and short answers

Answer in one or two sentences, then stop. "Good question, I'd need to check that" is a fine answer too.

### The Tamil GAN

| Question | Short answer |
|---|---|
| What did you change from a standard DCGAN, and why? | My first version was close to DCGAN and gave poor letters. I made it conditional (conditional batch norm, projection head), used skip-z and a mode-seeking term against mode collapse, spectral norm, hinge loss and TTUR so the Discriminator couldn't overpower the Generator, DiffAugment against memorising 3,714 letters, upsample-then-convolve against checkerboards, and an EMA Generator. The full table is in speech_10min.md. |
| Why is it called adversarial? | The two networks are opponents. The Generator wins when the Discriminator scores a forgery as real; the Discriminator wins when it catches it. |
| Supervised or unsupervised? | Generative and mostly unsupervised: there is no target image for each output. The letter labels only say which letter to draw, which makes it a conditional GAN. |
| What is z in your model? | 120 random numbers from a normal distribution, split into 6 slices of 20. The first slice goes into the linear layer and one slice goes into each of the 5 blocks (skip-z). |
| What makes it conditional? | The letter becomes a 128-number embedding. In the Generator it sets the scale and shift of every batch norm; in the Discriminator it enters the projection head. |
| What is conditional batch norm? | Normal batch norm, but the scale (γ) and shift (β) are computed from the letter and that block's noise slice, so the letter steers every layer. |
| Why upsample + conv, not ConvTranspose2d? | Stride-2 transposed convolutions overlap unevenly and leave checkerboard patterns. Nearest upsampling followed by a 3×3 conv avoids them. |
| Why Tanh at the end of the Generator? | It outputs −1 to +1, the same range as my normalised real letters (−1 paper, +1 ink), so real and generated look alike to the Discriminator. |
| Why ReLU in your Discriminator, not LeakyReLU? | Spectral norm keeps the Discriminator smooth and the residual shortcuts carry gradients, so plain ReLU works, as in SNGAN and BigGAN. My DCGAN option, which has no shortcuts, uses LeakyReLU(0.1). |
| Why no sigmoid in the Discriminator? | Hinge loss uses the raw score. A sigmoid saturates once the Discriminator is confident, and then the Generator gets almost no gradient. |
| What is hinge loss? | Discriminator: max(0, 1 − D(real)) + max(0, 1 + D(fake)), so real scores go above +1 and generated letters below −1, with no reward for going further. Generator: −D(fake). |
| What does spectral normalisation do? | It divides each layer's weights by their largest singular value, so no layer can stretch its input more than 1×. The Discriminator stays smooth and training stays stable. |
| What is the projection head? | Score = fc(h) + ⟨embed(y), h⟩. The first part asks "is this handwriting?", the second asks "is it this letter?" (Miyato and Koyama, 2018). |
| Why DiffAugment? | With only 3,714 training letters the Discriminator could memorise them. The same random shift and cutout go on real and generated letters, and because it is differentiable the Generator still gets gradients. |
| What is TTUR? | Two time-scale update rule: the Discriminator learns at 4e-4 and the Generator at 1e-4, so the Discriminator stays a useful teacher. |
| What is EMA? | A moving average of the Generator's weights. The averaged copy draws the final letters, which are smoother and steadier than the latest weights. |
| What is mode collapse? Did you see it? | The Generator makes only a few shapes. I hit it, and a mode-seeking term (image difference divided by noise difference) fixed it. Diversity is now 1.61 times real handwriting. |
| How did you measure accuracy? | With a separate judge: a ResNet-18 trained only on real letters (98.3% on the real test letters) reads 2,476 generated letters it has never seen. 91.6% are read as the letter I asked for. The notebook also reports Fréchet distance, diversity and the Discriminator's AUC. |
| The Discriminator's AUC is 1.0. Is that bad? | It means the Discriminator still separates every forgery from real letters: the generated letters are readable but not yet indistinguishable. That is the honest limit after 10,000 steps. |
| Did generated letters help the recogniser? | No: 98.26% with real letters only, 98.22% with 1,800 generated letters added. Real data already saturates it; generated letters may help when real data is scarce. |
| Why are இ and அ the hardest? | They are the loopiest letters, where a small change in a loop or curve changes the letter: 69.1% and 76.8% read correctly. |
| Why thin the letters and redraw at 5 pixels? | Writers use different pens. Thinning then redrawing gives every stroke the same width, so the Generator learns shape, not pen thickness. |
| What is the minimax game? | Goodfellow's game: min over G, max over D of E[log D(x)] + E[log(1 − D(G(z)))]. My model plays the hinge version of the same game. |
| Did your Discriminator reach 50/50? | No, it stayed ahead (AUC 1.0). 50/50 is the theoretical ideal; in practice GANs rarely reach it. |
| How long did training take? | 10,000 steps with batch 64: 49.5 minutes on a Colab T4 GPU with mixed precision. |
| What would you improve? | Train longer or with a bigger Generator, add more writers, extend to more Tamil letters, and ask people to judge the generated letters. |

### Activation functions

| Question | Short answer |
|---|---|
| Why ReLU in YuNet? | YuNet learns straight from labelled faces, so the cheapest activation is fine; no Generator depends on its gradients. |
| Why not Sigmoid or Tanh in hidden layers? | They saturate: for large inputs their slope is almost zero, so gradients vanish in deep networks. ReLU-type functions don't saturate for positive inputs. |

### YuNet

| Question | Short answer |
|---|---|
| What does anchor-free mean? | No preset anchor boxes. YuNet predicts one box per location directly, which is simpler and faster. |
| What is a depthwise separable conv? | It splits a normal 3×3 conv into two cheap parts: a 1×1 conv that mixes channels and a 3×3 conv on each channel alone. About 1/8 to 1/9 the cost (36,864 vs 4,672 weights for 64 channels). |
| Why does YuNet focus on small faces? | Most WIDER FACE faces are under 20 pixels, and small faces come from shallow, high-resolution maps. So deep layers stay thin. |
| What dataset tests YuNet? | WIDER FACE: 32,203 images, 393,703 faces, in easy, medium and hard sets. |
| What is NMS? | Non-maximum suppression: keeps the highest-scoring box and removes overlapping duplicates (YuNet's tests use IoU 0.45). |
| What is simOTA? | A training rule that decides which grid cells are responsible for each real face: it picks the cells whose predicted boxes fit that face best. |
| What is the EIoU loss? | An extended IoU loss: it trains the box by how much it overlaps the true box, which is how detections are judged anyway. |
| How did you get YuNet's feature maps and boxes? | I ran the released YuNet model (ONNX file from the authors' repository) with ONNX Runtime on my own illustration, and saved the outputs of its inner layers. It found all 7 faces in about 3 ms. |
| Why does YuNet stay at 64 channels? | Deep layers mostly find big, easy faces. The paper shows RetinaFace's deepest layer has about 64% of its parameters but finds under 10% of the faces, so YuNet keeps those layers thin. |
| How do GAN and YuNet relate? | Different jobs: GANs generate images, YuNet detects faces. Both are CNNs, and GANs can make extra training images for detectors. |
