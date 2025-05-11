Introduction:
This GitHub repo contains our reimplementation of the Noiseless Transform paper. The paper replaces stochastic noise in diffusion with fixed degradation operations like blurring, snowing, and pixelation.

Chosen Result:
We reproduce Figure 3 by training our model to reconstruct severely blurred images and evaluating its ability to restore them.

GitHub Contents:
Our repo has four main folders—code, poster, report, and results.

code/ contains all training and evaluation scripts

poster/ holds the project poster

report/ includes the final write-up

results/ shows sampled images that our model successfully unblurred

Reimplementation Details:
We embed time steps with sinusoidal positional embeddings and a two-layer MLP to condition each ConvNext block. The U-Net backbone downsamples with ConvNext blocks and linear attention, then upsamples with skip connections and a final ConvNext head, training on randomly blurred inputs using L1 loss and Adam. To reduce computational complexity, we replaced large convolutional kernels with smaller ones and reduced the number of downsampling/upsampling layers.

Reproduction Steps:

Clone the repo and install dependencies

In the code/ folder, run:
python main.py        # to train the model
python eval.py        # to compute FID using saved weights
Results:
<img width="405" alt="PNG image" src="https://github.com/user-attachments/assets/44a469f6-cbfc-4920-ab9c-5cde216f39be" />

Sample Blurred Images and Reconstructions:
<img width="611" alt="image" src="https://github.com/user-attachments/assets/0a3ca765-dc1a-452e-8436-ecc434a7a625" />

Conclusion:
Our model demonstrates strong ability to regenerate heavily blurred images. We found that kernel implementation significantly impacts performance—our initial “swish kernel” underperformed compared to the blur kernel. Future work will explore cold generation, where we suspect the model may experience mode collapse similar to GANs.

References:
Cold Diffusion: Inverting Arbitrary Image Transforms Without Noise.

