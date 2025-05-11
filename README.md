
Inroduction: 
This github is our reimplementation of the Noiseless Transfrom Paper. The Noiseless Transform Paper implements diffusion without stochastic noise instead 
opting for fixed degradation operations liek blurring, snowing, and pixelation. 

Chosen Result: 
We focus on reproducing Figure 3 in the paper. Specifically we trained our model to reconstruct images that have been severely blurred and tested its 
abillity to reconstruct these iamges, 

Github Contents: 
Our github has four main folders code, poster, report, results. The code folder has our all of our code for training, and evaluation. The poster folder has our poster, and the 
report folder has our report. The evlauation folder has some sampled images our model was able to unblur. 

Reimplementation Details: 
We embed time steps via sinusoidal positional embeddings and a two-layer MLP to condition each ConvNext block. 
The U Net backbone downsamples with ConvNext blocks and linear attention, then upsamples with skip connections and a final ConvNext head, training on randomly blurred inputs with L1 loss and Adam.
We made the Unet less complext by breaking large convulutional kernels into smaller kernels and reducing the amount of downsampling and upsampling layers to 
help reduce the computational complexity of the model. 

Reprodcution Steps: 
TODO

Results: 
TODO paste

Conclusion: 
Our model shows significant abillity to regenerate heavily blurred images to their original version. We learned kernel implentation significantly matters as 
our early effors to make a "swish kernel" was significantly less performant then the blur kernel. For future work, we are heavily interested in seeing how 
the model fairs for cold generation and suspect it might suffer heavy mode collapse akin to GAN's. 
References: 
Our work is based on Cold Diffusion: Inverting Arbitrary Image Transforms Without Noise 

