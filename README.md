# Transformer-based synthetic video generator
- Generate synthetic video data using a transformer-based architercture
  - Auto-gressive sampling is a bottleneck, can we do differently
  - Using the learned simulator for training driving models
  - Scaling to more bits and larger transformer when sampling is not the bottleneck
  - Better video tokenizer
  - Since our goal is to compress video and not generate high-fidelity ones, a VQ-VAE will do the job
  - Deal with spatio-temporal consistency

# First we will be implementing a VQ-VAE
![#VQ-GAN Architecture](assets/vqgan.PNG)

# Look at the VQ-VAE loss
![#VQ-VAE loss](assets/vqvae_loss.PNG)

- The VQ-VAE learns to compress and decompress videos by maximizing the variational lower bound
    -That is, to minimize error between compression and decompression + minimizing the error of the codebook-encoded_vector
    - Finally, the goal is to minimize the KL-divergence ratio, that is to prevent the encoder and decoder's data distribution from diverging from each other.
# GPT (Autoregressive sampling)
![#VQ-VAE loss](assets/autoreg.PNG)
- With the encoder, decoder learned, discrete latents indices pointing to codebook vectors, we proceed to training a transformer with the goal of predicting the next token
- Since a video is a set of set of tokens, the goal of the transformer is to predict tokens based on not only the current frame's tokens, but also on previous images' ones.

# Look at the attention residual block
![#Encoder-Block](assets/architect.PNG)

- Notice that we'll incorporate 3D convolution to handle video dimension, as well as positional encoding to handle the temporal dependency feature present in videos
