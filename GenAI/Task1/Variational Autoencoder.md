[[Auto-encoders]] [[GenAI]]
Latent space is the lower dimension compressed representation of the input data. It captures the most essential features and thus helps in the dimensionality reduction.

Unlike vanilla auto-encoders that encode a single vector of discrete latent variables, VAEs encode latent variables of training data not as a fixed discrete value **z**, but as a continuous range of possibilities expressed as a probability distribution **_p_(_z_)**.

By randomly sampling from within this range of encoded possibilities, VAEs can synthesize new data samples that, while unique and original unto themselves, resemble the original training data.

### Reconstruction Loss
Just like autoencoders, we need to measure the difference between the original raw input compared to the reconstructed version. This is done by measuring the mean squared error or cross entropy. 

###  Kullback-Leibler divergence
It measures how different the two probability distributions are. How much information is lost when we approximate P with Q?
- If distributions are identical → KL = 0
- If they differ → KL increases
Minimizing the KL divergence between the learned distribution of latent variables and a simple Gaussian distribution whose values range from 0 to 1 forces the learned encoding of latent variables to follow a normal distribution. This allows for smooth interpolation of any point in latent space, and thereby the generation of new images.

### ELBO
$$E_{q(z∣x)}​[logp(x∣z)]−D_{KL}​(q(z∣x)∣∣p(z))$$

- The first term encourages decoder to reconstruct the input . This is equivalent of reconstruction loss.
- The second term forces the latent distributions to match the prior(mostly gaussian distribution)
### Re-parameterization Trick
The encoder outputs:
$q(z∣x)=N(μ(x),σ(x)2)q(z|x) = N(\mu(x),\sigma(x)^2)q(z∣x)=N(μ(x),σ(x)2)$

To generate a latent vector we sample:
$z∼N(μ,σ2)$
Since we do random sampling, there is no “best” random; a vector of random values, by definition, has no derivative and therefore can’t be optimized through backpropagation by using any form of gradient descent.
In order to solve this problem we use a random value $\epsilon$ selected from the random distribution between 0 and 1.
$$z=μ+σϵ$$
This makes the function differentiable.