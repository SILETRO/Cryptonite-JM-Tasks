[[Latent space]]
PCA works well with linear data only so we need SNE or t-SNE
- First it converts the pairwise distances between points into a gaussian probability distribution.
- Then it creates low dimensional points and compute their probabilities again.
- Then we minimize the distance between them to project onto 2d plane.
### Problem with SNE:
In higher dimensions many points can be moderately distant. But in **2D space**, there is very limited area. So many points get **squeezed into the center**.

t-SNE solves this by using a different way to calculate probability and uses t-distribution

### Perplexity
It basically represents how many neighbours will it look around and consider as important. Since in high dimensional space, there will be several points which are close to a given point.
Mathematically it calculates how wide would the gaussian distribution be.

- Small perplexity focuses on the small local structure
- Perplexity is a very difficult hyperparameter to adjust since tweaking it changes the entire look of the clusters.

Resources:
- [medium article](https://medium.com/@sachinsoni600517/mastering-t-sne-t-distributed-stochastic-neighbor-embedding-0e365ee898ea)
- [yt video](https://youtu.be/o_cAOa5fMhE?si=whCo6jDqJlN14JKE)