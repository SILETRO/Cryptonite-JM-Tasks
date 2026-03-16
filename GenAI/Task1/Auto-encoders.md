[[Neural Networks]] [[GenAI]] [[Regularization]]

Autoencoders are a special type of neural networks that learn to compress data into a compact form and then reconstruct it to closely match the original input. They consist of an:

- Encoder that captures important features by reducing dimensionality.
- Decoder that rebuilds the data from this compressed representation.

### Encoder
It compress the input data into a smaller, more manageable form by reducing its dimensionality while preserving important information. It has three layers: input, hidden, and output
- The input layer is where raw data is entered. 
- The hidden layers applies weights and  activations which compress the data. 
- The encoder outputs a compressed vector known as the latent representation or encoding.
### Bottleneck
It is the smallest layer of the network which represents the most compressed version of the input data. It serves as the information bottleneck which force the network to prioritize the most significant features. This compact representation helps the model learn the underlying structure and key patterns of the input helps in enabling better generalization and efficient data encoding.
### Decoder
- Hidden layer and output layer tries to best restore the the original data. 
-  The reconstructed output is then compared to the “ground truth” to gauge the efficacy of the autoencoder. The difference between the output and ground truth is called _reconstruction error._
### Regularization and Bottleneck
It may be possible for the NN to reconstruct the output perfectly. Thus learning the identity function. To prevent this we use bottleneck and regularization.
Adding a bottleneck is achieved by making the latent feature’s dimensionality lower (often much lower) than the input’s.

### Reconstruction Error
It tells us how well the the autoencoder is capable of reconstructing the data. The most typical RE is MSE.

### Applications
- In dimensionality reduction, it can outperform PCA because it is capable of considering non-linear transformations also and can deal with big amount of data efficiently.
- It is used for anomaly detection. If data point with very high reconstruction error compared to other data points would most likely be an anomaly. It can also be used to find outliers.
- It can be used to remove noise from corrupted, noisy data forcing it to learn robust features.
### Resources
- [Autoencoders in Machine Learning - GeeksforGeeks](https://www.geeksforgeeks.org/machine-learning/auto-encoders/)
- [What Is an Autoencoder? | IBM](https://www.ibm.com/think/topics/autoencoder)
- An Introduction to Autoencoders Umberto Michelucci