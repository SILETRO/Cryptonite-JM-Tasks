[[Latent space]]
PCA (Principal Component Analysis) is a dimensionality reduction technique and helps us to reduce the number of features in a dataset while keeping the most important information. It changes complex datasets by transforming correlated features into a smaller set of uncorrelated components.
- Standardize the data and then create a co-variance matrix to see how components relate to eachother.
- Identify new axes where the data spreads the most (covers the most variance). These directions come from the eigenvectors of the covariance matrix and their importance is measured by eigenvalues
- Select the top k components and transform the data by projecting it onto the top components.
