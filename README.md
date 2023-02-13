# Processing Electrophysiology Data to Extract Neural Trajectories

Raw electrophysiology data is very high dimensional and contains a lot of noisy, spiky, activity. Due to this, it must be heavily processed before the accurate neural trajectories can be extracted.
We utilized two important methods in our study to reduce its dimensions and smooth our data: Factor Analysis and Gaussian Processes. 

# Variational Latent Gaussian Process

In a variational latent Gaussian process (VLGP), the observed data, y, is modeled as a Gaussian process, with mean function, f(x), and covariance function, k(x, x') as explained in section 2.2. The underlying structure in the data is captured by latent variables, z, which are treated as random variables. The prior distribution over the latent variables is modeled as a Gaussian distribution.

The goal of the VLGP is to infer the posterior distribution, q(z|x), over the latent variables given the observed data. This is done using variational inference by minimizing the objective function, also known as the evidence lower bound (ELBO), given by:
$ELBO = -D_{KL}(q(z|x) || p(z)) + E_{q(z|x)}[log(p(y|z,x))]$
where $D_{KL}$ is the Kullback-Leibler divergence, which measures the difference between two distributions, and E is the expected value. The first term in the ELBO encourages the approximate posterior, q(z|x), to be close to the prior, p(z), while the second term represents the negative log-likelihood of the data given the latent variables. [1]

The optimization problem can be solved using gradient-based optimization algorithms, such as gradient descent or conjugate gradient. The solution provides estimates of the latent variables, which can be used to reconstruct the hidden patterns in the data. For the purposes of our project, vLGP is used to extract neural trajectories, which are the underlying patterns in neural activity that reflect how the brain processes information.
# To Run

> python run.py

Outputs of run.py will be stored in "output/"
