# Processing Electrophysiology Data to Extract Neural Trajectories

Raw electrophysiology data is very high dimensional and contains a lot of noisy, spiky, activity. Due to this, it must be heavily processed before the accurate neural trajectories can be extracted.
We utilized two important methods in our study to reduce its dimensions and smooth our data: Factor Analysis and Gaussian Processes. 

# Factor Analysis

Factor Analysis is a statistical method that attempts to reduce the dimensionality of our dataset. The algorithm we went with to accomplish this is Expectation-Maximization (EM) as it is also capable of finding latent variables in our data. As can be seen in the name, the algorithm has two parts. The expectation or E step is done by estimating the values of the the latent variables. The maximization or M step then optimizes the model's parameters by maximizing the likelihood of the model's accuracy based on our training and test data. These steps are then repeated until a reasonable convergence has occured. However, since we are creating latent variables that we are not sure exist in the data, the EM algorithm will never come to a guaranteed solution. This process helps us to decide which variables are important for our predictions, and which are not worth keeping in our model.

# Gaussian Process

Gaussian Processes are stochastic processes used to turn discrete datasets into continuous ones. It works by fitting the probabilistically most likely function onto a set of measured data points based on the Gaussian distribution. When given a set a points, an infinite number of functions can be used to describe the relationship between the values. To sample from these functions, we need to supply our Gaussian Process with the mean of our dataset and a covariance function or kernel which is chosen based on the shape of the data. The kernel function we went with is a standard squared quclidean based kernel. Using this knowledge, we can compose a Gaussian distribution of these functions and conclude that the mean function is the most likely relationship based on the definition of a Gaussian distribution. We also found that adding a little bit of noise to our values worked as a regularization tool in our predictions, and made them more accurate. This derived function will provide us with a much smoother and less noisy dataset than the raw electrophysiology data. 

# To Run

> python run.py

Outputs of run.py will be stored in "output/"