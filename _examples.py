"""Toy example for the DPGMM model"""

# Author: Neulpum Jeong <pumjeo@gmail.com>
# License: BSD 3 clause
# Time : 2024/09/04

import numpy as np
from ._DPGMM_basic import DPGMM_basic
from ._DPGMM_mixed import DPGMM_mixed
from ._example_generators import data_generator_basic, data_generator_mixed_effect, graph_generator

"""Basic Model"""

# Generate data
x, y, counts, true_label_temp = data_generator_basic(poisson_parameter=10, scale=0.1, 
                                                     number_subgroups=1000, random_seed=100)

# Generate design matrix using basis expansion
knot = np.linspace(0, 1, num=30, endpoint=False)
N = x.shape[0]
D = knot.shape[0]+4
B = np.zeros((N, D))
for i in range(N):
    B[i,:] = np.array([1, x[i], x[i]**2, x[i]**3] + [abs(x[i]-t)**3 for t in knot])

# Fit the model
model = DPGMM_basic(n_components=30, tol=1e-3, reg_covar = 1e-6, max_iter=10000, 
            random_state=42, verbose=2, verbose_interval=10).fit(B, y, counts)

# Check the results
model.weights_ # Weights of each cluster
model.beta_mean_ # Posterior mean of beta
np.sqrt(model.precision_rate_/model.precision_shape_) # Posterior mean of standard deviation
predicted_label = model.predict(B, y, counts) # Predicted label

# Draw the estimated graphs
graph_generator(B, knot, counts, model.beta_star_mean_, model.beta_covariance_, model.precision_shape_, 
                model.precision_rate_, predicted_label, percentage=0.95, 
                graph_threshold = 100, option='line_without_minor', interval=True)


"""Mixed Effect Model"""

# Generate data
x, y, true_xai, counts = data_generator_basic(poisson_parameter=10, scale=0.1, 
                                              number_subgroups=1000, random_seed=100)

# Generate design matrix using basis expansion
knot = np.linspace(0, 1, num=30, endpoint=False)
N = x.shape[0]
D = knot.shape[0]+4
B = np.zeros((N, D))
for i in range(N):
    B[i,:] = np.array([1, x[i], x[i]**2, x[i]**3] + [abs(x[i]-t)**3 for t in knot])

# Fit the model
model = DPGMM_mixed(n_components=30, n_features2=2, tol=1e-3, reg_covar = 1e-6, max_iter=10000, 
                    random_state=42, verbose=2, verbose_interval=10).fit(B, y, counts)

# Check the results
model.weights_ # Weights of each cluster
model.beta_mean_ # Posterior mean of beta
np.sqrt(model.precision_rate_/model.precision_shape_) # Posterior mean of standard deviation
predicted_label = model.predict(B, y, counts) # Predicted label

# Check the results - estimation of true xai
Sigma_diag = np.vstack([np.diag(mat) for mat in model.Sigma])
Sigma_sqrt_diag = np.sqrt(Sigma_diag)

upper = model.mu + 2 * Sigma_sqrt_diag
lower = model.mu - 2 * Sigma_sqrt_diag

print("Total number of true Xai is : ", true_xai.shape[0] * true_xai.shape[1])
print("Total number of valid Xai is : ", np.sum((lower < true_xai) & (true_xai < upper)))

# Draw the estimated graphs
graph_generator(B, knot, counts, model.beta_star_mean_, model.beta_covariance_, model.precision_shape_, 
                model.precision_rate_, predicted_label, percentage=0.95, 
                graph_threshold = 100, option='line_without_minor', interval=True)
