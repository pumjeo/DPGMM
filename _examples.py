"""Toy example for the DPGMM model"""

# Author: Neulpum Jeong <pumjeo@gmail.com>
# License: BSD 3 clause
# Time : 2024/08/31


import numpy as np
from ._DPGMM_basic import DPGMM_basic
from ._example_generator import data_generator, graph_generator

# Generate data
x, y, counts, true_label_temp = data_generator(poisson_parameter=10, scale=0.1, number_subgroups=1000, random_seed=101)

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
graph_generator(B, x, counts, predicted_label, knot, model.beta_star_mean_, model.beta_covariance_, 
                model.precision_shape_, model.precision_rate_, percentage=0.95, graph_threshold = 100, interval=True)