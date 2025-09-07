"""
==================================================================
Isomap Solvers Comparison Benchmark: Reconstruction Error vs. Components
==================================================================

This benchmark demonstrates how different eigen solvers in Isomap impact the reconstruction error. The goal is to analyze the effect of varying `n_components` on the error using two solvers: 'full' and 'randomized'.

Description:
------------
A fixed dataset (`digits` with 6 classes) is used for dimensionality reduction. The number of principal components tested varies between 2 and 500, with steps of 10.

Isomap models are trained on the dataset for each value of `n_components` with two different `eigen_solver` values: 'auto' (dense) and 'randomized_value'. The reconstruction errors are computed and plotted.

What you can observe:
---------------------
As the number of components increases, the reconstruction error is expected to decrease. The randomized solver provides an approximate solution and might yield a slightly higher error than the dense solver.

Going further:
--------------
You can modify the range of `n_components` to observe its effect on different datasets. Additionally, experimenting with different `n_neighbors` values may impact the results.

"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.manifold import Isomap

# Function to plot reconstruction error
def plot_reconstruction_error(n_components_list, errors_randomized, errors_full):
    plt.figure(figsize=(10, 6))
    plt.plot(n_components_list, errors_full, label='full', color='b', marker='*')
    plt.plot(n_components_list, errors_randomized, label='randomized', color='r', marker='x')

    plt.title('Reconstruction Error vs. Number of Components')
    plt.xlabel('Number of Components')
    plt.ylabel('Reconstruction Error')
    plt.legend()
    plt.grid(True)
    plt.show()

# Fixed dataset: Digits with 6 classes
digits = datasets.load_digits(n_class=6)
X, y = digits.data, digits.target

# List of components to test
n_components_list = list(range(2, 200, 10))

# Initialize lists for reconstruction errors
errors_randomized = []
errors_full = []

# Loop over different values of n_components
for n_components in n_components_list:
    # Create Isomap objects with different solvers
    isomap_randomized = Isomap(n_neighbors=30, n_components=n_components, eigen_solver='randomized_value')
    isomap_dense = Isomap(n_neighbors=30, n_components=n_components, eigen_solver='auto')

    # Fit Isomap and compute reconstruction error
    isomap_randomized.fit(X)
    errors_randomized.append(isomap_randomized.reconstruction_error())
    isomap_dense.fit(X)
    errors_full.append(isomap_dense.reconstruction_error())

# Plot reconstruction errors
plot_reconstruction_error(n_components_list, errors_randomized, errors_full)
