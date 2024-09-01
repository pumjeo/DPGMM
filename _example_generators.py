"""Toy example generators for the DPGMM model"""

# Author: Neulpum Jeong <pumjeo@gmail.com>
# License: BSD 3 clause
# Time : 2024/08/31

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def signal1(x):
    return x + 2*np.exp(-(16*(x-0.5))**2)-0.5

def signal2(x):
    return np.sin(2*np.pi*x**3)**3

def signal3(x):
    return np.sqrt(x*(1-x)) * np.sin((2*np.pi*(1+2**(-3/5)))/(x+2**(-3/5))) + 0.1

def data_generator(poisson_parameter, scale, number_subgroups, random_seed):
    """Make functional linear regression data from three different smooth functions"""
    
    np.random.seed(random_seed)
    counts = np.random.poisson(poisson_parameter, size=number_subgroups)

    x = np.array([])
    y = np.array([])
    
    temp = number_subgroups//3
    a = temp
    b = 2*temp
    
    for i in range(0, a):
        temp_x = np.random.uniform(0, 1, size=counts[i]) 
        temp_y = signal1(temp_x) + np.random.normal(scale=scale, size=counts[i]) # scale is standard deviation
        x = np.concatenate((x, temp_x), axis=0)
        y = np.concatenate((y, temp_y), axis=0)

    for i in range(a, b):
        temp_x = np.random.uniform(0, 1, size=counts[i]) 
        temp_y = signal2(temp_x) + np.random.normal(scale=scale, size=counts[i]) 
        x = np.concatenate((x, temp_x), axis=0)
        y = np.concatenate((y, temp_y), axis=0)

    for i in range(b, number_subgroups):
        temp_x = np.random.uniform(0, 1, size=counts[i])
        temp_y = signal3(temp_x) + np.random.normal(scale=scale, size=counts[i])
        x = np.concatenate((x, temp_x), axis=0)
        y = np.concatenate((y, temp_y), axis=0)

    point1 = np.sum(counts[:a])
    point2 = point1 + np.sum(counts[a:b])
    point3 = point2 + np.sum(counts[b:number_subgroups])

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(x[:point1], y[:point1], alpha = 0.4, s=2)
    ax.scatter(x[point1:point2], y[point1:point2], alpha = 0.4, s=2)
    ax.scatter(x[point2:point3], y[point2:point3], alpha = 0.4, s=2)
    plt.show()

    print(x.shape)
    
    true_label_temp = np.concatenate([np.repeat(0,a), np.repeat(1,b-a), np.repeat(2,number_subgroups-b)], axis=0)
    
    return x, y, counts, true_label_temp


def graph_generator(B, x, counts, predicted_label, knot, mean_star, 
                    C_beta, a, b, percentage=0.95, graph_threshold = 100, interval=True):
    """ Draw the estimated graphs that describes the basis expansion from each cluster with 95% credible interval"""
    K, _ = mean_star.shape
    
    # Design Matrix Standardization
    intercept = B[:, 0]
    mean_B = np.mean(B[:, 1:], axis=0) 
    std_dev_B = np.std(B[:, 1:], axis=0)
    standardized = (B[:, 1:] - mean_B) / std_dev_B
    B_star = np.column_stack((intercept, standardized))

    x_split = np.split(x, np.cumsum(counts)[:-1], axis=0)
    B_star_split = np.split(B_star, np.cumsum(counts)[:-1], axis=0)

    BS = [[] for _ in range(K)]
    x_graph = [[] for _ in range(K)]

    for i in range(counts.shape[0]):
        k = predicted_label[i]
        temp = np.dot(B_star_split[i], mean_star[k])
        BS[k].extend(temp)
        x_graph[k].extend(x_split[i])
    
    # True Graph
    x_space = np.linspace(0, 1, num=500, endpoint=False)
    graph_y1 = signal1(x_space)
    graph_y2 = signal2(x_space)
    graph_y3 = signal3(x_space)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x_space, graph_y1, linestyle='--', lw=1, c='black', label='True Graph')
    ax.plot(x_space, graph_y2, linestyle='--', lw=1, c='black')
    ax.plot(x_space, graph_y3, linestyle='--', lw=1, c='black')


    # Estimated Graph
    for k in range(K):
        if len(BS[k]) == 0 : continue
        ax.scatter(x_graph[k], BS[k], s=2, label='Estimate Graph of '+str(k+1))

    if interval:
        # 95% Credible Interval
        x_new = np.random.uniform(0, 1, size=10000)
        N_new = x_new.shape[0]
        D_new = knot.shape[0]+4
        B_new = np.zeros((N_new, D_new))

        for i in range(N_new): 
            B_new[i,:] = np.array([1, x_new[i], x_new[i]**2, x_new[i]**3] + [abs(x_new[i]-t)**3 for t in knot])

        # Transformation using original B
        mean_B = np.mean(B[:, 1:], axis=0) 
        std_dev_B = np.std(B[:, 1:], axis=0)

        intercept = B_new[:, 0]
        standardized = (B_new[:, 1:] - mean_B) / std_dev_B
        B_new_star = np.column_stack((intercept, standardized))

        for k in range(K):
            if len(BS[k]) < graph_threshold : continue
            mu = np.dot(B_new_star, mean_star[k])
            phi = np.einsum('ij,ij->i', B_new_star, np.dot(B_new_star, C_beta[k]))
            quantile_t = stats.t(df=2*a[k]).ppf((1-(1-percentage)/2))
            lower = mu - np.sqrt(phi*(b[k]/a[k])) * quantile_t # Lower Bound
            upper = mu + np.sqrt(phi*(b[k]/a[k])) * quantile_t # Upper Bound

            graph_interval_mat = np.vstack((x_new, lower, upper)).T
            sorted_graph_interval_mat = graph_interval_mat[graph_interval_mat[:,0].argsort()] # X-axis Sorting
            ax.fill_between(sorted_graph_interval_mat[:,0], sorted_graph_interval_mat[:,1], sorted_graph_interval_mat[:,2], 
                            color='blue', alpha=0.3) # Confidence Interval

    ax.legend()
    plt.title('Estimated Result')
    plt.show()
    
    for k in range(K):
        if len(BS[k]) > 0: 
            print("     " + str(k+1) + "th Cluster Has", len(BS[k]), "Samples.")
    
    return None