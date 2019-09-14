import matplotlib.pyplot as plt
import numpy as np

def plot_svc_decision_function(model, Dataset, ax=None, plot_support=True,customKernel=False):
    #Plot the decision function for a two-dimensional SVM    
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # CREATE GRID TO EVALUATE MODEL
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    X, Y = np.meshgrid(x,y)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)

    # PLOT DECISION BOUNDARY AND MARGINS
    ax.contour(X, Y, P, colors='k', levels = [-1, 0, 1], alpha=0.5, linestyles = ['--', '-', '--'])

    # PLOT SUPPORT VECTORS
    if plot_support:
        if customKernel:
            ax.scatter(Dataset[np.array(model.support_),:1],
                       Dataset[np.array(model.support_),1:2],
                       s=100, linewidth=1, facecolors='none', edgecolors='k')
        else:
            ax.scatter(model.support_vectors_[:,0],
                       model.support_vectors_[:,1],
                       s=100, linewidth=1, facecolors='none', edgecolors='k')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)