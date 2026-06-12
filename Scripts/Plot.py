import matplotlib.pyplot as plt
import jax
import jax.numpy as np
import numpy as np
def plot_gaussian(data, ax, cols,rlabel, axis=0):
    X_full = data["/constant_data"].X_test.values
    if len(X_full.shape)>1:
        index_test = np.argsort(X_full[:, cols])
        X_test = X_full[:, cols][index_test]
    else:
        index_test = np.argsort(X_full)
        X_test = X_full[index_test]

    y_pp = data["/posterior_predictive"].y
    mean_prediction = np.mean(y_pp, axis=(0, 1))
    lower, upper = np.percentile(y_pp, [5, 95], axis=(0, 1))

    r2  = data["/constant_data"]["r2_per_draw"].values
    rms = data["/constant_data"]["rmse_per_draw"].values
    log_like = data["/log_likelihood"]["y"].mean(axis=(0,1,2)).round(2).values
    mean_r2,  std_r2  = np.round(r2.mean(), 2),  np.round(r2.std(), 2)
    mean_rmse, std_rmse = np.round(rms.mean(), 2), np.round(rms.std(), 2)
    label = rlabel+" $R^2$:"+str(mean_r2)+" $\pm$"+str(std_r2)+", "+"RMSE: "+str(mean_rmse)+"$\pm$"+str(std_rmse)+" -Loglike: "+str(log_like) 

    ax.plot(X_test, mean_prediction[index_test], "-", label=label)
    ax.fill_between(X_test, lower[index_test], upper[index_test], alpha=0.4)