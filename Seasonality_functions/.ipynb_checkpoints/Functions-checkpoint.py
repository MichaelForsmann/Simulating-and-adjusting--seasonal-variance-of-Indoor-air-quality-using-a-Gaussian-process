import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from pyro import clear_param_store
import pyro.contrib.gp as gp
from pyro.nn import PyroSample
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS, Predictive,HMC
import torch
import arviz as az
from bokeh.models import Band, ColumnDataSource
def make_periodic(data,variable,device):
    # remove weeks before and after 0 and 52.5
    data.loc[:,"theta"]=data.loc[:,variable]/data.loc[:,variable].max()*2*np.pi
    #convert theta into a sinus and cosinus component to force periodic boundaries
    data.loc[:,"cos_theta"]=np.cos(data.loc[:,"theta"])
    #make periodic x2 component
    data.loc[:,"sin_theta"]=np.sin(data.loc[:,"theta"])
    return data 
def make_Xy(data,X,y,device):
    # remove non zeros for X,y
    data=data.dropna(subset=y)
    # return tensor for the selected X 
    X_tensor = torch.from_numpy(data.loc[:,X].dropna().values.astype("float64")).to(device)
    # create tensor for the selected y
    y_tensor=torch.tensor(data.loc[:,y].values).float().to(device)
    return X_tensor,y_tensor
def model(X,y,device):
    # clear parameters for earlier models
    clear_param_store()
    # define kernel 
    rbf = gp.kernels.RBF(input_dim=X.shape[1])
    # define distribution for the varience
    rbf.variance = PyroSample(dist.HalfNormal(y.mean()))
    # define distribution for the lengthscale
    rbf.lengthscale = PyroSample(dist.HalfNormal(torch.tensor(5.)))
    # define model and put it on device
    gpr = gp.models.GPRegression(X,y, rbf).to(device)
    # define the noise of the data
    gpr.noise = PyroSample(dist.HalfNormal(y.std()))
    # convert gpr to a nuts kernel
    nuts_kernel = NUTS(gpr.model)
    return nuts_kernel,gpr
def train_model(X,nuts_kernel,gpr,device):
    # choising method for traning and sampling   
    mcmc = MCMC(nuts_kernel,warmup_steps=7000, num_samples=7000,num_chains=1)
    #train the model with mcm sampler
    mcmc.run()
    # generate 500 samples from the posterio distribution
    posterior_samples = mcmc.get_samples(500)
    # return the predicted X,y form the model
    posterior_predictive= Predictive(gpr, posterior_samples)(X)
    #return 500 samples from before traning
    prior = Predictive(gpr, num_samples=500)(X)
    # save data as a artriz file
    pyro_data = az.from_pyro(mcmc,
    prior=prior,
    posterior_predictive=posterior_predictive)
    return pyro_data,gpr
def train_model_stability(X,nuts_kernel,gpr,device,steps=7000):
    # choising method for traning and sampling   
    mcmc = MCMC(nuts_kernel,warmup_steps=steps, num_samples=steps,num_chains=1)
    #train the model with mcm sampler
    mcmc.run()
    return gpr
def gaussian_plot(x,model,device):
    cosx=np.cos(x/x.max()*np.pi*2)
    sinx=np.sin(x/x.max()*np.pi*2)
    data=pd.DataFrame(np.array([cosx,sinx]).T,columns=["x_cos","x_sin"])
    linmod=torch.from_numpy(data.values.astype("float64")).to(device)
    model_y,model_y_std=model(linmod,full_cov=True)
    model_np,model_std_np=model_y.cpu().detach().numpy().copy(),model_y_std.diag().sqrt().cpu().detach().numpy().copy()
    lower = model_np - model_std_np
    upper = model_np + model_std_np
    data_plot=pd.DataFrame([model_np,lower,upper],index=["y","lower","upper"]).T.set_index(x)
    return ColumnDataSource(data_plot.reset_index()),data_plot
def roll_week(data,resolution,week,particle):
    mean=[]
    std=[]
    particle=particle
    x=np.linspace(0,52.5,resolution,endpoint=True)
    sort=data.sort_values("corrected_week")
    for i in x:
        mean.append(sort.loc[(i-week<sort.corrected_week)&(i+week>sort.corrected_week),particle].mean())
        std.append(sort.loc[(i-week<sort.corrected_week)&(i+week>sort.corrected_week),particle].std())
    mean_1,std_1=pd.DataFrame(mean,index=x),pd.DataFrame(std,index=x)
    lower_std = mean_1 - std_1
    upper_std = mean_1 + std_1
    mean=pd.concat([mean_1,lower_std,upper_std],axis=1)
    mean.columns=["y","lower","upper"]
    mean=mean.set_index(x)
    mean1 = ColumnDataSource(mean)
    return mean1,mean