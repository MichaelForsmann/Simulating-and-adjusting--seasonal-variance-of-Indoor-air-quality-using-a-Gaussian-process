
# Inferring indoor air quality from incomplete data and Gaussian Process modelling
This project/paper was made to correct and simulate indoor air pollution in cohorts.
Some particles and gasses have a strong seasonal component due to temperature and habits of the homes, such as $NO_x$ and Black Carbon. 
When working with data on these components, adjusting for the seasonality is extremely important since different periods can have 2-4 fold difference due to traffic and family habits.
## Install Environment 
```
conda env create -f Gaussian-environment.yml
```
## Data
The data is a synthetic version of the Original COPSAC data. It has the household ID and the measurement of the house where it was taken.
There are multiple indoor and outdoor sources and sinks. The indoor is the number of days the kids are exposed to the source.
The outdoor sources are the total area of the outdoor source as an approximation for the total exposure rate.
The synthetic data was created since it is clinical data, which doesn't allow us to open-source it. 
The data is mostly there to showcase the application and that the code is working.
## Pipeline 
![plot](https://github.com/MichaelForsmann/Inferring-indoor-air-quality-from-incomplete-data-and-Gaussian-Process-modelling/blob/main/Seasonality_functions/Plots/Pipeline.png)
## Plots and performence to estimate general function for seasonality of the gases and particles
![plot](https://github.com/MichaelForsmann/Inferring-indoor-air-quality-from-incomplete-data-and-Gaussian-Process-modelling/blob/main/Seasonality_functions/Plots/seasonal_function.png)
### References 
- [Pyro: bingham2019pyro:](https://arxiv.org/abs/1810.09538) 
- [Scipy: 2020SciPy-NMeth:](https://www.nature.com/articles/s41592-019-0686-2)
- [PyTorch: An Imperative Style, High-Performance Deep Learning Library]( https://openreview.net/forum?id=BJJsrmfCZ)
- [arviz: arviz_2019:](https://joss.theoj.org/papers/10.21105/joss.01143)
- [Kennard-stone](https://www.researchgate.net/publication/357491012_Kennard-Stone_method_outperforms_the_Random_Sampling_in_the_selection_of_calibration_samples_in_SNPs_and_NIR_data)
