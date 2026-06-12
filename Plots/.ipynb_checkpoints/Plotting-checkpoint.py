import numpy as np
import torch
import pandas as pd
from bokeh.models import Band, ColumnDataSource
from sklearn.metrics import r2_score
import torch
import arviz as az
from scipy.optimize import curve_fit
from bokeh.plotting import figure, show,output_file, save
from bokeh.transform import factor_cmap, factor_mark
from bokeh.palettes import Spectral
from bokeh.models import Slope, Div,Label
from bokeh.io import curdoc,output_notebook,export_png
from bokeh.layouts import column,gridplot

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def gaussian_plot(data,model,input_x="corrected_week",resolution=500):
    x=np.linspace(data[input_x].min(),data[input_x].max(),resolution,endpoint=True)
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
def boundary_roll(data,step=2):
    data=data.set_index("corrected_week")
    data_end=data.loc[data.index>data.index.max()-step]
    data_end.index=data_end.index-data_end.index.max()
    data_start=data.loc[data.index<data.index.min()+step]
    data_start.index=data.index.max()+data_start.index
    return pd.concat([data,data_end,data_start]).reset_index()
def roll_week(data,particle,input_x="corrected_week",resolution=500,step=2):
    mean=[]
    std=[]
    x=np.linspace(data.loc[:,input_x].min(),data.loc[:,input_x].max(),resolution,endpoint=True)
    bound_data=boundary_roll(data)
    sort=bound_data.sort_values(input_x)
    for i in x:
        mean.append(sort.loc[(i-step<sort[input_x])&(i+step>sort[input_x]),particle].mean())
        std.append(sort.loc[(i-step<sort[input_x])&(i+step>sort[input_x]),particle].std())
    mean_1,std_1=pd.DataFrame(mean,index=x),pd.DataFrame(std,index=x)
    lower_std = mean_1 - std_1
    upper_std = mean_1 + std_1
    mean=pd.concat([mean_1,lower_std,upper_std],axis=1)
    mean.columns=["y","lower","upper"]
    mean=mean.set_index(x)
    mean1 = ColumnDataSource(mean)
    return mean1,mean
def plot(particle,title,train,test,model,x_label=r"$$Week \ of \ the \ year$$",y_label=r"$$\frac{\mu g}{m^3} $$",diffrence=10):
    TOOLS="hover,crosshair,pan,wheel_zoom,zoom_in,zoom_out,box_zoom,undo,redo,reset,tap,save,box_select,poly_select,lasso_select,examine,help"
    mean_train,roll_train=roll_week(train,particle)
    mean_test,roll_test=roll_week(test,particle)
    model_plot,model_dataframe=gaussian_plot(train,model)
    x_range_start=(model_dataframe.index.min(), model_dataframe.index.max())
    y_range_start=(roll_train.y.min()-diffrence,roll_train.y.max()+diffrence)
    r2_train,r2_test,r2_train_vs_test=r2_score(roll_train.y,model_dataframe.y),r2_score(roll_test.y,model_dataframe.y),r2_score(roll_test.y,roll_train.y)
    p = figure(tools=TOOLS,x_range=x_range_start,y_range=y_range_start);
    p.title.text_font_size = '15pt'
    p.title.text =title;  
    p.ygrid.grid_line_alpha=0.5;
    p.line(roll_train.index, roll_train.y, line_width=3,color="green");
    p.line(roll_test.index, roll_test.y, line_width=3,color="orange");
    p.line(model_dataframe.index, model_dataframe.y, line_width=3,color="red");
    p.scatter(train.corrected_week, y=train[particle], color="blue", marker="dot", size=20, alpha=0.4);
    band_data= Band(base="index", lower="lower", upper="upper",source=mean_train, fill_color="red", line_color="black",fill_alpha=0.2);
    band_model = Band(base="index", lower="lower", upper="upper",source=model_plot,fill_alpha=0.5, fill_color="blue", line_color="black");
    p.yaxis.axis_label_orientation  = 0
    train_label=Label(x=model_dataframe.index.max()-model_dataframe.index.max()/2,y=y_range_start[1]-diffrence/4,text=r"$$Train \ R^2$$= "+str(round(r2_train,2)),text_font_size="13pt",text_color="green")
    test_label=Label(x=model_dataframe.index.max()-model_dataframe.index.max()/2,y=y_range_start[1]-diffrence*2/4,text=r"$$Test \ R^2$$= "+str(round(r2_test,2)),text_font_size="13pt",text_color="orange")
    test_train=Label(x=model_dataframe.index.max()-model_dataframe.index.max()/2,y=y_range_start[1]-diffrence*3/4,text=r"$$Test vs train \ R^2$$= "+str(round(r2_train_vs_test,2)),text_font_size="13pt",text_color="blue")
    p.add_layout(train_label)
    p.add_layout(test_label)
    p.add_layout(test_train)
    p.add_layout(band_data);
    p.add_layout(band_model);
    p.xaxis.axis_label = x_label
    p.yaxis.axis_label = y_label
    p.xaxis.axis_label_text_font_size = "20px";
    p.yaxis.axis_label_text_font_size = "20px";
    return p