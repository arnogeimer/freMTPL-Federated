# -*- coding: utf-8 -*-
import warnings
import numpy as np
import tqdm
import matplotlib.pyplot as plt
import auxiliary_functions as aux
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_poisson_deviance

warnings.filterwarnings("ignore", category=UserWarning)

clientnbr=22
localupdates=aux.localupdates
num_rounds=aux.num_rounds
bins=aux.bins

class GLM():
    def __init__(self) -> None:
        self.model= Pipeline(
            [
                ("preprocessor", aux.linear_model_preprocessor),
                ("regressor", PoissonRegressor(alpha=1e-12,warm_start=True)),
            ]
        )

def train(glm, trainloader, epochs): 
    glm.model.named_steps['regressor'].max_iter = epochs
    features,targets,sample_weights=trainloader
    glm.model.fit(features, targets, regressor__sample_weight=sample_weights)

def test(glm, testloader):
    features,targets,sample_weights=testloader
    y_pred=glm.model.predict(features)
    mse = mean_squared_error(
        targets, y_pred, sample_weight=sample_weights
    )
    mae = mean_absolute_error(
        targets, y_pred, sample_weight=sample_weights
    )
    mask = y_pred > 0
    if (~mask).any():
        n_masked, n_samples = (~mask).sum(), mask.shape[0]
        print(
            "WARNING: Estimator yields invalid, non-positive predictions "
            f" for {n_masked} samples out of {n_samples}. These predictions "
            "are ignored when computing the Poisson deviance."
        )
    mpd=mean_poisson_deviance(
        targets[mask],
        y_pred[mask],
        sample_weight=sample_weights[mask],
    )
    return mpd

def load_data(dataframe):
    return (dataframe,dataframe["Frequency"],dataframe["Exposure"])

def Plot(clientnbr):
    fig, (ax1,ax2) = plt.subplots(1, 2, sharex=False, figsize=(6, 6))
    ax2.set_xlim(0, 1.0)
    ax2.set_ylim(0, 0.5)
    ax2.set(
        xlabel="Fraction of samples sorted by y_pred",
        ylabel="Mean Frequency (y_pred)",
    )
    boxplots=[]
    minimum = False
    maximum = False
    for i in tqdm.trange(clientnbr):
        df_=aux.df[aux.df["Region"]==aux.Regions[i]]
        glm = GLM()
        trainloader = load_data(df_)
        globaltestloader=load_data(aux.df)
        features,targets,sample_weights=globaltestloader
        
        clientloss=[]
        for j in range(num_rounds):
            if j == 0:
                #train(glm,trainloader,localupdates)
                train(glm, trainloader, num_rounds*localupdates)
                loss=test(glm,globaltestloader)
            clientloss.append(loss)
        if minimum is None:
            minimum=clientloss
        if maximum is None:
            maximum=clientloss
        else:
            minimum=np.minimum(minimum,clientloss)
            maximum=np.maximum(maximum,clientloss)
        boxplots.append(clientloss)
        #### Local precisions on ax2
        y_pred=glm.model.predict(features)
        q,_,y_pred_seg=aux._mean_frequency_by_risk_group(targets, y_pred, sample_weight=sample_weights, n_bins=10)
        if i== 0:
            ax2.plot(q, y_pred_seg, marker="x", color="grey", linestyle="-.", alpha=.4, label="Local GLM on Data")
        else:
            ax2.plot(q, y_pred_seg, marker="x", color="grey", linestyle="-.", alpha=.4)
    
    ax1.boxplot(np.array(boxplots),positions=np.linspace(0,num_rounds-1,num_rounds),whis=.75,bootstrap=10000,showfliers=False,widths=.35)
    
    del glm
    glm = GLM()
    
    ### Total precision on ax2
    trainloader = load_data(aux.df)
    features,targets,sample_weights=trainloader  
    train(glm, trainloader, num_rounds*localupdates)
    y_pred=glm.model.predict(features)
    _,y_true_seg,_=aux._mean_frequency_by_risk_group(targets, y_pred, sample_weight=sample_weights,n_bins=10)
    ax2.plot(q, y_true_seg, marker="o", color="blue", linestyle="--", alpha=1, label="Total GLM on Data")
    
    ### FL precision on ax2
    server_loss=np.load("./Data/ServerLoss.npy")
    best_round=np.argmin(server_loss)
    print(f"Best round: Round No.{best_round}")
    weightname=f"./Data/round-weights{best_round}.npy"
    interceptname=f"./Data/round-intercept{best_round}.npy"
    coefficients,intercept=np.load(weightname),np.load(interceptname)
    glm.model.named_steps['regressor'].coef_=coefficients
    glm.model.named_steps['regressor'].intercept_=intercept
    y_pred=glm.model.predict(features)
    q,_,y_pred_seg=aux._mean_frequency_by_risk_group(targets, y_pred, sample_weight=sample_weights, n_bins=10)
    plt.plot(q, y_pred_seg, marker="x", color="orange", linestyle="--", alpha=1, label="FL model on Data")
    
    ### Plot Losses on ax1
    x=np.linspace(0,num_rounds-1,num_rounds)
    
    # Plot Client global Losses on ax1
    ax1.fill_between(x,maximum,minimum,color="blue",alpha=.2)
    ax1.plot(minimum,color="blue",alpha=.5,linestyle="--",label="Local model losses Best/Worst")
    ax1.plot(maximum,color="blue",alpha=.5,linestyle="--")
    # Plot Server global Loss
    ax1.plot(server_loss,color="orange",label="FL model loss",linewidth=2)
    
    ax1.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False  # labels along the bottom edge are off
    ) 
    
    ax1.set_ylabel("Mean Poisson Deviance")
    ax1.legend(loc="upper right")
    
    ax2.set_ylabel("Predicted number of claims")
    ax2.set_xlabel("Client percentage")
    ax2.legend(loc="upper left")
    
Plot(clientnbr)
