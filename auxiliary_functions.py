# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 13:15:52 2022

@author: arno.geimer
"""

import warnings
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)

### HYPERPARAMETERS

localupdates=25
num_rounds=75
bins=5

# Get the Data, transform it
from sklearn.datasets import fetch_openml

def get_categories(datacolumn):
    return list(set(datacolumn))

df=fetch_openml(data_id=41214, as_frame=True).frame
df["Frequency"] = df["ClaimNb"] / df["Exposure"]

Regions=get_categories(df["Region"])
Areas=get_categories(df["Area"])
VehPowers=get_categories(df["VehPower"])
VehBrands=get_categories(df["VehBrand"])
VehGas=get_categories(df["VehGas"])

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from sklearn.compose import ColumnTransformer

log_scale_transformer = make_pipeline(
    FunctionTransformer(np.log, validate=False), StandardScaler()
)

linear_model_preprocessor = ColumnTransformer(
    [
        ("passthrough_numeric", "passthrough", ["BonusMalus"]),
        ("binned_numeric", KBinsDiscretizer(n_bins=bins), ["DrivAge","VehAge"]),
        ("log_scaled_numeric", log_scale_transformer, ["Density"]),
        ("onehot_categorical", OneHotEncoder(categories=[VehBrands,VehPowers,VehGas,Areas]),
            ["VehBrand", "VehPower", "VehGas", "Area"],
        ),
    ],
    remainder="drop",
)

from sklearn.utils import gen_even_slices

def _mean_frequency_by_risk_group(y_true, y_pred, sample_weight=None, n_bins=100):
    
    idx_sort = np.argsort(y_pred)
    bin_centers = np.arange(0, 1, 1 / n_bins) + 0.5 / n_bins
    y_pred_bin = np.zeros(n_bins)
    y_true_bin = np.zeros(n_bins)

    for n, sl in enumerate(gen_even_slices(len(y_true), n_bins)):
        weights = sample_weight[idx_sort][sl]
        y_pred_bin[n] = np.average(y_pred[idx_sort][sl], weights=weights)
        y_true_bin[n] = np.average(y_true[idx_sort][sl], weights=weights)
    return bin_centers, y_true_bin, y_pred_bin