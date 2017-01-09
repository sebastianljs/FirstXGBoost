# FirstXGBoost

## Summary  

This repository contains my initial experiments in [Python's XGBoost library](https://github.com/dmlc/xgboost/tree/master/python-package). 

The following datasets from UCI's Machine Learning Repository are used 

+ [Iris dataset](https://archive.ics.uci.edu/ml/datasets/Iris)
+ [Breast cancer dataset](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer)
+ [Horse-colic dataset](https://archive.ics.uci.edu/ml/datasets/Horse+Colic)

## Problem Overview 

### Iris Dataset
### Breast Cancer Dataset 
### Horse-colic Dataset

## Algorithm Overview

The algorithm 

## Hacks I have discovered   

### Automatically generating python script and HTML whenever a Jupyter Notebook is saved 

I found this excellent [article](http://svds.com/jupyter-notebook-best-practices-for-data-science/) which configures Jupyter notebook in a way that it generates a corresponding python script and HTML. This greatly simplifies version control 

### Modifying python.tpl in order to remove execution count when generating python script from Jupyter

I modified the [python.tpl](https://github.com/jupyter/nbconvert/blob/master/nbconvert/templates/python.tpl) file by changing the block prompt format to 
```{tpl}
{% block in_prompt -%}
{% endblock in_prompt %}
```
This removes the execution count `In[]` which by default appears as comments in the python script generated from Jupyter notebooks, which reduces clutter. 
