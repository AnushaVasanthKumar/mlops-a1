\# ML Ops Assignment 1 — Boston Housing



Baseline regression pipelines on the Boston Housing dataset with:

\- \*\*DecisionTreeRegressor\*\* (`train.py`)

\- \*\*KernelRidge (RBF)\*\* (`train2.py`)



A shared `misc.py` handles data loading, preprocessing, and evaluation.  

A GitHub Actions workflow runs both models on pushes to the \*\*`kernelridge`\*\* branch and prints MSEs in the logs.



