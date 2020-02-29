# Drug consumption prediction for NHS

Done as a project at [AIHack 2020](https://aihack.org/) at Imperial College London.

#### Motivation

As life expectancy increases, it is crucial to improve the quality of life among the elderly. As people age, they are more prone to having health issues and degenerative diseases such as cancer, dementia, infections due to viruses and bacteria. Osteoporosis and arthritis affecting the bones and joints, neurodegenerative diseases such as Alzheimer’s and Parkinson's, cancer, and diabetes are among the most common degenerative disease.

Our project is aimed at measuring correlations between different diseases in regions in the UK. We buolt a dataset consisting of time-series data consisting of the number of prescriptions per GP in the UK and we attempt to predict thwat is going to be the monthly consumption for specific medicine per GP.


The main contributions of this projects are:

1. [Dataset](#Dataset)
2. [Method](#Method)
3. [Visualisation](#Visualisation) 

To run our project proceed to [Running](#Running) and the contributions are all summarised in `tutorial.ipynb` inside the root of the project. 

## Dataset

The initial datased consisted of a meta-data file mapping medical procedure codes to descriptions accompanied with a unique CSV for each medical process that contained over ~5 years worth of prescription counts for every GP in the UK that would prescribe that procedure at least once. Our contribution is deemed in processing the dataset and extrapolating non-existent values into a time-series dataset consisting of prescription data from more than ~6000 GPs in the UK for the thirty most-commonly prescribed medical processes over a ~5 year period.   

## Method

Our method is a multi-layered **Recurrent neural network** that is trained through backpropagation. We are able to estimate the uncertainty of our predictions on the procedure counts through Bayesian inference that is provided through using [MC Dropout](https://arxiv.org/pdf/1512.05287.pdf).

## Visualisation



### Running

#### Requirements

The requirements to run the code are in `requirements.txt`. The required libraries are:

```
numpy
ipykernel
scipy
sklearn
seaborn
pytorch
```

To install them create a virtual environemnt for Python>=3.6 with:

```
virtualenv -p python3 venv
source venv/bin/activate 
```

And install the dependecies with: 

```
pip3 install -r requirements.txt
```

### Running 

Running is just as easy. Make sure that you have insalled all the libraries and then the tutorial is in [`tutorial.ipynb`](tutorial.ipynb). You can run it comfortably in a [Jupyter Notebook](https://jupyter.org/) environment with:

```
jupyter notebook
```