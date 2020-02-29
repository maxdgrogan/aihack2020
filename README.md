# Drug predcition analysis for NHS

Done as a project at [AIHack 2020](https://aihack.org/) at Imperial College London.

![Image](schedule.png?raw=true "schedule")


### Requirements

The requirements to run the code are in `requirements.txt`. The required libraries are:

```
numpy
gpflow<=1.5.1
tensorflow<2.0.0
ipykernel
scipy
sklearn
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