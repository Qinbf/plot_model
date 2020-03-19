## plot_model

GitHub:[https://github.com/Qinbf/plot_model.git](https://github.com/Qinbf/plot_model.git)

plot_model is a API for model visualization reference to [tensorflow.keras.utils.plot_model](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/utils/vis_utils.py).

 <p align="center"><img  src="https://raw.githubusercontent.com/Qinbf/plot_model/master/img/1.png" width="50%" height="50%"></p>
 
 ------------------
 
 ## Installation

Before installing plot_model, please install one of its engines: **TensorFlow**, **Keras**. 


You may also consider installing the following :

```sh
pip install pydot
pip install pydot_ng
pip install graphviz
```
- Finally,download and install [graphviz](https://graphviz.gitlab.io/download/) to plot model graphs.

Then, you can install plot_model itself. There are two ways to install plot_model:

- **Install plot_model from PyPI (recommended):**

Note: These installation steps assume that you are on a Linux or Mac environment.
If you are on Windows, you will need to remove `sudo` to run the commands below.

```sh
sudo pip install plot_model
```

If you are using a virtualenv, you may want to avoid using sudo:

```sh
pip install plot_model
```

- **Alternatively: install plot_model from the GitHub source:**

First, clone plot_model using `git`:

```sh
git clone https://github.com/Qinbf/plot_model.git
```

 Then, `cd` to the plot_model folder and run the install command:
```sh
cd plot_model
sudo python setup.py install
```

------------------

## Getting started

API is similar to tensorflow.keras.utils.plot_model

```python
from plot_model import plot_model
plot_model(model)
```

default parameters:
```python
plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=False, rankdir='TB', expand_nested=False, style=0, color=True, dpi=96)
```

**color**: whether to display color. Default True.
 
**style**: 0 new style. 1 original style. Default 0. 

------------------

## Examples
<h5 align="center">ResNet</h5>
<p align="center"><img  src="https://raw.githubusercontent.com/Qinbf/plot_model/master/img/2.png" width="25%" height="25%"></p>

 
<h5 align="center">Inception</h5>
<p align="center"><img  src="https://raw.githubusercontent.com/Qinbf/plot_model/master/img/3.png" width="50%" height="50%"></p>
 
 
<h5 align="center">ResNeXt</h5>
<p align="center"><img  src="https://raw.githubusercontent.com/Qinbf/plot_model/master/img/4.png" width="65%" height="65%"></p>


