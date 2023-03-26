# training adopted [palette repo](https://github.com/LouisRouss/Diffusion-Based-Model-for-Colorization) using discrete optimal transport from [POT](https://pythonot.github.io)

Expected to have torch, torchvision, kaggle, numpy, matplotlib

To select dataset and regularization type one can change two last lines in conf.yml
Also you can change the mode: 1 - for training, 2 - for metrics

To run code you can run 
~~~
!python3 main.py
~~~