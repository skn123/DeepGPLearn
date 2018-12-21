# Sampling for the hidden states of a two stage GP

We have a model like this

- z ~ GP(x) 
- y ~ GP(z)

I/O, x and y are observed, but z are hidden form the model. 
In the real problem, the kernels are not known and need to be optimized for. 
Here we just assume we have knowledge of that thus focus on the sample of the states. 
For that reason, the training data are sampled from a known kernel.

