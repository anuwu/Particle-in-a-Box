import autograd.numpy as np
from autograd import grad, elementwise_grad
import autograd.numpy.random as npr
from autograd.misc.optimizers import adam

def init_random_params(scale, layer_sizes, rs=npr.RandomState(42)):
    """Build a list of (weights, biases) tuples, one for each layer."""
    return [(rs.randn(insize, outsize) * scale,   # weight matrix
             rs.randn(outsize) * scale)           # bias vector
            for insize, outsize in zip(layer_sizes[:-1], layer_sizes[1:])]

def swish(x):
    "see https://arxiv.org/pdf/1710.05941.pdf"
    return x / (1.0 + np.exp(-x))

def psi(nnparams, inputs):
    "Neural network wavefunction"
    for W, b in nnparams:
        outputs = np.dot(inputs, W) + b
        inputs = swish(outputs)    
    return outputs

psip = elementwise_grad(psi, 1) # dpsi/dx 
psipp = elementwise_grad(psip, 1) # d^2psi/dx^2