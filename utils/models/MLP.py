import jax 
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from jaxtyping import PyTree
from typing import Callable, List, Literal

class MLP(eqx.Module):
    layers : List[PyTree]
    #loss_log : List[float]
    activation : jax.nn

    def __init__(self, d_in, d_out, n_hidden, n_neurons, key, activation = jax.nn.tanh): 
        self.layers = []
        layers_size = [d_in] + [n_neurons]*n_hidden + [d_out]
        def init_layer(key, d_in, d_out):
            k1, k2 = jr.split(key)
            glorot_stddev = 1.0 / jnp.sqrt((d_in + d_out) / 2.)
            W = glorot_stddev * jr.normal(k1, (d_in, d_out))
            b = jr.normal(k2, (d_out,))
            return W, b
        key, *subkeys = jr.split(key, len(layers_size))
        self.layers = list(map(init_layer, subkeys, layers_size[:-1], layers_size[1:]))
        self.activation = activation

    @eqx.filter_jit 
    def __call__(self, a):
        for (W, b) in self.layers[:-1]:
            a = self.activation(a@W + b)
        W, b = self.layers[-1]
        a = a@W + b
        if a.shape[0] ==1:
            return a.squeeze(0)
        else:
            return a

## vectorized verison of MLP in 2D
class MLP2D(eqx.Module): 
    R : eqx.nn.MLP

    def __init__(self, 
                 in_channels : int | Literal['scalar'],
                 out_channels : int | Literal['scalar'],
                 n_hidden : int , n_neurons : int, activation : Callable= jax.nn.tanh, *, key
    ):
        key, rkey = jr.split(key)
        self.R = eqx.nn.MLP(
            in_size= in_channels, 
            out_size = out_channels,
            width_size=  n_neurons, 
            depth = n_hidden,
            activation= activation,
            key = rkey,
        )
    
    @eqx.filter_jit
    def __call__(
            self, x
    ): 
        return jax.vmap(
            jax.vmap(self.R, in_axes=(-1), out_axes=(-1)), in_axes=(-1), out_axes=(-1)
            )(x)


class MLP2Dflatten(eqx.Module): 
    N_x : int
    N_y : int
    kernel_size : int
    R : eqx.Module

    def __init__(self, 
                 N_x, 
                 N_y, 
                 kernel_size, 
                 depth : int , width_size : int, activation = jax.nn.tanh, *, key
    ):
        self.N_x = N_x
        self.N_y = N_y
        self.kernel_size = kernel_size
        self.R = eqx.nn.MLP(
            in_size= N_x * N_y, 
            out_size = kernel_size**2,
            width_size=  width_size, 
            depth = depth,
            activation= activation,
            key = key,
        )
    
    def __call__(
            self, x
    ):
        x = x.flatten()

        return self.R(x).reshape(self.kernel_size, self.kernel_size)