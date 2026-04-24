import jax 
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from jaxtyping import PyTree
from typing import Callable, List
from jax.tree_util import tree_map

class ResidualBlock(eqx.Module): 
    NNBlock: eqx.Module

    def __init__(self, NNBlock): 
        self.NNBlock = NNBlock
    
    @eqx.filter_jit
    def __call__(self, x): 
        r = self.NNBlock(x)
        # assert(x.shape == r.shape)
        return (x + r).squeeze() if (r.shape == x.shape) else (r + x[0:1]).squeeze()

class ReactionBlock(eqx.Module):
    activation: Callable
    proj_list : List[eqx.nn.Conv2d]
    bias_list : List[eqx.nn.Conv2d]

    def __init__(
            self, in_channels, out_channels, width, n_layers, activation,*, key
    ): 
        layers_list = [in_channels]  + [width] * n_layers + [out_channels]
        self.proj_list = list()
        self.bias_list = list()

        for in_c, out_c in zip(layers_list[:-1], layers_list[1:]):
            key, pkey = jax.random.split(key, 2)
            self.proj_list.append(
                eqx.nn.Conv2d( 
                    in_c, 
                    out_c, 
                    1, 
                    key=pkey
                )
            )
            key, bkey = jr.split(key)
            self.bias_list.append( 
                jr.normal(bkey, shape=(out_c, 1, 1))
            )


        self.activation = activation 

    def __call__(
            self, 
            x
    ):
        for (projection, bias) in zip(self.proj_list[:-1], self.bias_list[:-1]):
            x = projection(x) + bias 
            x = self.activation(x)
        
        projection, bias = self.proj_list[-1], self.bias_list[-1]
        x =  projection(x) + bias

        return x 



class NNsequences(eqx.Module):
    D : List[eqx.Module]

    def __init__(
            self, 
            D,
    ): 
        self.D = D
    
    def __call__(
            self, x
    ): 
        for block in self.D:
            x = block(x)
        
        return x