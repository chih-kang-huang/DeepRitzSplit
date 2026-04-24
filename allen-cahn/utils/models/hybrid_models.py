import jax 
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from jaxtyping import PyTree
from typing import Callable, List
from jax.tree_util import tree_map


#class aniso(eqx.Module): 
#    lifting: eqx.nn.Conv2d
#    lifting_phi: eqx.nn.Conv2d
#    down_sampling_blocks: list[eqx.nn.Conv2d]
#    left_arc_blocks: list[DoubleConv2d]
#    right_arc_blocks: list[DoubleConv2d]
#    R1: ReactionBlock
#    projection: eqx.nn.Conv2d
#
#    kernel_size : int
#    bias : jax.Array
#    down_sampling_blocks: list[eqx.nn.Conv2d]
#    left_arc_blocks: list[DoubleConv2d]
#    right_arc_blocks: list[DoubleConv2d]
#    R1: ReactionBlock
#    projection: eqx.nn.Conv2d
#
#    kernel_size : int
#    bias : jax.Array
#    activation: Callable
#
#    def __init__(
#            self, 
#            in_channels : int, 
#            out_channels : int,
#            hidden_channels :int,
#            num_levels :int,
#            width :int, 
#            kernel_size : int, 
#            padding : int,
#            n_layers : int,
#            activation : Callable, 
#        #    N_x, N_y,
#            *, key
#    ): 
#        self.kernel_size = kernel_size
#        self.activation = activation
#
#        key, lifting_key = jax.random.split(key)
#        self.lifting = eqx.nn.Conv2d(
#            in_channels,
#            width,
#            kernel_size = 1,
#            key=lifting_key,
#        )
#        key, lifting_key2 = jax.random.split(key)
#        self.lifting_phi = eqx.nn.Conv2d(
#            1,
#            width,
#            kernel_size = 1,
#            key=lifting_key2,
#        )
#        key, R1key = jax.random.split(key)
#        self.R1 = ReactionBlock(
#            in_channels=  width,
#            out_channels= width,
#            width=4*width,
#            n_layers = n_layers,
#            activation = activation,
#            key=R1key,
#        )
#
#        key, bkey = jr.split(key)
#        self.bias = jr.normal(bkey)
#
#
#        key, projection_key = jax.random.split(key, 2)
#
#        self.projection = eqx.nn.Conv2d(
#            width,
#            out_channels,
#            kernel_size = 1,
#            key=projection_key,
#        )
#
#
#        channel_list = [width] + [hidden_channels * 2**i for i in range(0, num_levels+1)]
#        channel_list.append(width)
#
#        self.down_sampling_blocks = []
#        self.left_arc_blocks = []
#        self.right_arc_blocks = []
#
#        for (upper_level_channels, lower_level_channels) in zip(
#            channel_list[:-1], channel_list[1:]
#        ):
#            key, down_key, left_key, right_key = jax.random.split(key, 4)
#
#            self.down_sampling_blocks.append(
#                eqx.nn.Conv2d(
#                    upper_level_channels,
#                    upper_level_channels,
#                    kernel_size=kernel_size,
#                    padding=padding,
#                    padding_mode = "CIRCULAR",
#                    key=down_key,
#                )
#            )
#            self.left_arc_blocks.append(
#                DoubleConv2d(
#                    upper_level_channels,
#                    lower_level_channels,
#                    kernel_size= kernel_size, 
#                    padding=padding,
#                    activation = activation,
#                    key=left_key,
#                )
#            )
#            self.right_arc_blocks.append(
#                DoubleConv2d(
#                    lower_level_channels,
#                    upper_level_channels,
#                    kernel_size= kernel_size, 
#                    padding=padding,
#                    activation = activation,
#                    key=right_key,
#                )
#            )
#
#    @eqx.filter_jit
#    def __call__(self, 
#                 phi_u
#                 ): 
#
#        phi = phi_u[0:1]
#        phi = self.lifting_phi(phi)
#        # phi_skips = []
#
#        rho = self.lifting(phi_u)
#        rho = self.R1(rho) + self.bias
#
#        rho = self.projection(phi_u)
#
#
#        # Left part of the arc
#        for down, left in zip(self.down_sampling_blocks, self.left_arc_blocks):
#            # phi_skips.append(phi)
#            phi = down(phi)
#            phi = left(phi)
#        
#        # # Right part of the arc
#        # for right, up in zip(reversed(self.right_arc_blocks), reversed(self.up_sampling_blocks)):
#        #     phi = up(phi)
#        #     # Equinox is without batch axis by default, hence channels are at axis 0
#        #     phi = jnp.concatenate([phi, phi_skips.pop()], axis=0)
#        #     phi = right(phi)
#        phi = self.projection(phi).squeeze(0)
#
#        # D = eqx.nn.Conv2d(1, 1, phi.shape[0], 
#        #                   #padding = phi.shape[0]//2,
#        #                   padding = 'SAME',
#        #                   padding_mode = 'CIRCULAR', 
#        #                   key = jax.random.PRNGKey(0))
#        # D = eqx.tree_at(lambda t: t.weight, D, phi[jnp.newaxis, jnp.newaxis])
#        # print(phi.shape, rho.shape)
#        return convolve_wrap(rho.squeeze(0), phi)
#        # return D(rho).squeeze(0)
#    def shape(self, 
#                phi_u
#                 ): 
#        phi = phi_u[0:1]
#        phi = self.lifting_phi(phi)
#        # phi_skips = []
#
#        rho = self.lifting(phi_u)
#        rho = self.R1(rho) + self.bias
#
#        rho = self.projection(phi_u)
#
#
#        # Left part of the arc
#        for down, left in zip(self.down_sampling_blocks, self.left_arc_blocks):
#            # phi_skips.append(phi)
#            phi = down(phi)
#            phi = left(phi)
#        
#        # # Right part of the arc
#        # for right, up in zip(reversed(self.right_arc_blocks), reversed(self.up_sampling_blocks)):
#        #     phi = up(phi)
#        #     # Equinox is without batch axis by default, hence channels are at axis 0
#        #     phi = jnp.concatenate([phi, phi_skips.pop()], axis=0)
#        #     phi = right(phi)
#        phi = self.projection(phi).squeeze(0)
#
#        # D = eqx.nn.Conv2d(1, 1, phi.shape[0], 
#        #                   #padding = phi.shape[0]//2,
#        #                   padding = 'SAME',
#        #                   padding_mode = 'CIRCULAR', 
#        #                   key = jax.random.PRNGKey(0))
#        # D = eqx.tree_at(lambda t: t.weight, D, phi[jnp.newaxis, jnp.newaxis])
#        print(phi.shape, rho.shape)
#        # return convolve_wrap(rho.squeeze(0), phi)
#        # return D(rho).squeeze(0)



class ReactionDiffusion2D(eqx.Module):
    R: eqx.Module
    D: eqx.Module

    def __init__(
            self, 
            React, 
            Diffuse, 
    ):
        self.R = React
        self.D = Diffuse

    def convolve_wrap(self, a1,a2
    ):

        N2 = a2.shape[0]
        pad_size = N2 // 2
        a1 = jnp.pad(a1,pad_size,mode='wrap')

        return jax.scipy.signal.convolve2d(a1, a2,mode='same')[pad_size:-pad_size,pad_size:-pad_size]

    @eqx.filter_jit
    def __call__(
            self, x
    ):
        react = self.R(x).squeeze()
        #if react.shape[0] == 1:
        #    react = react.squeeze(0)

        phi = x[0:1]
        diffuse = self.D(phi).squeeze()
        #if diffuse.shape[0] == 1:
        #    diffuse  = diffuse.squeeze(0)

        return self.convolve_wrap(react, diffuse)
        # return jnp.fft.irfft2(
        #            jnp.fft.rfft2(diffuse) * jnp.fft.rfft2(react)
        #            , s=(react.shape[-2], react.shape[-1])
        #        )