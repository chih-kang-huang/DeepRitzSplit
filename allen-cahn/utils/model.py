import jax 
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from jaxtyping import PyTree
from typing import Callable, List


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


#    def __call__(self, x):
#        a = x
#        for layer in self.layers[:-1]:
#            a = jax.nn.tanh(layer(a))
#        a = self.layers[-1](a)
#
#        return a
        
    @eqx.filter_jit 
    def __call__(self, *x):
        a = jnp.asarray(list(x))
        for (W, b) in self.layers[:-1]:
            #a = jax.nn.tanh(jnp.dot(a, W) + b)
            #a = self.activation(jnp.dot(a, W) + b)
            a = self.activation(a@W + b)
        W, b = self.layers[-1]
        #a = jnp.dot(a, W) + b
        a = a@W + b
        if a.shape[0] ==1:
            return a.squeeze(0)
        else:
            return a

class SpectralConv2d(eqx.Module):
    real_weights: jax.Array
    imag_weights: jax.Array
    in_channels: int
    out_channels: int
    modes1: int
    modes2: int

    def __init__(
            self,
            in_channels,
            out_channels,
            modes1,
            modes2,
            *,
            key,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        scale = 1.0 / (in_channels * out_channels)

        real_key, imag_key = jax.random.split(key)
        self.real_weights = jax.random.uniform(
            real_key,
            (in_channels, out_channels, self.modes1, self.modes2),
            minval=-scale,
            maxval=+scale,
        )
        self.imag_weights = jax.random.uniform(
            imag_key,
            (in_channels, out_channels, self.modes1, self.modes2),
            minval=-scale,
            maxval=+scale,
        )

    def complex_mult2d(
            self,
            x_hat,
            w,
    ):
    # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return jnp.einsum("ixy,ioxy->oxy", x_hat, w)


    def __call__(
            self,
            x
    ):
        channels, spatial_points_0, spatial_points_1 = x.shape

        # shape of x_hat is (in_channels, spatial_points_0, spatial_points_1//2+1)
        x_hat = jnp.fft.rfft2(x)
        # shape of x_hat_under_modes is (in_channels, self.modes1, self.modes2)
        x_hat_under_modes = x_hat[:, :self.modes1, :self.modes2]
        weights = self.real_weights + 1j * self.imag_weights

        # shape of out_hat_under_modes is (out_channels, self.modes)
        out_hat_under_modes = self.complex_mult2d(x_hat_under_modes, weights)

        # shape of out_hat is (out_channels, spatial_points_0, spatial_points_1//2+1)
        out_hat = jnp.zeros(
            (self.out_channels, x_hat.shape[-2], x_hat.shape[-1]),
            dtype=x_hat.dtype
        )
        out_hat = out_hat.at[:, :self.modes1, :self.modes2].set(out_hat_under_modes)

        out = jnp.fft.irfft2(out_hat, s=(x.shape[-2], x.shape[-1]))

        return out

class FNOBlock2d(eqx.Module):
    spectral_conv: SpectralConv2d
    bypass_conv: eqx.nn.Conv1d
    activation: Callable

    def __init__(
            self,
            in_channels,
            out_channels,
            modes1,
            modes2,
            activation=jax.nn.tanh,
            *,
            key,
    ):
        spectral_conv_key, bypass_key = jax.random.split(key)
        self.spectral_conv = SpectralConv2d(
            in_channels,
            out_channels,
            modes1,
            modes2,
            key=spectral_conv_key,
        )
        self.bypass_conv = eqx.nn.Conv2d(
            in_channels,
            out_channels,
            1,  # Kernel size is one
            key=bypass_key,
        )

        self.activation = activation

    def __call__(
            self,
            x,
    ):
        return self.activation(
            self.spectral_conv(x) + self.bypass_conv(x)
        )
class FNO2d(eqx.Module):
    lifting: eqx.nn.Conv2d
    fno_blocks: List[FNOBlock2d]
    projection1: eqx.nn.Conv2d
    projection2: eqx.nn.Conv2d
    in_channels: int
    out_channels: int
    modes1: int
    modes2: int
    width: int
    n_blocks: int
    activation: Callable
    bias1: jax.Array
    bias2: jax.Array

    def __init__(
            self,
            in_channels,
            out_channels,
            modes1,
            modes2,
            width,
            activation=jax.nn.tanh,
            n_blocks = 4,
            *,
            key,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.n_blocks = n_blocks
        self.activation = activation
        key, lifting_key = jax.random.split(key)
        self.lifting = eqx.nn.Conv2d(
            in_channels,
            width,
            1,
            key=lifting_key,
        )

        self.fno_blocks = []
        for i in range(n_blocks):
            key, subkey = jax.random.split(key)
            if i != n_blocks - 1:
                self.fno_blocks.append(FNOBlock2d(
                    width,
                    width,
                    modes1,
                    modes2,
                    activation,
                    key=subkey,
                ))
            else: 
                self.fno_blocks.append(FNOBlock2d(
                    width,
                    width,
                    modes1,
                    modes2,
                    activation = lambda x: x,
                    key=subkey,
                ))

        key, projection1_key, projection2_key = jax.random.split(key, 3)
        self.projection1 = eqx.nn.Conv2d(
            width,
            width*4,
            1,
            key=projection1_key,
        )
        self.projection2 = eqx.nn.Conv2d(
            width*4,
            out_channels,
            1,
            key=projection2_key,
        )
        key, b1key, b2key = jax.random.split(key, 3)
        self.bias1 = jr.normal(b1key)
        self.bias2 = jr.normal(b2key)

    @eqx.filter_jit
    def __call__(
            self,
            x,
    ):
        assert(x.shape[0] == self.in_channels)

        x = self.lifting(x)

        for fno_block in self.fno_blocks:
            x = fno_block(x)

        x = self.projection1(x) + self.bias1
        x = self.activation(x) 
        x = self.projection2(x) + self.bias2

        return x if self.out_channels != 1 else x.squeeze(0)




# class ReactionBlock(eqx.Module):
#     activation: Callable
#     bias1: jax.Array
#     bias2: jax.Array
#     projection1: eqx.nn.Conv2d
#     projection2: eqx.nn.Conv2d
# 
#     def __init__(
#             self, in_channels, out_channels, width, activation, *, key
#     ): 
# 
#         key, projection1_key, projection2_key = jax.random.split(key, 3)
#         self.projection1 = eqx.nn.Conv2d(
#             in_channels,
#             width,
#             1,
#             key=projection1_key,
#         )
# 
#         key, b1key, b2key = jax.random.split(key, 3)
# 
#         self.bias1 = jr.normal(b1key, shape=(width, 1, 1))
# 
#         self.projection2 = eqx.nn.Conv2d(
#             width,
#             out_channels,
#             1,
#             key=projection2_key,
#         )
# 
#         self.bias2 = jr.normal(b2key, shape=(out_channels, 1, 1))
#         self.activation = activation 
# 
#     def __call__(
#             self, 
#             x
#     ):
#         x = self.projection1(x) + self.bias1 
#         x = self.activation(x)
#         x = self.projection2(x) + self.bias2
#         return x 

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

    
class DRBlock(eqx.Module): 
    R: ReactionBlock
    #D : SpectralConv2d
    D : eqx.nn.Conv2d

    # projection: eqx.nn.Conv2d
    # width: int
    # activation: Callable

    def __init__(
            self, 
            in_channels, 
            out_channels, 
            width, 
            kernel_size,
            n_layers,
            activation, 
            *, key
    ): 
        key, Rkey = jax.random.split(key)
        self.R = ReactionBlock(
            in_channels= in_channels,
            out_channels=width,
            width=4*width,
            n_layers = n_layers,
            activation = activation,
            key=Rkey,
        )

        key, Dkey = jr.split(key)

        #self.D = SpectralConv2d(
        #    in_channels=width,
        #    out_channels=out_channels, 
        #    modes1 = modes1, 
        #    modes2 = modes2, 
        #    key = Dkey
        #)
        
        self.D = eqx.nn.Conv2d( 
            width,
            out_channels,
            kernel_size = kernel_size,
            padding = (kernel_size -1) // 2, 
            padding_mode = "CIRCULAR",
            key=Dkey,
        )


    @eqx.filter_jit    
    def __call__(self, 
                 x,
                 ): 

        x = self.R(x)
        x = self.D(x)
        return x

class DR1(eqx.Module):
    lifting: eqx.nn.Conv2d
    DR : DRBlock

    projection: eqx.nn.Conv2d
    bias : jax.Array

    def __init__(
            self, 
            in_channels, 
            out_channels, 
            width, 
            kernel_size, 
            n_layers,
            activation = jax.nn.tanh, 
            *, 
            key
    ): 
        key, lifting_key = jax.random.split(key)
        self.lifting = eqx.nn.Conv2d(
            in_channels,
            width,
            kernel_size = 1,
            key=lifting_key,
        )

        #key, Rkey = jax.random.split(key)
        #self.R = ReactionBlock(
        #    in_channels= width,
        #    out_channels=width,
        #    width=4*width,
        #    activation = activation,
        #    key=Rkey,
        #)

        #key, Dkey = jr.split(key)

        #self.D = eqx.nn.Conv2d( 
        #    width,
        #    width,
        #    kernel_size = kernel_size,
        #    padding = (kernel_size -1) // 2, 
        #    padding_mode = "CIRCULAR",
        #    key=Dkey,
        #)
        key, DRkey = jr.split(key)
        self.DR = DRBlock(
            width, 
            width, 
            width, 
            kernel_size, 
            n_layers = n_layers,
            activation=activation,
            key=DRkey
        )
        
        key, projection_key = jax.random.split(key, 2)

        self.projection = eqx.nn.Conv2d(
            width,
            out_channels,
            1,
            key=projection_key,
        )
        key, bkey = jr.split(key)
        self.bias = jr.normal(bkey)

    
    @eqx.filter_jit    
    def __call__(self, 
                 *x,
                 ): 
        x = jnp.stack(x, axis=0)

        x = self.lifting(x)

        x = self.DR(x)

        x = self.projection(x) + self.bias

        return x.squeeze(0)

# class DR2(eqx.Module):
#     lifting: eqx.nn.Conv2d
#     DR1 : DRBlock
#     DR2 : DRBlock
#     R: ReactionBlock
# 
#     projection: eqx.nn.Conv2d
#     bias : jax.Array
# 
# 
#     def __init__(
#             self, 
#             in_channels, 
#             out_channels, 
#             width, 
#             kernel_size, 
#             activation = jax.nn.tanh, 
#             *, 
#             key
#     ): 
#         key, lifting_key = jax.random.split(key)
#         self.lifting = eqx.nn.Conv2d(
#             in_channels,
#             width,
#             kernel_size = 1,
#             key=lifting_key,
#         )
# 
#         key, DR1key, DR2key = jr.split(key, 3)
#         self.DR1 = DRBlock(
#             width, 
#             width, 
#             width, 
#             kernel_size, 
#             activation=activation,
#             key=DR1key
#         )
# 
#         self.DR2 = DRBlock(
#             width, 
#             width, 
#             width, 
#             kernel_size, 
#             activation=activation,
#             key=DR2key
#         )
#         
#         key, Rkey = jr.split(key)
# 
#         self.R = ReactionBlock(
#             in_channels= width,
#             out_channels= width, 
#             width = width,
#             activation= activation,
#             key = Rkey
#         )
# 
# 
#         key, projection_key = jax.random.split(key, 2)
# 
#         self.projection = eqx.nn.Conv2d(
#             width,
#             out_channels,
#             1,
#             key=projection_key,
#         )
#         key, bkey = jr.split(key)
#         self.bias = jr.normal(bkey)
# 
#     @eqx.filter_jit    
#     def __call__(self, 
#                  *x,
#                  ): 
#         x = jnp.stack(x, axis=0)
# 
#         x = self.lifting(x)
# 
#         x1 = self.DR1(x)
#         x2 = x + self.R(x)
# 
#         x = self.DR2(x1+x2)
# 
#         x = self.projection(x) + self.bias
# 
#         return x.squeeze(0)
class DR2(eqx.Module):
   lifting: eqx.nn.Conv2d
   DR1 : DRBlock
   DR2 : DRBlock
   R: ReactionBlock

   projection: eqx.nn.Conv2d
   bias : jax.Array
   projection1: eqx.nn.Conv2d
   bias1 : jax.Array


   def __init__(
           self, 
           in_channels, 
           out_channels, 
           width, 
           kernel_size, 
           n_layers, 
           activation = jax.nn.tanh, 
           *, 
           key
   ): 
       key, lifting_key = jax.random.split(key)
       self.lifting = eqx.nn.Conv2d(
           in_channels,
           width,
           kernel_size = 1,
           key=lifting_key,
       )

       key, DR1key, DR2key = jr.split(key, 3)
       self.DR1 = DRBlock(
           width, 
           width, 
           width, 
           kernel_size, 
           n_layers= n_layers,
           activation=activation,
           key=DR1key
       )

       self.DR2 = DRBlock(
           width, 
           width, 
           width, 
           kernel_size, 
           n_layers= n_layers,
           activation=activation,
           key=DR2key
       )
       
       key, Rkey = jr.split(key)

       self.R = ReactionBlock(
           in_channels= width,
           out_channels= width, 
           width = 4*width,
           n_layers= n_layers,
           activation= activation,
           key = Rkey
       )


       key, projection_key = jax.random.split(key, 2)

       self.projection = eqx.nn.Conv2d(
           width,
           out_channels,
           1,
           key=projection_key,
       )
       key, bkey = jr.split(key)
       self.bias = jr.normal(bkey)

       key, p1key = jax.random.split(key, 2)

       self.projection1 = eqx.nn.Conv2d(
           width,
           out_channels,
           1,
           key=p1key,
       )
       key, b1key = jr.split(key)
       self.bias1 = jr.normal(b1key)

   @eqx.filter_jit 
   def __call__(self, 
                *x,
                ): 
       x = jnp.stack(x, axis=0)

       x = self.lifting(x)

       x1 = self.DR1(x)
       x2 = x + self.R(x)

       x = self.DR2(x1+x2)

       x = self.projection(x) + self.bias

       x1 = self.projection1(x1) + self.bias1

       return x1.squeeze(0), x.squeeze(0)

class DR1Block(eqx.Module):
    lifting: eqx.nn.Conv2d
    DR : DRBlock

    projection: eqx.nn.Conv2d
    bias : jax.Array

    def __init__(
            self, 
            in_channels, 
            out_channels, 
            width, 
            kernel_size, 
            n_layers,
            activation = jax.nn.tanh, 
            *, 
            key
    ): 
        key, lifting_key = jax.random.split(key)
        self.lifting = eqx.nn.Conv2d(
            in_channels,
            width,
            kernel_size = 1,
            key=lifting_key,
        )

        #key, Rkey = jax.random.split(key)
        #self.R = ReactionBlock(
        #    in_channels= width,
        #    out_channels=width,
        #    width=4*width,
        #    activation = activation,
        #    key=Rkey,
        #)

        #key, Dkey = jr.split(key)

        #self.D = eqx.nn.Conv2d( 
        #    width,
        #    width,
        #    kernel_size = kernel_size,
        #    padding = (kernel_size -1) // 2, 
        #    padding_mode = "CIRCULAR",
        #    key=Dkey,
        #)
        key, DRkey = jr.split(key)
        self.DR = DRBlock(
            width, 
            width, 
            width, 
            kernel_size, 
            n_layers = n_layers,
            activation=activation,
            key=DRkey
        )
        
        key, projection_key = jax.random.split(key, 2)

        self.projection = eqx.nn.Conv2d(
            width,
            out_channels,
            1,
            key=projection_key,
        )
        key, bkey = jr.split(key)
        self.bias = jr.normal(bkey)

    
    @eqx.filter_jit    
    def __call__(self, 
                 x,
                 ): 
        if len(x.shape) == 2: 
            x = x[jnp.newaxis, :, :]

        x = self.lifting(x)

        x = self.DR(x)

        x = self.projection(x) + self.bias

        return x.squeeze(0) if x.shape[0] == 1 else x

class DoubleConv(eqx.Module):
    conv_1: eqx.nn.Conv
    conv_2: eqx.nn.Conv
    activation: Callable

    def __init__(
        self,
        num_spatial_dims: int,
        in_channels: int,
        out_channels: int,
        activation: Callable,
        *,
        key,
    ):
        c_1_key, c_2_key = jax.random.split(key)
        self.conv_1 = eqx.nn.Conv(
            num_spatial_dims,
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            key=c_1_key,
        )
        self.conv_2 = eqx.nn.Conv(
            num_spatial_dims,
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            key=c_2_key,
        )
        self.activation = activation
    
    def __call__(self, x: jax.Array):
        x = self.conv_1(x)
        x = self.activation(x)
        x = self.conv_2(x)
        x = self.activation(x)
        return x
class UNet(eqx.Module):
    lifting: DoubleConv
    down_sampling_blocks: list[eqx.nn.Conv]
    left_arc_blocks: list[DoubleConv]
    right_arc_blocks: list[DoubleConv]
    up_sampling_blocks: list[eqx.nn.Conv]
    projection: eqx.nn.Conv

    def __init__(
        self,
        num_spatial_dims: int,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        num_levels: int,
        activation: Callable,
        *,
        key,
    ):
        key, lifting_key, projection_key = jax.random.split(key, 3)
        self.lifting = DoubleConv(
            num_spatial_dims,
            in_channels,
            hidden_channels,
            activation,
            key=lifting_key,
        )
        self.projection = eqx.nn.Conv(
            num_spatial_dims,
            hidden_channels,
            out_channels,
            kernel_size=1,
            key=projection_key,
        )

        channel_list = [hidden_channels * 2**i for i in range(0, num_levels+1)]

        self.down_sampling_blocks = []
        self.left_arc_blocks = []
        self.right_arc_blocks = []
        self.up_sampling_blocks = []

        for (upper_level_channels, lower_level_channels) in zip(
            channel_list[:-1], channel_list[1:]
        ):
            key, down_key, left_key, right_key, up_key = jax.random.split(key, 5)

            self.down_sampling_blocks.append(
                eqx.nn.Conv(
                    num_spatial_dims,
                    upper_level_channels,
                    upper_level_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    key=down_key,
                )
            )
            self.left_arc_blocks.append(
                DoubleConv(
                    num_spatial_dims,
                    upper_level_channels,
                    lower_level_channels,
                    activation,
                    key=left_key,
                )
            )
            self.right_arc_blocks.append(
                DoubleConv(
                    num_spatial_dims,
                    lower_level_channels,
                    upper_level_channels,
                    activation,
                    key=right_key,
                )
            )
            self.up_sampling_blocks.append(
                eqx.nn.ConvTranspose(
                    num_spatial_dims,
                    lower_level_channels,
                    upper_level_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                    key=up_key,
                )
            )

    def __call__(self, x: jax.Array):

        if len(x.shape) == 2: 
            x = x[jnp.newaxis, :, :]
        x = self.lifting(x)
        x_skips = []

        # Left part of the arc
        for down, left in zip(self.down_sampling_blocks, self.left_arc_blocks):
            x_skips.append(x)
            x = down(x)
            x = left(x)
        
        # Right part of the arc
        for right, up in zip(reversed(self.right_arc_blocks), reversed(self.up_sampling_blocks)):
            x = up(x)
            # Equinox is without batch axis by default, hence channels are at axis 0
            x = jnp.concatenate([x, x_skips.pop()], axis=0)
            x = right(x)
        
        x = self.projection(x)

        return x.squeeze(0) if x.shape[0] == 1 else x