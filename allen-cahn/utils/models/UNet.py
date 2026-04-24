import jax 
import jax.numpy as jnp
import equinox as eqx
from typing import Callable, List

class DoubleConv(eqx.Module):
    conv_1: eqx.nn.Conv
    conv_2: eqx.nn.Conv
    activation: Callable

    def __init__(
        self,
        num_spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size : int = 3,
        padding : int = 1, 
        padding_mode = 'CIRCULAR',
        stride : int = 1,
        activation: Callable = jax.nn.tanh, # jax.tree_util.Partial(jax.nn.tanh),
        *,
        key,
    ):
        c_1_key, c_2_key = jax.random.split(key)
        self.conv_1 = eqx.nn.Conv(
            num_spatial_dims,
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            padding_mode=padding_mode,
            stride = stride, 
            key=c_1_key,
        )
        self.conv_2 = eqx.nn.Conv(
            num_spatial_dims,
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            padding_mode=padding_mode,
            stride = stride, 
            key=c_2_key,
        )
        self.activation = activation
    
    def __call__(self, x: jax.Array):
        x = self.conv_1(x)
        x = self.activation(x)
        x = self.conv_2(x)
        x = self.activation(x)
        return x

class ResDoubleConv(eqx.Module):
    conv_1: eqx.nn.Conv
    conv_2: eqx.nn.Conv
    activation: Callable

    def __init__(
        self,
        num_spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size : int = 3,
        padding : int = 1, 
        padding_mode = 'CIRCULAR',
        stride : int = 1,
        activation: Callable = jax.nn.tanh, # jax.tree_util.Partial(jax.nn.tanh),
        *,
        key,
    ):
        c_1_key, c_2_key = jax.random.split(key)
        self.conv_1 = eqx.nn.Conv(
            num_spatial_dims,
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            padding_mode=padding_mode,
            stride = stride, 
            key=c_1_key,
        )
        self.conv_2 = eqx.nn.Conv(
            num_spatial_dims,
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            padding_mode=padding_mode,
            stride = stride, 
            key=c_2_key,
        )
        self.activation = activation
    
    def __call__(self, x: jax.Array):
        x = self.conv_1(x)
        a = self.activation(x)
        x = self.conv_2(a) +x 
        x = self.activation(x)
        return x

class UNet(eqx.Module):
    lifting: DoubleConv
    down_sampling_blocks: list[eqx.nn.Conv]
    left_arc_blocks: list[DoubleConv]
    right_arc_blocks: list[DoubleConv]
    up_sampling_blocks: list[eqx.nn.Conv]
    projection: eqx.nn.Conv
    in_channels: int

    def __init__(
        self,
        num_spatial_dims: int,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        num_levels: int,
        kernel_size : int = 3,
        padding : int = 1, 
        padding_mode = 'CIRCULAR',
        stride : int = 1,
        activation: Callable = jax.nn.tanh,
        *,
        key,
    ):
        self.in_channels = in_channels
        key, lifting_key, projection_key = jax.random.split(key, 3)
        self.lifting = DoubleConv(
            num_spatial_dims,
            in_channels,
            hidden_channels,
            kernel_size = kernel_size, 
            padding=padding, 
            padding_mode=padding_mode, 
            stride=stride, 
            activation=activation,
            key=lifting_key,
        )
        self.projection = eqx.nn.Conv(
            num_spatial_dims,
            hidden_channels,
            out_channels,
            kernel_size=1,
            padding = 0, 
            padding_mode = padding_mode,
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
                    padding_mode = padding_mode, 
                    key=down_key,
                )
            )
            self.left_arc_blocks.append(
                DoubleConv(
                    num_spatial_dims,
                    upper_level_channels,
                    lower_level_channels,
                    kernel_size = kernel_size, 
                    padding=padding, 
                    padding_mode=padding_mode, 
                    stride=stride, 
                    activation=activation,
                    key=left_key,
                )
            )
            self.right_arc_blocks.append(
                DoubleConv(
                    num_spatial_dims,
                    lower_level_channels,
                    upper_level_channels,
                    kernel_size = kernel_size, 
                    padding=padding, 
                    padding_mode=padding_mode, 
                    stride=stride, 
                    activation=activation,
                    key=right_key,
                )
            )
            if padding_mode == 'CIRCULAR':
                self.up_sampling_blocks.append(
                    eqx.nn.ConvTranspose(
                        num_spatial_dims,
                        lower_level_channels,
                        upper_level_channels,
                        kernel_size=3,
                        stride=2,
                        padding='SAME',
                        output_padding=0,
                        padding_mode = padding_mode, 
                        key=up_key,
                    )
                )
            else :
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

        if self.in_channels == 1:
            x = x[jnp.newaxis]
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

        return x.squeeze()

class ResUNet(eqx.Module):
    unet : UNet

    def __init__(
        self,
        num_spatial_dims: int,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        num_levels: int,
        kernel_size : int = 3,
        padding : int = 1, 
        padding_mode = 'CIRCULAR',
        stride : int = 1,
        activation: Callable = jax.nn.tanh,
        *,
        key,
    ):
        self.unet = UNet(
            num_spatial_dims, 
            in_channels, 
            out_channels, hidden_channels, 
            num_levels, 
            kernel_size=kernel_size,
            padding= padding, 
            padding_mode= padding_mode,
            stride = stride, 
            activation=activation, 
            key = key
        )
    
    @eqx.filter_jit
    def __call__(
            self, x
    ): 
        r = self.unet(x)
        if (r.shape == x.shape): 
            return (x + r).squeeze()
        if (r.shape == x.shape[1:]):
            return (r + x[0:1]).squeeze()

class ResUNetTest(eqx.Module):
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
        kernel_size : int = 3,
        padding : int = 1, 
        padding_mode = 'CIRCULAR',
        stride : int = 1,
        activation: Callable = jax.nn.tanh,
        *,
        key,
    ):
        key, lifting_key, projection_key = jax.random.split(key, 3)
        self.lifting = DoubleConv(
            num_spatial_dims,
            in_channels,
            hidden_channels,
            kernel_size = kernel_size, 
            padding=padding, 
            padding_mode=padding_mode, 
            stride=stride, 
            activation=activation,
            key=lifting_key,
        )
        self.projection = eqx.nn.Conv(
            num_spatial_dims,
            hidden_channels,
            out_channels,
            kernel_size=1,
            padding_mode = 'CIRCULAR',
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
                    padding_mode = padding_mode,
                    key=down_key,
                )
            )
            self.left_arc_blocks.append(
                ResDoubleConv(
                    num_spatial_dims,
                    upper_level_channels,
                    lower_level_channels,
                    kernel_size = kernel_size, 
                    padding=padding, 
                    padding_mode=padding_mode, 
                    stride=stride, 
                    activation=activation,
                    key=left_key,
                )
            )
            self.right_arc_blocks.append(
                DoubleConv(
                    num_spatial_dims,
                    lower_level_channels,
                    upper_level_channels,
                    kernel_size = kernel_size, 
                    padding=padding, 
                    padding_mode=padding_mode, 
                    stride=stride, 
                    activation=activation,
                    key=right_key,
                )
            )
            if padding_mode == 'CIRCULAR':
                self.up_sampling_blocks.append(
                    eqx.nn.ConvTranspose(
                        num_spatial_dims,
                        lower_level_channels,
                        upper_level_channels,
                        kernel_size=3,
                        stride=2,
                        padding='SAME',
                        output_padding=0,
                        padding_mode = padding_mode, 
                        key=up_key,
                    )
                )
            else :
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
        x0 = self.lifting(x)
        x = self.lifting(x)
        #print(x.shape, "lifted")
        x_skips = []

        # Left part of the arc
        for down, left in zip(self.down_sampling_blocks, self.left_arc_blocks):
            x_skips.append(x)
            #print(x.shape, "skipped")
            x = down(x)
            #print(x.shape, "down")
            x = left(x) 
            #print(x.shape, "left")

        
        # Right part of the arc
        for right, up in zip(reversed(self.right_arc_blocks), reversed(self.up_sampling_blocks)):
            x = up(x)
            # print(x.shape, "up")
            # Equinox is without batch axis by default, hence channels are at axis 0
            x = jnp.concatenate([x, x_skips.pop()], axis=0)
            #print(x.shape, "concatenated")
            x = right(x)
            #print(x.shape, "right")
        
        x = x + x0 # Res
        x = self.projection(x)
        #print(x.shape, " projected")

        return x.squeeze()