import jax 
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from jaxtyping import PyTree
from typing import Callable, List
from jax.tree_util import tree_map



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
            padding_mode = 'CIRCULAR',
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
            padding_mode = padding_mode,
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

        return x.squeeze(0) if x.shape[0] == 1 else x

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
            key,
    ): 
        key, lifting_key = jax.random.split(key)
        self.lifting = eqx.nn.Conv2d(
            in_channels,
            width,
            kernel_size = 1,
            key=lifting_key,
        )

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

        x = self.lifting(x)
        x = self.DR(x)
        x = self.projection(x) + self.bias

        return x.squeeze(0) if x.shape[0] == 1 else x

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



class nnSAV(eqx.Module):
    iter : int
    DR : DR1Block
    # aux : DR1Block
    # E : DR1Block
    # eta_zeta : eqx.nn.MLP
    dt : float
    dE_t : Callable
    energy : Callable


    def __init__(
            self, 
            in_channels, 
            out_channels, 
            width, 
            kernel_size, 
            n_layers,
            dt,
            energy,
            dE_t,
            iter = 1,
            activation = jax.tree_util.Partial(jax.nn.tanh) , 
            *, 
            key
    ): 
        self.energy = energy
        self.dE_t = dE_t
        self.dt = dt

        key, DRkey = jax.random.split(key)
        self.DR = DR1Block(
            in_channels= in_channels , 
            out_channels= out_channels, 
            width = width, 
            kernel_size= kernel_size, 
            n_layers= n_layers,
            activation = activation, 
            key = DRkey,
        )

        # key, auxkey = jr.split(key)
        # self.aux = DR1Block(
        #     in_channels= in_channels, 
        #     out_channels= 1, 
        #     width = width, 
        #     kernel_size= kernel_size, 
        #     n_layers= n_layers,
        #     activation = activation, 
        #     key = auxkey,
        # )
        # key, ekey = jr.split(key)
        # self.E = DR1Block(
        #     in_channels= in_channels, 
        #     out_channels= 1, 
        #     width = width, 
        #     kernel_size= kernel_size, 
        #     n_layers= n_layers,
        #     activation = activation, 
        #     key = ekey,
        # )

        self.iter = iter

        #key, SAVkey = jr.split(key)
        #self.SAVblock = eqx.nn.MLP(
        #    in_size = 2, 
        #    out_size = 'scalar', 
        #    width_size = 128, 
        #    depth = 3, 
        #    activation = activation,
        #    key = SAVkey,
        #) 

    
    @eqx.filter_jit    
    def __call__(self, 
                 x,
                 ): 

        # def helper(i, xs):
        #     # x = jnp.stack(x, axis=0)
        #     q = xs[-1]
        #     x = xs[0]
        #     x_bar = self.DR(x) # phi_bar, u_bar
        #     latent_bar = self.aux(x_bar).squeeze(0)
        #     q_bar = q/(jnp.mean(latent_bar))
        #     eta, zeta = self.eta_zeta(q_bar)

        #     x_next = eta*x_bar
        #     e_next = jnp.mean(self.E(x_next))
        #     q_next = zeta * q_bar + (1- zeta) * e_next
        #     return (x_next, q_next)
        

        #     
        # return jax.lax.fori_loop(0, self.iter, helper, (x, q))
        q0 = self.energy(x)
        xtilde = self.DR(x)
        qtilde =  q0/(1-self.dt*self.dE_t(xtilde)/self.energy(xtilde))

        ksi = qtilde/self.energy(xtilde) 
        eta = 1 - (1-ksi)**2
        x1 = eta*xtilde
        return xtilde, x1


class EnergyBlock2d(eqx.Module): 
    DR : DR1Block
    H_inv : DR1Block

    def __init__(
            self, 
            in_channels, 
            width, 
            kernel_size, 
            n_layers,
            activation = jax.tree_util.Partial(jax.nn.tanh) , 
            *, 
            key
    ): 
        key, DRkey = jax.random.split(key)
        self.DR = DR1Block(
            in_channels= 2*in_channels , 
            out_channels= in_channels, 
            width = width, 
            kernel_size= kernel_size, 
            n_layers= n_layers,
            activation = activation, 
            key = DRkey,
        )
        key, Hkey = jr.split(key)
        self.H_inv = DR1Block(
            in_channels= 2*in_channels , 
            out_channels= in_channels, 
            width = width, 
            kernel_size= kernel_size, 
            n_layers= n_layers,
            activation = activation, 
            key = Hkey,
        )

    @eqx.filter_jit
    def __call__(
        self,
        phis, 
        q, 
    ):
        H_inv = self.H_inv(phis)
        DR_phis = self.DR(phis)
        q_next =  H_inv * DR_phis
        phis_next = 2* H_inv * (q_next - q) + phis

        return phis_next, q_next
        


class Estab2d(eqx.Module):
    n_blocks : int
    Eblocks : List[EnergyBlock2d]
    in_channels : int


    def __init__(
            self, 
            in_channels, 
            width, 
            kernel_size, 
            n_layers,
            n_blocks = 1,
            activation = jax.tree_util.Partial(jax.nn.tanh) , 
            *, 
            key
    ): 
        self.in_channels = in_channels
        self.Eblocks = []
        for i in range(n_blocks):
            key, ekey = jr.split(key)
            self.Eblocks.append(  
                EnergyBlock2d(
                in_channels= in_channels, 
                width = width, 
                kernel_size= kernel_size, 
                n_layers= n_layers,
                activation = activation, 
                key = ekey,
                )
            )

        self.n_blocks = n_blocks

    
    @eqx.filter_jit    
    def __call__(self, 
                 x,
                 q,
                 ): 

        if (self.in_channels == 1) and (q.shape[0] != 1):
            q = q[jnp.newaxis, :]
        
        for eblock in self.Eblocks:
            x, q = eblock(x, q)
        
        return (x, q) if self.in_channels != 1 else (x, q.squeeze(0))
