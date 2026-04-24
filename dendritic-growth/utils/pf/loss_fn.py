import jax
import jax.numpy as jnp
import equinox as eqx

@eqx.filter_jit
def residu_splitting(model, phi_u_init, E1, E2, dE2, tau_0, dt, beta): 
    phi_init, u_init = phi_u_init[0], phi_u_init[1]
    phi_tilde = model(phi_u_init)

    #  Splitting scheme
    E = (
       E1(phi_tilde) +
       dE2(phi_init, u_init, beta)* (phi_tilde - phi_init)
       + E2(phi_init, u_init, beta)
    )

    step =  tau_0 * (phi_tilde - phi_init)**2/(2*dt)

    return 2*dt*(step + E)

def loss_splitting(model, phi_u_inits, integral_method): 
    r_var = jax.vmap(residu_splitting, in_axes=(None, 0))(model, phi_u_inits)
    #return jnp.mean(r_var)
    return jnp.mean(
        jax.vmap(integral_method, in_axes=0)(r_var)
    )



# class splittingLoss: 



# Loss function related to data-driven
@eqx.filter_jit
def residu_data(model, phi_u_init, phi_u_next):
    phi_pred = model(phi_u_init)

    return (phi_pred - phi_u_next[0])

@eqx.filter_jit
def loss_data(model, phi_u_init, phi_u_next):
    r_phi_u= jax.vmap(residu_data, in_axes=(None, 0, 0))(model, phi_u_init, phi_u_next)
    return jnp.mean(r_phi_u**2)

@eqx.filter_jit
def reg_hyperparams(model, filter_spec=eqx.is_array, d =2):
    params, _ = eqx.partition(model, filter_spec=filter_spec)
    regu = jnp.sum(
        jnp.abs(jax.flatten_util.ravel_pytree(params)[0])**d
    )

    return regu



