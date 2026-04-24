import jax 
import jax.numpy as jnp
from jax import jit
import equinox as eqx



@jit
def Laplacian(p, h, gamma=1/3): 
    """
    nine-point isotropic Laplacian

    Delta p(i, j) = (2./3.)*((p(i,j-1)+p(i,j+1)+p(i+1,j)+p(i-1,j))
    +0.25*(p(i+1,j-1)+p(i-1,j-1)+p(i+1,j+1)+p(i-1,j+1))
    -5.*p(i,j))
    """
    assert(len(p.shape) == 2)
    gamma = 1/3
    f = (1/(h**2)) * (
        (1-gamma) * jnp.array(
        [
            [0, 1, 0], 
            [1, -4, 1], 
            [0, 1, 0], 
        ]
    ) + (gamma) * jnp.array(
        [
            [1/2, 0, 1/2], 
            [0, -2, 0], 
            [1/2, 0, 1/2], 
        ]
    )
    )

    #f = f.at[1:-1, 1:-1].set(
    #    (1-gamma)*(p[1:-1, 0:-2] + p[1:-1, 2:] + p[2:, 1:-1] + p[0:-2, 1:-1])
    #    + gamma/2 * (p[2:, 0:-2] + p[0:-2, 0:-2] + p[2:, 2:] + p[0:-2, 2:])
    #    - (4*(1-gamma) + 2*gamma/2)*p[1:-1, 1:-1]
    #)

    return jax.scipy.signal.convolve2d(p, f)[1:-1, 1:-1]
#    return jax.scipy.signal.convolve2d(p, f,  boundary='wrap', mode='same')

def compute_u_next(U, N_x, N_y, dx, dt, eps=1/64):
    """ 
    u(n, i, j) = u(t_n, x_i, y_j), for n>1
    """
    #if n == 0: 
    #    return f_im
    ##elif (i == 0) or (i == N_x -1) or  (j ==0) or (j== N_y -1):
    ##    return -1.
    #else: 
    #B = jnp.diag(jnp.ones(N_x-1), k =1) + jnp.diag(jnp.ones(N_x -1), k = -1) 
    #Delta_u = (B@ U + U @ B - 4*U) /(h**2)
    Delta_u = Laplacian(U, h=dx)
    U_next  = jnp.zeros( (N_x, N_y))
    U_next = U_next.at[:, :].set(
        (U + dt*10* (40**(-2)) * (Delta_u  + (eps**(-2))*(U -U**3)))
    )
    U_next  = U_next.at[0, :].set(-1)
    U_next  = U_next.at[-1, :].set(-1)
    U_next  = U_next.at[:, 0].set(-1)
    U_next  = U_next.at[:, -1].set(-1)
    #U_next = (
    #    (U + delta_t*10* (Delta_u *(eps**2) + (U -U**3)))
    #)
    #U_next[0, :] = (-1)
    #U_next[-1, :] = (-1)
    #U_next[:, 0] = (-1)
    #U_next[:, -1] = (-1)
    return U_next

@eqx.filter_jit 
def splitting_next(u, N_x, N_y, dx, eps, alpha, dt, C=1): 
    """
    A x u_{n+1, h} = B(u_{n, h}) 
    """


    B = jnp.fft.rfft2( 
        u  + C*dt* eps**(-2) * ( u - u**3 + alpha*u)
    )

    I = jnp.ones_like(B)

    wave = jnp.fft.fftfreq(N_x, d=dx) * 2j* jnp.pi
    wave_real = jnp.fft.rfftfreq(N_y, d=dx)* 2j* jnp.pi
    k_x, k_y = jnp.meshgrid(wave, wave_real, indexing='ij')

    A = I - C*dt* ( (k_x**2 + k_y**2) - alpha*(eps**(-2))*I) 
    
    return  jnp.fft.irfft2(B*(A**(-1)), s=(N_x, N_y))

@eqx.filter_jit 
def DR1_next(u, N_x, N_y, dx, dt, eps, alpha, C=1, gamma_10 = 1): 
    """
    A x u_{n+1, h} = B(u_{n, h}) 
    """


    B = jnp.fft.rfft2( 
        u  + C*(dt/gamma_10)* eps**(-2) * ( u - u**3 + alpha*u)
    )

    I = jnp.ones_like(B)

    wave = jnp.fft.fftfreq(N_x, d=dx) * 2j* jnp.pi
    wave_real = jnp.fft.rfftfreq(N_y, d=dx)* 2j* jnp.pi
    k_x, k_y = jnp.meshgrid(wave, wave_real, indexing='ij')

    A = I - C*(dt/gamma_10)* ( (k_x**2 + k_y**2) - alpha*(eps**(-2))*I)
    return  jnp.fft.irfft2(B*(A**(-1)), s=(N_x, N_y))
    
@eqx.filter_jit 
def DR2_next(u_init, N_x, N_y, dx, dt, eps, alpha, 
             gamma_10 , gamma_20, gamma_21, 
             theta_20 , theta_21, C = 1,
             ): 
    u_star =  DR1_next(u_init, N_x, N_y, dx, dt, eps, alpha, gamma_10)

    B = jnp.fft.rfft2( 
        gamma_20* u_init  + theta_20* C*dt* eps**(-2) * ( u_init - u_init**3 + alpha*u_init) +
        gamma_21* u_star  + theta_21* C*dt* eps**(-2) * ( u_star - u_star**3 + alpha*u_star)
    )/(gamma_20 + gamma_21)

    I = jnp.ones_like(B)

    wave = jnp.fft.fftfreq(N_x, d=dx) * 2j* jnp.pi
    wave_real = jnp.fft.rfftfreq(N_y, d=dx)* 2j* jnp.pi
    k_x, k_y = jnp.meshgrid(wave, wave_real, indexing='ij')

    A = I - C*(dt/(gamma_20 + gamma_21))* ( (k_x**2 + k_y**2) - alpha*(eps**(-2))*I)

    return  jnp.fft.irfft2(B*(A**(-1)), s=(N_x, N_y))