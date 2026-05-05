import jax.numpy as jnp
from quadax import simpson, trapezoid

def fft_x(phi, N_x, N_y, dx):

    phi_h = jnp.fft.rfft2(phi)
    wave = jnp.fft.fftfreq(N_x, d= dx) * 2j* jnp.pi
    wave_real = jnp.fft.rfftfreq(N_y, d= dx)* 2j* jnp.pi
    k_x, k_y = jnp.meshgrid(wave, wave_real, indexing='ij')
    phi_x_h =  k_x * phi_h

    phi_x = jnp.fft.irfft2(phi_x_h, s=(N_x, N_y))

    return phi_x

def fft_y(phi, N_x, N_y, dx):

    phi_h = jnp.fft.rfft2(phi)
    wave = jnp.fft.fftfreq(N_x, d= dx) * 2j* jnp.pi
    wave_real = jnp.fft.rfftfreq(N_y, d= dx)* 2j* jnp.pi
    k_x, k_y = jnp.meshgrid(wave, wave_real, indexing='ij')
    phi_y_h =  k_y * phi_h

    phi_y = jnp.fft.irfft2(phi_y_h, s=(N_x, N_y))

    return phi_y

def Laplacian_fft(u, N_x, N_y, dx): 
    u_h = jnp.fft.rfft2(u)
    wave = jnp.fft.fftfreq(N_x, d= dx) * 2j* jnp.pi
    wave_real = jnp.fft.rfftfreq(N_y, d= dx)* 2j* jnp.pi
    k_x, k_y = jnp.meshgrid(wave, wave_real, indexing='ij')

    u_x_h =  k_x * u_h
    u_xx_h = k_x * u_x_h
    u_y_h =  k_y * u_h
    u_yy_h = k_y * u_y_h

    u_xx = jnp.fft.irfft2(u_xx_h, s=(N_x, N_y))
    u_yy = jnp.fft.irfft2(u_yy_h, s=(N_x, N_y))

    return (u_xx + u_yy)

def trapezoid2D(integrand, dx): 
    return trapezoid(trapezoid(integrand, dx = dx, axis=-1) , dx =dx, axis=-1)

def simpson2D(integrand, dx, dy): 
    return simpson(simpson(integrand, dx = dy, axis=-1) , dx =dx, axis=-1)
