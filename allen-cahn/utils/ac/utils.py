import jax 
import jax.numpy as jnp
import numpy as np
from jax import jit
import equinox as eqx
import matplotlib.pyplot as plt

@eqx.filter_jit
def energy(u_sol, N_x, N_y, dx, eps): 
    u_pred = u_sol
    u_h = jnp.fft.rfft2(u_pred)
    wave = jnp.fft.fftfreq(N_x, d= dx) * 2j* jnp.pi 
    wave_real = jnp.fft.rfftfreq(N_y, d= dx)* 2j* jnp.pi
    k_x, k_y = jnp.meshgrid(wave, wave_real, indexing='ij')
    u_x_h =  k_x * u_h
    u_y_h =  k_y * u_h

    u_x = jnp.fft.irfft2(u_x_h, s=(N_x, N_y))
    u_y = jnp.fft.irfft2(u_y_h, s=(N_x, N_y))
    # Constant from the double potential
    return (jnp.mean(eps* (u_x**2 + u_y**2)/2 + ((u_pred**2 - 1)**2)/(4*eps)))/(4/(3*jnp.sqrt(2)))


@eqx.filter_jit
def volume(u_sol):
    assert(len(u_sol.shape) == 2)
    def integrand(s): 
        return (s - s**3/3 + 2/3)/jnp.sqrt(2)
    v = integrand(u_sol)
    return jnp.mean(v)/(4/(3*jnp.sqrt(2)))
    #return jnp.mean((u_sol +1 )/2)


def compare_DR(*f_inits, freqs, 
               model, DR_next, N_x, N_y, dx, dt, eps,
               x_lb, x_rb, y_lb, y_ub, alpha,
               title = None, savepath = None, freq_init = 1, method = 1, top = 0.978): 

    def helper(n, L, f_init, freq): 
        u_NN = f_init 
        u_num = f_init
        diff_history = []

        energy_NN = [energy(f_init, N_x, N_y, dx, eps)]
        energy_num = [energy(f_init, N_x, N_y, dx, eps)]
        for i in range(3):
            if i == 0 :
                for _ in range(freq_init):
                    if method == 1: 
                        u_NN = model(u_NN)
                    elif method == 2:
                    #    _, u_NN = model(u_NN)
                        u_NN = model(u_NN)
                    u_num = DR_next(u_num)
                    diff_history.append(jnp.mean(abs(u_NN - u_num)))
                    energy_NN.append(energy(u_NN, N_x, N_y, dx, eps))
                    energy_num.append(energy(u_num, N_x, N_y, dx, eps))
                plt.subplot(4*L, 3, 12*n +1 + 3*i)
                plt.imshow(u_num, cmap='rainbow', extent=[x_lb, x_rb, y_lb, y_ub], origin='lower', vmin=-1, vmax=1)
                plt.title(f"u_splittting at T={dt*freq_init:.2e}")
                plt.subplot(4*L, 3, 12*n  + 2 + 3*i)
                plt.imshow(u_NN, cmap='rainbow', extent=[x_lb, x_rb, y_lb, y_ub], origin='lower', vmin=-1, vmax=1)
                plt.title(f"u_NN at T={dt*freq_init:.2e}")
                plt.subplot(4*L, 3, 12*n + 3 + 3*i)
                plt.imshow(abs(u_num-u_NN), cmap='rainbow', extent=[x_lb, x_rb, y_lb, y_ub], origin='lower')
                plt.colorbar()
                plt.title(f"abs_diff at T={dt*freq_init:.2e}")
            else :
                for _ in range(freq):
                    if method == 1: 
                        u_NN = model(u_NN)
                    elif method == 2:
                    #    _, u_NN = model(u_NN)
                        u_NN = model(u_NN)
                    u_num = DR_next(u_num)
                    diff_history.append(jnp.mean(abs(u_NN - u_num)))
                    energy_NN.append(energy(u_NN, N_x, N_y, dx, eps))
                    energy_num.append(energy(u_num, N_x, N_y, dx, eps))
                plt.subplot(4*L, 3, 12*n + 1 + 3*i)
                plt.imshow(u_num, cmap='rainbow', extent=[x_lb, x_rb, y_lb, y_ub], origin='lower', vmin=-1, vmax=1)
                plt.title(f"u_splittting at T={(i)*dt*freq:.2e}")
                plt.subplot(4*L, 3, 12*n + 2 + 3*i)
                plt.imshow(u_NN, cmap='rainbow', extent=[x_lb, x_rb, y_lb, y_ub], origin='lower', vmin=-1, vmax=1)
                plt.title(f"u_NN at T={(i)*dt*freq:.2e}")
                plt.subplot(4*L, 3, 12*n +  3 + 3*i)
                plt.imshow(abs(u_num-u_NN), cmap='rainbow', extent=[x_lb, x_rb, y_lb, y_ub], origin='lower')
                plt.colorbar()
                plt.title(f"abs_diff at T={(i)*dt*freq:.2e}")
        plt.subplot(4*L, 1, 4*(n+1))
        plt.plot(dt*np.arange(len(energy_NN)), energy_NN, label="energy_NN")
        plt.plot(dt*np.arange(len(energy_NN)), energy_num, label="energy_num")
        plt.legend()
        plt.xlabel("Time")
        plt.ylabel("Energy")


    L = len(f_inits)
    plt.figure(figsize=(9, 15*L))
    if not isinstance(freqs, list) or (len(freqs) != L): 
        print(f"Generate freqs = 5")
        freqs = [5]*L

    for n in range(L):
        f_init = f_inits[n]
        freq = freqs[n]
        helper(n, L, f_init, freq)
    
    
    
    if title : 
        plt.suptitle(title, y = 0.98)
        #plt.tight_layout()
        plt.subplots_adjust(top=top)
    
    
    if savepath: 
        plt.savefig(savepath)
        



def print_DR(*f_inits, freqs, 
               model, DR_next, N_x, N_y, dx, dt, eps,
               x_lb, x_rb, y_lb, y_ub, alpha,
               title = None, savepath = None, freq_init = 1, method = 1, top = 0.978): 

    def helper(n, L, f_init, freq): 
        u_NN = f_init 
        u_num = f_init
        diff_history = []

        energy_NN = [energy(f_init, N_x, N_y, dx, eps)]
        energy_num = [energy(f_init, N_x, N_y, dx, eps)]
        for i in range(3):
            if i == 0 :
                for _ in range(freq_init):
                    u_NN = model(u_NN)
                    u_num = DR_next(u_num)
                    diff_history.append(jnp.mean(abs(u_NN - u_num)))
                    energy_NN.append(energy(u_NN, N_x, N_y, dx, eps))
                    energy_num.append(energy(u_num, N_x, N_y, dx, eps))
                plt.subplot(4*L, 3, 12*n +1 + 3*i)
                plt.imshow(u_num, cmap='rainbow', extent=[x_lb, x_rb, y_lb, y_ub], origin='lower', vmin=-1, vmax=1)
                plt.title(f"u_splittting at T={dt*freq_init:.2e}")
                plt.subplot(4*L, 3, 12*n  + 2 + 3*i)
                plt.imshow(u_NN, cmap='rainbow', extent=[x_lb, x_rb, y_lb, y_ub], origin='lower', vmin=-1, vmax=1)
                plt.title(f"u_NN at T={dt*freq_init:.2e}")
                plt.subplot(4*L, 3, 12*n + 3 + 3*i)
                plt.imshow(abs(u_num-u_NN), cmap='rainbow', extent=[x_lb, x_rb, y_lb, y_ub], origin='lower')
                plt.colorbar()
                plt.title(f"abs_diff at T={dt*freq_init:.2e}")
            else :
                for _ in range(freq):
                    if method == 1: 
                        u_NN = model(u_NN)
                    elif method == 2:
                    #    _, u_NN = model(u_NN)
                        u_NN = model(u_NN)
                    u_num = DR_next(u_num)
                    diff_history.append(jnp.mean(abs(u_NN - u_num)))
                    energy_NN.append(energy(u_NN, N_x, N_y, dx, eps))
                    energy_num.append(energy(u_num, N_x, N_y, dx, eps))
                plt.subplot(4*L, 3, 12*n + 1 + 3*i)
                plt.imshow(u_num, cmap='rainbow', extent=[x_lb, x_rb, y_lb, y_ub], origin='lower', vmin=-1, vmax=1)
                plt.title(f"u_splittting at T={(i)*dt*freq:.2e}")
                plt.subplot(4*L, 3, 12*n + 2 + 3*i)
                plt.imshow(u_NN, cmap='rainbow', extent=[x_lb, x_rb, y_lb, y_ub], origin='lower', vmin=-1, vmax=1)
                plt.title(f"u_NN at T={(i)*dt*freq:.2e}")
                plt.subplot(4*L, 3, 12*n +  3 + 3*i)
                plt.imshow(abs(u_num-u_NN), cmap='rainbow', extent=[x_lb, x_rb, y_lb, y_ub], origin='lower')
                plt.colorbar()
                plt.title(f"abs_diff at T={(i)*dt*freq:.2e}")


        plt.subplot(4*L, 1, 4*(n+1))
        plt.plot(dt*np.arange(len(energy_NN)), energy_NN, label="Perimeter_NN")
        plt.plot(dt*np.arange(len(energy_NN)), energy_num, label="Perimeter_num")
        plt.legend()
        plt.xlabel("Time")
        plt.ylabel("Perimeter")


    L = len(f_inits)
    plt.figure(figsize=(9, 15*L))
    if not isinstance(freqs, list) or (len(freqs) != L): 
        print(f"Generate freqs = 5")
        freqs = [5]*L

    for n in range(L):
        f_init = f_inits[n]
        freq = freqs[n]
        helper(n, L, f_init, freq)
    
    
    
    if title : 
        plt.suptitle(title, y = 0.98)
        #plt.tight_layout()
        plt.subplots_adjust(top=top)
    
    
    if savepath: 
        plt.savefig(savepath)
        