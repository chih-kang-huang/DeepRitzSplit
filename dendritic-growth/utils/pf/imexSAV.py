import jax 
import jax.numpy as jnp
import jax.random as jr
from jax import grad, vmap, jit
import numpy as np
import equinox as eqx
import time
import matplotlib.pyplot as plt
from functools import partial
# from utils.derivatives import fft_x, fft_y, Laplacian_fft


# Double Potential
@eqx.filter_jit
def W(s): 
    return ((s**2 - 1)**2)/4

# h' = 4W
@eqx.filter_jit
def h(s):
    return s**5/5 - 2/3 * s**3 + s



# BDF schemes
@eqx.filter_jit
def alpha_BDF(k): 
    # if k == 1:
    #     return 1
    # elif k == 2:
    #     return 3/2 
    # elif k == 3:
    #     return 11/6
    return (k-2)*(k-3)*1/2 + (k-1)*(k-3)*(-3/2) + (k-1)*(k-2)*11/12

def A_BDF(k, f_n, f_prev, f_penu):
    # if k == 1:
    #     return f_n 
    # elif k == 2: 
    #     return 2* f_n - 1/2* f_prev
    # elif k == 3: 
    #     return 3* f_n - 3/2*f_prev + 1/3* f_penu
    return (
        ((k-2)*(k-3)*1/2 + (k-1)*(k-3)*(-2) + (k-1)*(k-2)*3/2)*f_n +
        ((k-1)*(k-3)*1/2 +(k-1)*(k-2)*(-3/4))*f_prev +
        (k-1)*(k-2)*1/6 * f_penu
    )


def B_BDF(k, f_n, f_prev, f_penu):
    # if k == 1:
    #     return f_n 
    # elif k == 2: 
    #     return 2* f_n - f_prev
    # elif k == 3: 
    #     return 3* f_n - 3*f_prev + f_penu
    return (
        ((k-2)*(k-3)*1/2 + (k-1)*(k-3)*(-2) + (k-1)*(k-2)*3/2)*f_n +
        ((k-1)*(k-3)*1 +(k-1)*(k-2)*(-3/2))*f_prev +
        (k-1)*(k-2)*1/2 * f_penu
    )

def compute_u_tilde_next_fft(u_init, phi_init, phi_next, dt, D, K, N_x, N_y, dx):
    
    B = jnp.fft.rfft2( 
        1/dt * u_init + K * 4*W(phi_init) * (phi_next - phi_init)/dt
    )

    ones = jnp.ones_like(B)

    wave = jnp.fft.fftfreq(N_x, d=dx) * 2j* jnp.pi
    wave_real = jnp.fft.rfftfreq(N_y, d=dx)* 2j* jnp.pi
    k_x, k_y = jnp.meshgrid(wave, wave_real, indexing='ij')

    A = ones * (1/dt) - D*(k_x**2 + k_y**2)

    return  jnp.fft.irfft2(B*(A**(-1)), s=(N_x, N_y))

class Solver_IMEXSAV: 
    def __init__(
            self, 
            eps_m, eps, lambda_0, D, K, theta_0, tau_0, Delta,
            N_x, N_y, dx,
            alpha, beta,
    ):
        # Mesh 
        self.N_x = N_x 
        self.N_y = N_y 
        self.dx = dx
        # Define FFT
        from utils.derivatives import fft_x, fft_y, Laplacian_fft
        self.fft_x = eqx.filter_jit(partial(fft_x, N_x = N_x, N_y = N_y,dx = dx))
        self.fft_y = eqx.filter_jit(partial(fft_y, N_x = N_x, N_y = N_y,dx = dx))
        self.Laplacian = eqx.filter_jit(partial(Laplacian_fft, N_x = N_x, N_y = N_y,dx = dx))

        # Physical Parameters 
        self.eps_m = eps_m
        self.eps = eps
        self.lambda_0 = lambda_0
        self.D = D
        self.K = K # latent heat parameter
        self.tau_0 = tau_0
        self.Delta = Delta
        self.theta_0 = theta_0

        # Splitting Parameters 
        self.alpha = alpha 
        self.beta = beta



    # Anisotropic term
    @eqx.filter_jit
    def a(self, phi, tol = 1e-12): 
        # m = 4
        phi_x = self.fft_x(phi)
        phi_y = self.fft_y(phi)
        # phi_x = centered_derivative_x(phi)
        # phi_y = centered_derivative_y(phi)

        return (1-3*self.eps_m) * (
            1 + 4*self.eps_m/(1-3*self.eps_m) * 
            (phi_x**4 + phi_y**4)/((phi_x**2 + phi_y**2 + tol)**2)
        )

    @eqx.filter_jit
    def E(self, phi, u):
        phi_x = self.fft_x(phi)
        phi_y = self.fft_y(phi)
        return 1/2*(self.a(phi)**2)*(phi_x**2 + phi_y**2) + W(phi)/(self.eps**2) + self.lambda_0 * u**2 /(2*self.eps*self.K)

    @eqx.filter_jit
    def energy(self, phi, u, Area = 1): 
        return Area * jnp.mean(self.E(phi, u)) + 1


    @eqx.filter_jit
    def E1(self, phi, u):
        phi_x = self.fft_x(phi)
        phi_y = self.fft_y(phi)
        
        return self.alpha*(phi_x**2 + phi_y **2)/2 + self.beta*phi**2/(2*self.eps**2)

    @eqx.filter_jit
    def energy1(self, phi, u, Area = 1): 
        return Area * jnp.mean(self.E1(phi, u)) + 1
    

    @eqx.filter_jit
    def E2(self, phi, u):
        return self.E(phi, u) - self.E1(phi, u)


    @eqx.filter_jit
    def H(self, phi, tol=1e-12): 
        phi_x = self.fft_x(phi)
        phi_y = self.fft_y(phi)

        H1 = 4 * self.eps_m * 4 * phi_x * (phi_x**2 * phi_y**2 - phi_y**4) /(phi_x**2 + phi_y**2 + tol)**3
        H2 = 4 * self.eps_m * 4 * phi_y * (phi_x**2 * phi_y**2 - phi_x**4) /(phi_x**2 + phi_y**2 + tol)**3
        return H1, H2


    @eqx.filter_jit
    def dE1(self, phi): 
        Laplacian_phi = self.Laplacian(phi)
        return -self.alpha * Laplacian_phi + self.beta*phi/self.eps**2

    @eqx.filter_jit
    def dE(self, phi): 
        phi_x = self.fft_x(phi)
        phi_y = self.fft_y(phi)
        H1, H2 = self.H(phi)

        A1 =  self.a(phi)**2 * phi_x + (phi_x**2 + phi_y**2)*self.a(phi)*H1
        A2 =  self.a(phi)**2 * phi_y + (phi_x**2 + phi_y**2)*self.a(phi)*H2

        A1_x = self.fft_x(A1)
        A2_y = self.fft_y(A2)

        return - (A1_x + A2_y) + (phi**3 - phi)/self.eps**2

    @eqx.filter_jit
    def dE2(self, phi): 
        return self.dE(phi) - self.dE1(phi)

    @eqx.filter_jit
    def dE_t(self, phi, u, Area=1): 
        u_x = self.fft_x(u)
        u_y = self.fft_y(u)

        integrand = self.dE(phi) + self.lambda_0/self.eps * 4*W(phi)*u


        return - Area*(
            (jnp.mean(integrand**2))/self.tau_0 + 
            self.lambda_0*self.D/(self.eps*self.K) * jnp.mean((u_x**2 + u_y**2))
        )


    @eqx.filter_jit
    def phi_bar(self,
                k, phi_init, u_init, 
                phi_prev, u_prev, 
                phi_penu, u_penu,
                dt
                ): 
        """ 
        A phi_bar = B
        """
        A_k_phi = A_BDF(k, phi_init, phi_prev, phi_penu)
        B_k_phi = B_BDF(k, phi_init, phi_prev, phi_penu)
        B_k_u = B_BDF(k, u_init, u_prev, u_penu)

        B = jnp.fft.rfft2( 
            self.tau_0/dt * A_k_phi - self.dE2(B_k_phi) - self.lambda_0/self.eps * 4 * W(B_k_phi) * B_k_u 
        )

        ones = jnp.ones_like(B)

        wave = jnp.fft.fftfreq(self.N_x, d=self.dx) * 2j* jnp.pi
        wave_real = jnp.fft.rfftfreq(self.N_y, d=self.dx)* 2j* jnp.pi
        k_x, k_y = jnp.meshgrid(wave, wave_real, indexing='ij')

        A = ones * (self.tau_0 * alpha_BDF(k)/dt + self.beta/self.eps**2) - self.alpha *(k_x**2 + k_y**2)

        return  jnp.fft.irfft2(B*(A**(-1)), s=(self.N_x, self.N_y))

    @eqx.filter_jit
    def u_bar(self, k, phi_init, u_init, 
            phi_prev, u_prev, 
            phi_penu, u_penu,
            dt
        ): 

        """ 
        A u_bar = B
        """
        A_k_phi = A_BDF(k, phi_init, phi_prev, phi_penu)
        B_k_phi = B_BDF(k, phi_init, phi_prev, phi_penu)
        A_k_u = A_BDF(k, u_init, u_prev, u_penu)

        phi_b = self.phi_bar(k,
            phi_init, u_init, 
            phi_prev, u_prev,
            phi_penu, u_penu, 
            dt
        )

        B = jnp.fft.rfft2( 
            1/dt * A_k_u + self.K * 4*W(B_k_phi) * (alpha_BDF(k)*phi_b - A_k_phi )/dt
        )

        ones = jnp.ones_like(B)
        wave = jnp.fft.fftfreq(self.N_x, d=self.dx) * 2j* jnp.pi
        wave_real = jnp.fft.rfftfreq(self.N_y, d=self.dx)* 2j* jnp.pi
        k_x, k_y = jnp.meshgrid(wave, wave_real, indexing='ij')

        A = ones * (alpha_BDF(k)/dt) - self.D*(k_x**2 + k_y**2)

        return  jnp.fft.irfft2(B*(A**(-1)), s=(self.N_x, self.N_y))
    
    @eqx.filter_jit
    def q_bar(self, q, phi_bar, u_bar, dt): 
        H = - self.dE_t(phi_bar, u_bar)
        e = self.energy(phi_bar, u_bar)
        return q / (
        1 + dt* H/e
    )
    def compute_next(
            self,
            k, phi, u, q, 
            phi_prev, u_prev, 
            phi_penu, u_penu,
            dt):




        p_b = self.phi_bar(
                    k, phi, u, 
                    phi_prev, u_prev, 
                    phi_penu, u_penu,
                    dt
                    )

        u_b = self.u_bar(
                    k, phi, u, 
                    phi_prev, u_prev, 
                    phi_penu, u_penu,
                    dt
                    )
        q_b = self.q_bar(q, p_b, u_b, dt)
    
        xi = q_b/(self.energy(p_b, u_b))
        eta = 1 - (1-xi)**(k+1)
    
        phi_next = p_b * eta
    
        u_next = u_b * eta
    
        def zeta(phi_next, u_next, phi_b, u_b, q_b):
            e = self.energy(phi_next, u_next)
            E_relaxed = e + dt * q_b * self.dE_t(phi_b, u_b)/self.energy(phi_b, u_b)
            if q_b >= e:
                return 0.
            elif (q_b < e) and (q_b >= E_relaxed):
                return 0.
            else:
                return (1 - dt * q_b * (- self.dE_t(phi_b, u_b))/(self.energy(phi_b, u_b) * (e - q_b))) * (1) # * (1+1e-4)
    
    
        ze = zeta(phi_next, u_next, p_b, u_b, q_b)
        q_next = ze * q_b + (1-ze)*self.energy(phi_next, u_next)
    
        assert( 
            (q_next - q_b)/dt <=
            q_b* (-self.dE_t(p_b, u_b))/self.energy(p_b, u_b)
        )
    
        return phi_next, u_next, q_next
   



def anisotropy_fft(phi, eps_m, fft_x, fft_y, tol=1e-12): 
    # m = 4
    phi_x = fft_x(phi)
    phi_y = fft_y(phi)
    # phi_x = centered_derivative_x(phi)
    # phi_y = centered_derivative_y(phi)

    return (1-3*eps_m) * (
        1 + 4*eps_m/(1-3*eps_m) * 
        (phi_x**4 + phi_y**4)/((phi_x**2 + phi_y**2 + tol)**2)
    )

@eqx.filter_jit
def normal_fft(phi,fft_x, fft_y, tol=1e-12): 
    phi_x =fft_x(phi)
    phi_y =fft_y(phi)
    n_1 = phi_x/(jnp.sqrt( phi_x**2 + phi_y**2 + tol))
    n_2 = phi_y/(jnp.sqrt( phi_x**2 + phi_y**2 + tol))
    return jnp.stack([n_1, n_2], axis=-1)

### Do not work well
@eqx.filter_jit
def curvature_div(phi, normal, fft_x, fft_y) :
    n = normal(phi)
    n1 = n[..., 0]
    n2 = n[..., -1]
    return fft_x(n1) + fft_y(n2)

@eqx.filter_jit
def perimeter_pf(phi, eps, fft_x, fft_y, trapezoid2D): 
    phi_x = fft_x(phi)
    phi_y = fft_y(phi)
    W_vec = jax.vmap(jax.vmap(W))
    integrand = eps*(phi_x**2 + phi_y**2)/2 + W_vec(phi)/eps
    return trapezoid2D(integrand)


@eqx.filter_jit
def E(phi, u, eps, lambda_0, K, a, fft_x, fft_y):
    phi_x = fft_x(phi)
    phi_y = fft_y(phi)
    return 1/2*(a(phi)**2)*(phi_x**2 + phi_y**2) + W(phi)/(eps**2) + lambda_0 * u**2 /(2*eps*K)

@eqx.filter_jit
def energy(phi_u, E, quad, Area = 1): 
    phi, u = phi_u[0], phi_u[1]
    return Area * quad(E(phi, u)) + 1




def compare_with_SAV(model, phi_u_init, 
                    x_lb, x_rb, y_lb, y_ub,
                    dt, 
                    solver, 
                    freq = 10, 
                    delay = 0,
                    save_path=None,
                    ):


    phi_u_SAV = phi_u_init
    q_SAV = solver.energy(phi_u_SAV[0], phi_u_SAV[1])

    for _ in range(delay):
        phi_SAV, u_SAV, q_SAV = solver.compute_next(1, phi_u_SAV[0], phi_u_SAV[1], q_SAV,
                                phi_u_SAV[0], phi_u_SAV[1],
                                phi_u_SAV[0], phi_u_SAV[1],
                                dt
                            )
        phi_u_SAV = jnp.stack([phi_SAV, u_SAV], axis =0)
        
    phi_u = phi_u_SAV
    q_SAV = solver.energy(phi_u_SAV[0], phi_u_SAV[1])
    q_checkpoint = [solver.energy(phi_u[0], phi_u[1])]
    q_SAV_checkpoint = [q_SAV]


    time_NN = 0
    time_SAV = 0

    plt.figure(figsize=(25, 30))

    for i in range(5): 
        if i == 0 : 
            T = 0
            phi_SAV = phi_u_init[0]
        else:
            for _ in range(freq):
                time_in = time.time()
                phi_u = model.compute_next(phi_u)
                time_out = time.time()
                q = solver.energy(phi_u[0], phi_u[1])
                q_checkpoint.append(q)
                time_NN += time_out - time_in


                time_in = time.time()
                phi_SAV, u_SAV, q_SAV = solver.compute_next(1, phi_u_SAV[0], phi_u_SAV[1], q_SAV,
                                        phi_u_SAV[0], phi_u_SAV[1],
                                        phi_u_SAV[0], phi_u_SAV[1],
                                        dt
                                    )
                phi_u_SAV = jnp.stack([phi_SAV, u_SAV], axis =0)
                time_out = time.time()
                q_SAV_checkpoint.append(q_SAV)
                time_SAV += time_out - time_in
            T = dt*freq*i + 1
        plt.subplot(6, 5, 1+i)
        plt.imshow(phi_u[0], extent=[x_lb, x_rb, y_lb, y_ub], origin='lower', cmap='jet' )
        plt.title(f"phi_NN at T={T}")
        plt.colorbar()
        plt.subplot(6, 5, 6 + i)
        plt.imshow(phi_u[1], extent=[x_lb, x_rb, y_lb, y_ub], origin='lower', )
        plt.title(f"U at T={T}")
        plt.colorbar()
        plt.subplot(6, 5, 11 + i)
        plt.imshow(phi_u_SAV[0], extent=[x_lb, x_rb, y_lb, y_ub], origin='lower', cmap = 'jet')
        plt.title(f"phi_SAV at T={T}")
        plt.colorbar()
        plt.subplot(6, 5, 16 + i)
        plt.imshow(phi_u_SAV[1], extent=[x_lb, x_rb, y_lb, y_ub], origin='lower', )
        plt.title(f"U_SAV at T={T}")
        plt.colorbar()
        plt.subplot(6, 5, 21 + i)
        plt.imshow(abs(phi_u_SAV[0] - phi_u[0]), extent=[x_lb, x_rb, y_lb, y_ub], origin='lower', cmap='rainbow')
        plt.title(f"abs diff at T={T}")
        plt.colorbar()
    plt.subplot(6, 1, 6)
    plt.plot(dt*np.arange(len(q_checkpoint)), q_checkpoint, label='q_NN')
    plt.plot(dt*np.arange(len(q_checkpoint)), q_SAV_checkpoint, label='q_SAV')
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Energy")
    print(f"Computetime NN: {time_NN}| SAV {time_SAV}")
    if save_path:
        plt.savefig(f"{save_path}", bbox_inches='tight')