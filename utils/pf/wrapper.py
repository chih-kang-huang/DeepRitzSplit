import jax
import jax.numpy as jnp 
import equinox as eqx

# Double Potential
@eqx.filter_jit
def W(s): 
    return ((s**2 - 1)**2)/4

# h' = 4W
@eqx.filter_jit
def h(s):
    return s**5/5 - 2/3 * s**3 + s

class dendriteGrowth2D(eqx.Module): 
    NN : eqx.Module
    x_lb : float
    x_rb : float
    y_lb : float
    y_ub : float
    dt : float
    D : float
    K : float
    N_x : int
    N_y : int
    eps : float
    eps_m : float
    m : int
    tau_0 : float
    lambda_0 : float

    def __init__(self, 
            NN, dt, boundaries, nb_grids,
            D, K, eps, eps_m, m, tau_0, lambda_0
    ):
        self.NN = NN
        self.x_lb, self.x_rb, self.y_lb, self.y_ub = boundaries[0], boundaries[1], boundaries[2], boundaries[3]
        self.N_x, self.N_y = nb_grids[0], nb_grids[1]
        self.D = D
        self.K = K
        self.dt =dt
        self.eps = eps
        self.eps_m = eps_m
        self.m = m
        self.tau_0 = tau_0
        self.lambda_0 = lambda_0
    

    @eqx.filter_jit
    def fft_x(self, phi):

        dx = (self.x_rb-self.x_lb)/self.N_x
        dy = (self.y_ub-self.y_lb)/self.N_y

        phi_h = jnp.fft.rfft2(phi)
        wave = jnp.fft.fftfreq(self.N_x, d= dx) * 2j* jnp.pi
        wave_real = jnp.fft.rfftfreq(self.N_y, d= dy)* 2j* jnp.pi
        k_x, k_y = jnp.meshgrid(wave, wave_real, indexing='ij')
        phi_x_h =  k_x * phi_h

        phi_x = jnp.fft.irfft2(phi_x_h, s=(self.N_x, self.N_y))

        return phi_x

    @eqx.filter_jit
    def fft_y(self, phi):

        dx = (self.x_rb-self.x_lb)/self.N_x
        dy = (self.y_ub-self.y_lb)/self.N_y

        phi_h = jnp.fft.rfft2(phi)
        wave = jnp.fft.fftfreq(self.N_x, d= dx) * 2j* jnp.pi
        wave_real = jnp.fft.rfftfreq(self.N_y, d= dy)* 2j* jnp.pi
        k_x, k_y = jnp.meshgrid(wave, wave_real, indexing='ij')
        phi_y_h =  k_y * phi_h

        phi_y = jnp.fft.irfft2(phi_y_h, s=(self.N_x, self.N_y))

        return phi_y

    # Anisotropic term
    @eqx.filter_jit
    def a(self, phi, tol = 1e-12): 
        # m = 4
        phi_x = self.fft_x(phi)
        phi_y = self.fft_y(phi)

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
    def energy(self, phi_u, Area = 1): 
        return Area * jnp.mean(self.E(phi_u)) + 1

    @eqx.filter_jit
    def compute_u_next(self, u_init, phi_init, phi_next):
        B = jnp.fft.rfft2( 
            1/self.dt * u_init + self.K * 4*W(phi_init) * (phi_next - phi_init)/self.dt
        )
        dx = (self.x_rb - self.x_lb)/self.N_x
        dy = (self.y_ub - self.y_lb)/self.N_y

        ones = jnp.ones_like(B)

        wave = jnp.fft.fftfreq(self.N_x, d=dx) * 2j* jnp.pi
        wave_real = jnp.fft.rfftfreq(self.N_y, d=dy)* 2j* jnp.pi
        k_x, k_y = jnp.meshgrid(wave, wave_real, indexing='ij')

        A = ones * (1/self.dt) - self.D*(k_x**2 + k_y**2)

        return  jnp.fft.irfft2(B*(A**(-1)), s=(self.N_x, self.N_y))
    
    @eqx.filter_jit 
    def compute_next(self, phi_u_init): 
        phi_init, u_init = phi_u_init[0], phi_u_init[1]
        phi_next = self(phi_u_init)
        u_next = self.compute_u_next(u_init, phi_init, phi_next)
        return jnp.stack([phi_next, u_next], axis=0)
    

    @eqx.filter_jit
    def convolve_wrap(self, a1, a2):
        N = [a2.shape[0]//2 + 1, a2.shape[1]//2 + 1]
        NN = [2 * [N[0]], 2 * [N[1]]]
        a1 = jnp.pad(a1, NN, mode="wrap")
        return jax.scipy.signal.convolve2d(a1, a2, mode="same")[N[0] : -N[0], N[1] : -N[1]]

    @eqx.filter_jit
    def convolve_wrap_fft(self, react, diffuse):
        return jnp.fft.irfft2(
                   jnp.fft.rfft2(diffuse) * jnp.fft.rfft2(react)
                   , s=(react.shape[-2], react.shape[-1])
               )


    @eqx.filter_jit
    def __call__(
            self, x
    ):
        return self.NN(x)
#    @eqx.filter_jit
#    def __call__(
#            self, x, convolve_method=convolve_wrap
#    ):
#        react = self.nonlinear(x) # shape (N_x, N_y)
#
#        phi = x[0:1]
#        phi = self.lifting(phi)
#        phi = self.diffuseBlock(phi)
#        diffuse = (self.projection(phi) + self.bias).squeeze()
#
#        return convolve_method(self, react, diffuse) +react
