import numpy as np
from functools import partial 
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import time 
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
from tqdm import trange
import sys

import warnings
warnings.simplefilter('ignore', np.exceptions.RankWarning)



def regularizationBySAV(phi_u_init, solver, dt, delay=50):
    phi_init, u_init = phi_u_init[0], phi_u_init[1]
    q_init =solver.energy(phi_init, u_init)
    for _ in range(delay):
        phi_init, u_init, q_init  = solver.compute_next(
                        1, phi_init, u_init, q_init,
                        phi_init, u_init,
                        phi_init, u_init,
                        dt
        )
    phi_u_init = jnp.stack([phi_init, u_init])
    return phi_u_init

def choice_pts(n_gems, thickness,
                x_lb = 0, x_rb = 1, y_lb = 0, y_ub=1, 
                ecart = 15, 
                *,
                pointkey
                ): 
    points = jr.uniform(pointkey, (n_gems, 2), 
                        minval=np.array([
                            [x_lb + ecart*thickness, y_lb + ecart*thickness]
                        ]), 
                        maxval= np.array([
                            [x_rb - ecart*thickness, y_ub - ecart*thickness]
                        ]))
    return points


### Do not work well
# @eqx.filter_jit
# def curvature_div(phi) :
#     n = normal(phi)
#     n1 = n[..., 0]
#     n2 = n[..., -1]
#     return fft_x(n1) + fft_y(n2)

def curvature_AC(phi, eps, Laplacian):
    #return -(eps*Laplacian(phi) - (phi**3 - phi)/eps)/(2*np.sqrt(2)/3)
    return -(eps*Laplacian(phi) - (phi**3 - phi)/eps)*np.sqrt(2)
    
@eqx.filter_jit
def tip_idx(phi, j):
    N = phi.shape[0]
    #abs_phi= np.abs(phi[N//2:, N//2])
    #idx_tip = np.argmin(abs_phi) + (N//2)
    idx_tip = N//2

    def cond_fun(idx): 
        return  (phi[idx, N//2+j]*phi[idx+1, N//2+j] > 0)
    idx_tip = jax.lax.while_loop(
        cond_fun,
        lambda x : x+1, idx_tip
    )
    return jnp.min(jnp.array([idx_tip, N-1]))

    
@eqx.filter_jit
def tip_position(phi,  xs_grid, j =0, deg=4):
    # phi is an array of N*N where N is even
    """ 
    Use deg+1 points close to the tip along x axis at y = 0
    then interpolate to find x_tip such that phi(x_tip, 0) = 0
    """

    N = phi.shape[0]
    #jlist = jnp.arange(-deg//2, (deg//2) +1) 
    idx_tip = tip_idx(phi, j)
    idx_list = idx_tip +jnp.arange(-(deg//2), deg +1- (deg//2))
    print(idx_list)
    xs_fit = xs_grid[idx_list, N//2]
    phis_fit = phi[idx_list, N//2]

    poly = jnp.polyfit(
        xs_fit, 
        phis_fit,
        deg=deg
    );
    roots = jnp.roots(poly, strip_zeros=False)
    roots = jnp.where(
        jnp.isreal(roots),
        roots.real, jnp.float64('inf'))
    roots = jnp.where(
        roots>= xs_grid[idx_tip, N//2],
        roots, 
        jnp.float64('inf')
    ) 
    #roots = jnp.where(
    #    roots<= xs_grid[idx_tip+1, N//2],
    #    roots, 
    #    jnp.float64('inf')
    #) 

    return jnp.min(roots)
    # otherwise just linear function between idx_tip and idx_tip +1
    return tip_position(phi, xs_grid, deg = 1).real


@eqx.filter_jit
def tip_curvature_fit(phi, xs_grid, ys_grid, curvature_function, deg=4, kx=5, ky=5):
    N = phi.shape[0]
    curvature = curvature_function(phi) # shape N x N 
    x_tip = tip_position(phi, xs_grid=xs_grid, j=0)
    idx_tip = tip_idx(phi, j=0)
    idx_list = idx_tip +jnp.arange(-(deg//2), deg +1- (deg//2))
    vertical_idx = N//2 +jnp.arange(-(deg//2), deg +1- (deg//2))
    curvs_fit = curvature[idx_list][:, vertical_idx]

    xs_fit = xs_grid[idx_list, N//2]
    ys_fit = ys_grid[idx_tip, vertical_idx]

    poly = polyfit2d(
        xs_fit, ys_fit, curvs_fit, kx=kx, ky = ky
    )[0].reshape(kx+1, ky+1)

    x_val = jnp.array([ x_tip **k for k in range(kx+1)]).reshape(kx+1, -1)
    x_val = jnp.repeat(x_val, repeats = ky+1, axis=1)
    y_val = jnp.array([ys_grid[N//2,N//2]**k for k in range(ky+1) ]).reshape(1, ky+1)
    y_val = jnp.repeat(y_val, repeats=kx+1, axis=0)
    #print("numpy polyval", np.polynomial.polynomial.polyval2d(x_tip, ys_grid[N//2, N//2], poly))
    return jnp.sum( poly * x_val * y_val)
    #poly = jnp.polyfit(
    #    xs_fit,
    #    curvs_fit,
    #    deg=deg
    #)
    ## print(x_tip)
    #return  jnp.polyval(poly, x_tip)


@partial(jax.jit, static_argnums=(2,))
def tip_velocity(tip_position_list, dt, C = 1): 
    return jnp.gradient(jnp.array(tip_position_list)[::C], C*dt)

def polyfit2d(x, y, z, kx=3, ky=3, order=None):
    '''
    Two dimensional polynomial fitting by least squares.
    Fits the functional form f(x,y) = z.

    Notes
    -----
    Resultant fit can be plotted with:
    np.polynomial.polynomial.polygrid2d(x, y, soln.reshape((kx+1, ky+1)))

    Parameters
    ----------
    x, y: array-like, 1d
        x and y coordinates.
    z: np.ndarray, 2d
        Surface to fit.
    kx, ky: int, default is 3
        Polynomial order in x and y, respectively.
    order: int or None, default is None
        If None, all coefficients up to maxiumum kx, ky, ie. up to and including x^kx*y^ky, are considered.
        If int, coefficients up to a maximum of kx+ky <= order are considered.

    Returns
    -------
    Return paramters from np.linalg.lstsq.

    soln: np.ndarray
        Array of polynomial coefficients.
    residuals: np.ndarray
    rank: int
    s: np.ndarray

    '''

    # grid coords
    x, y = jnp.meshgrid(x, y, indexing='ij')
    # coefficient array, up to x^kx, y^ky
    coeffs = jnp.ones((kx+1, ky+1))

    # solve array
    a = jnp.zeros((coeffs.size, x.size))

    # for each coefficient produce array x^i, y^j
    for index, (i, j) in enumerate(np.ndindex(coeffs.shape)):
        # do not ikclude powers greater than order
        if order is not None and i + j > order:
            arr = jnp.zeros_like(x)
        else:
            arr = coeffs[i, j] * x**i * y**j
        #a[index] = arr.ravel()
        a  =a.at[index].set(arr.ravel())

    # do leastsq fitting and return leastsq result
    return jnp.linalg.lstsq(a.T, jnp.ravel(z), rcond=None)
@eqx.filter_jit
def tip_curvature_KR(phi, xs_grid, ys_grid, deg = 4): 
    N = phi.shape[1]
    idx_tip = tip_idx(phi, j=0)
    idx_list = idx_tip +jnp.arange(-(deg//2), deg +1- (deg//2))
    idy_list = N//2 + jnp.arange(-2, 3)
    xs_fit = xs_grid[idx_list, N//2]
    phis_fit = phi[idx_list, N//2]
    poly = jnp.polyfit(
        xs_fit, phis_fit, deg=deg
    )
    x_tip = tip_position(phi, xs_grid=xs_grid, j=0, deg=deg)

    poly_x = jnp.polyder(
        poly, m = 1
    )
    # phi_x at tip
    phi_x = jnp.polyval(poly_x, x_tip)

    l = []
    for i in range(len(idx_list)): 
        idx_i = idx_list[i]
        phis_fit_i = phi[idx_i, idy_list]
        ys_fit_i = ys_grid[idx_i, idy_list]
        poly = jnp.polyfit(
            ys_fit_i,  
            phis_fit_i, 
            deg = deg
        )
        poly_yy = jnp.polyder(poly, m = 2)
        l.append (jnp.polyval(poly_yy, ys_grid[N//2, N//2])) #(phi_yy at z_i)
    l = jnp.array(l)    
    poly = jnp.polyfit(
        xs_fit,
        l,
        deg=deg,
    )
    phi_yy = jnp.polyval(poly, x_tip)
    return phi_yy/phi_x
    
#
def tip_curvature_AC(phi, xs_grid, ys_grid, curvature_AC, deg =4):
    N = phi.shape[0]
    idx_tip = tip_idx(phi, j=0)
    curvature = curvature_AC(phi) # shape N x N 
    x_tip = tip_position(phi, xs_grid=xs_grid)
    idx_tip = tip_idx(phi, j=0)
    idx_list = idx_tip +jnp.arange(-(deg//2), deg +1- (deg//2))
    poly = np.polyfit(
        xs_grid[idx_list, N//2], 
        curvature[idx_list, N//2], 
        deg=deg
    )
    # print(x_tip)
    return  np.polyval(poly, x_tip)
#
# def tip_curvature_AC(phi, xs_grid, ys_grid, curvature_AC, deg = 4, degy= 5): 
#      N = phi.shape[1]
#      idx_tip = tip_idx(phi, j=0)
#      idx_list = idx_tip +jnp.arange(-(deg//2), deg +1- (deg//2))
#      idy_list = N//2 + jnp.arange(-(degy//2), degy +1- (degy//2))
#      xs_fit = xs_grid[idx_list, N//2]
#      phis_fit = phi[idx_list, N//2]
#      x_tip = tip_position(phi, xs_grid=xs_grid, j=0, deg=deg)
#  
#  
#      l = []
#      for i in range(len(idx_list)): 
#          idx_i = idx_list[i]
#          curvs_fit_i = curvature_AC(phi)[idx_i, idy_list]
#          ys_fit_i = ys_grid[idx_i, idy_list]
#          poly = jnp.polyfit(
#              ys_fit_i,  
#              curvs_fit_i, 
#              deg = degy
#          )
#          #poly_yy = jnp.polyder(poly, m = 2)
#          l.append (jnp.polyval(poly, ys_grid[N//2, N//2])) #(phi_yy at z_i)
#      l = jnp.array(l)    
#      poly = jnp.polyfit(
#          xs_fit,
#          l,
#          deg=deg,
#      )
#      return jnp.polyval(poly, x_tip)

# def tip_curvature_fit(phi, xs_grid, ys_grid):
# 
#     N = phi.shape[0]
#     idx_tip = tip_idx(phi)
#     x_tip = tip_position(phi, xs_grid)
#     p = phi[idx_tip-1:idx_tip+3, (N//2 -1):(N//2+3)]
#         
#     poly = polyfit2d(
#         x = xs_grid[idx_tip-1:idx_tip+3, N//2], 
#         y=ys_grid[idx_tip, (N//2 - 1):(N//2 +3)],
#         z=p, 
#         kx = 3, ky = 3
#     )[0].reshape(3+1, 3+1)
#     dpdx = np.polynomial.polynomial.polyder(
#         c= poly,
#         m =1,
#         axis=0
#     )
#     d2pdy2 = np.polynomial.polynomial.polyder(
#         c= poly,
#         m =2,
#         axis=1
#     )
#     #return np.polynomial.polynomial.polyval2d(x_tip, ys[N//2], d2pdy2)
#     return np.polynomial.polynomial.polyval2d(x_tip, ys_grid[idx_tip, N//2], d2pdy2)/np.polynomial.polynomial.polyval2d(x_tip, ys[idx_tip, N//2], dpdx)
 

def pf_perimeter(phi, eps, W, fft_x, fft_y, quad, Area = 1):
    phi_x = fft_x(phi)
    phi_y = fft_y(phi)
    integrand = eps*(phi_x**2 + phi_y**2)/2  + W(phi)/(eps)

    return Area*quad(
        integrand
    )

def NN_tip_position_list(model, phi_u_init, iter, energy, 
                         xs_grid, quad, tip_curvature, 
                         Area = 1, compute_u_next =None):
    phi_u = phi_u_init
    tip_position_list = []
    tip_curvature_list = []
    phi_u_list = [phi_u]
    q_list = [energy(phi_u)]
    ratio_solid = []
    per_list = []


    time_NN = 0

    for i in range(iter): 
        time_in = time.time()

        #phi_tilde = model(phi_u)
        #u_tilde = compute_u_next(phi_u[1], phi_u[0], phi_tilde)
        #phi_u = np.stack([phi_tilde, u_tilde], axis=0)
        phi_u = model.compute_next(phi_u)

        time_out = time.time()
        time_NN += time_out - time_in

        q_list.append(model.energy(phi_u))
        tip_position_list.append(tip_position(phi_u[0], xs_grid=xs_grid))
        tip_curvature_list.append(tip_curvature(phi_u[0]))
        ratio_solid.append(Area*quad((phi_u[0]+1) / 2))
        per_list.append(pf_perimeter(phi_u[0]))

        if (i+1) % (iter // 4) == 0:
            phi_u_list.append(phi_u)

    print(f"Computetime NN: {time_NN}")

    return {
        "phi_u" : phi_u_list,
        "q_list" : q_list,
        "tip_curvature" : tip_curvature_list, 
        "tip_position" : tip_position_list,  
        "ratio_solid" : ratio_solid,
        "perimeter" : per_list,
    }

def makeVideo(phi_u_init, model, solver, dt,
              x_lb = 0., x_rb = 1., y_lb = 0., y_ub = 1., 
              freq = 800*4, interval = 30, delay = 0,
              moviepath = None):
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
    frames = [] # for storing the generated images
    fig = plt.figure()
    for i in range(freq):
        frame = plt.imshow(phi_u[0], cmap='jet',animated=True, extent=[x_lb, x_rb, y_lb, y_ub], origin='lower', vmin=-1, vmax=1)
        #frames.append([plt.imshow(f, cmap='rainbow',animated=True, extent=[x_lb, x_rb, y_lb, y_ub], origin='lower', vmin=-1, vmax=1)])
        frames.append([frame])
        phi_u = model.compute_next(phi_u)

    ani = animation.ArtistAnimation(fig, frames, interval=interval, blit=True,
                                    repeat_delay=1000)
    if moviepath:                                    
        ani.save(moviepath)
    #plt.show()
def makeVideoSAV(phi_u_init, solver, dt, 
                 x_lb = 0., x_rb = 1., y_lb = 0., y_ub = 1., 
                 freq = 800*4, interval = 30, delay=0, 
                 moviepath = None
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
    frames = [] # for storing the generated images
    fig = plt.figure()
    for i in range(freq):
        frame = plt.imshow(phi_u_SAV[0], cmap='jet',animated=True, extent=[x_lb, x_rb, y_lb, y_ub], origin='lower', vmin=-1, vmax=1)
        frames.append([frame])
        phi_SAV, u_SAV, q_SAV = solver.compute_next(1, phi_u_SAV[0], phi_u_SAV[1], q_SAV,
                                phi_u_SAV[0], phi_u_SAV[1],
                                phi_u_SAV[0], phi_u_SAV[1],
                                dt
                            )
        phi_u_SAV = jnp.stack([phi_SAV, u_SAV], axis =0)

    ani = animation.ArtistAnimation(fig, frames, interval=interval, blit=True,
                                    repeat_delay=1000)
    if moviepath:                                    
        ani.save(moviepath)


def verificationPeclet(dic_NN, dic_SAV, dt, D, Delta, savepath=None, C=2):

    import math
    P = 0.19 * (-Delta/(1+Delta))**(1.7)
    def f(Pe): 
        return np.sqrt(np.pi*Pe) * np.exp(Pe) * math.erfc(np.sqrt(Pe)) + Delta

    from scipy.optimize import newton
    Pe = newton(f, 0.04)
    print(f"Theoretical Peclet Number : P = {P}, Pe  = {Pe}")

    tip_NN = dic_NN['tip_position']
    tip_curvature_NN = dic_NN['tip_curvature']
    tip_SAV = dic_SAV['tip_position']
    tip_curvature_SAV = dic_SAV['tip_curvature']

    plt.figure(figsize=(12, 12))

    ## Tip Speed 
    plt.subplot(2, 1, 1)
    v_NN = tip_velocity(tip_NN, dt = dt, C=C)
    v_SAV = tip_velocity(tip_SAV, dt = dt, C=C)
    plt.plot(dt*C*jnp.arange(1, len(v_NN)+1), v_NN, label='v_NN')
    plt.plot(dt*C*jnp.arange(1, len(v_SAV)+1), v_SAV, label='v_SAV')
    plt.xlabel("Time")
    plt.ylabel("Tip Velocity")
    plt.legend();
    v_NN_mean = jnp.mean(v_NN[len(v_NN)//2:])
    v_SAV_mean = jnp.mean(v_SAV[len(v_SAV)//2:])

    print(f"Average Tip Velociy : v_NN = {v_NN_mean} | v_SAV = {v_SAV_mean} | rel error = {(v_NN_mean - v_SAV_mean)/v_SAV_mean:.2%}")

    Pe_NN = v_NN/(2*D*np.array(tip_curvature_NN)[::C])
    Pe_SAV = v_SAV/(2*D*np.array(tip_curvature_SAV)[::C])
    Pe_NN_mean = jnp.mean(Pe_NN[len(Pe_NN)//2:])
    Pe_SAV_mean = jnp.mean(Pe_SAV[len(Pe_SAV)//2:])
    plt.subplot(2, 2, 3)
    plt.plot(dt*C*jnp.arange(1, len(v_NN)+1), np.clip(Pe_NN, 0, 10*Pe), label="NN")
    plt.plot(dt*C*jnp.arange(1, len(v_SAV)+1), np.clip(Pe_SAV, 0, 10*Pe), label="SAV")
    plt.plot(dt*C*jnp.arange(1, len(v_SAV)+1), Pe*np.ones_like(v_SAV), label="Peclet")
    plt.xlabel('Time')
    plt.ylabel('rho * V /(2D)')
    plt.title(f"Pe = {Pe}")
    plt.legend();
    print(f"Average Péclet Number : Pe_NN = {Pe_NN_mean} | Pe_SAV = {Pe_SAV_mean} | rel error = {(Pe_NN_mean - Pe_SAV_mean)/Pe_SAV_mean:.2%}")

    rho2v_NN= v_NN/(np.array(tip_curvature_NN)[::C])**2
    rho2v_SAV= v_SAV/(np.array(tip_curvature_SAV)[::C])**2
    rho2v_NN_mean = jnp.mean(rho2v_NN[len(rho2v_NN)//2:])
    rho2v_SAV_mean = jnp.mean(rho2v_SAV[len(rho2v_SAV)//2:])
    plt.subplot(2, 2, 4)
    plt.plot(dt*C*jnp.arange(1, len(v_NN)+1), np.clip(rho2v_NN, 0, 1e-4), label="NN")
    plt.plot(dt*C*jnp.arange(1, len(v_SAV)+1), np.clip(rho2v_SAV, 0, 1e-4), label="SAV")
    plt.xlabel('Time')
    plt.ylabel('rho^2 V ')
    plt.legend();
    print(f"Average rho2V : rho2v_NN = {rho2v_NN_mean} | rho2v_SAV = {rho2v_SAV_mean} | rel error = {(rho2v_NN_mean - rho2v_SAV_mean)/rho2v_SAV_mean:.2%}")
    if savepath:
        plt.savefig(f"{savepath}", bbox_inches='tight')


def SAV_tip_position_list(phi_u_init, solver, dt, iter, quad, tip_curvature, Area=1):

    phi_u_SAV = phi_u_init
    q_SAV = solver.energy(phi_u_SAV[0], phi_u_SAV[1])
    phi_u_list = [phi_u_init]
    tip_SAV_list = []
    tip_curvature_list = []
    q_list = [q_SAV]
    ratio_solid = []
    per_list = []

    time_SAV = 0

    for i in range(iter):
        time_in = time.time()

        phi_SAV, u_SAV, q_SAV = solver.compute_next(1, phi_u_SAV[0], phi_u_SAV[1], q_SAV,
                                phi_u_SAV[0], phi_u_SAV[1],
                                phi_u_SAV[0], phi_u_SAV[1],
                                dt
                            )
        phi_u_SAV = jnp.stack([phi_SAV, u_SAV], axis =0)

        time_out = time.time() 
        time_SAV += time_out - time_in
        q_list.append(solver.energy(phi_u_SAV[0], phi_u_SAV[1]))
        tip_SAV_list.append(tip_position(phi_SAV))
        tip_curvature_list.append(tip_curvature(phi_SAV))
        ratio_solid.append(Area*quad((phi_u_SAV[0]+1) / 2))
        per_list.append(pf_perimeter(phi_u_SAV[0]))
        if (i+1) % (iter // 4) == 0:
            phi_u_list.append(phi_u_SAV)


    return {
        "phi_u" : phi_u_list,
        "q_list" : q_list,
        "tip_curvature" : tip_curvature_list, 
        "tip_position" : tip_SAV_list,  
        "ratio_solid" : ratio_solid,
        "perimeter" : per_list,
    }

def benchmark_NN_SAV(
        NN_dic, SAV_dic, dt, save_path =None,
): 
    phi_u_NN = NN_dic['phi_u']
    q_NN_list = NN_dic['q_list']
    phi_u_SAV = SAV_dic['phi_u']
    q_SAV_list = SAV_dic['q_list']
    ratio_NN = NN_dic['ratio_solid']
    ratio_SAV = SAV_dic['ratio_solid']
    per_NN = NN_dic['perimeter']
    per_SAV = SAV_dic['perimeter']

    plt.figure(figsize=(12,12))
    plt.subplot(2, 2, 1)
    plt.plot(dt*np.arange(len(q_NN_list)), q_NN_list, label='energy_NN')
    plt.plot(dt*np.arange(len(q_SAV_list)), q_SAV_list, label='energy_SAV')
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Energy")
    plt.subplot(2, 2, 2)
    plt.plot(dt*np.arange(len(q_NN_list)-1), ratio_NN, label='ratio_NN')
    plt.plot(dt*np.arange(len(q_SAV_list)-1),ratio_SAV, label='ratio_SAV')
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Ratio-Solid")
    plt.subplot(2, 2, 3)
    plt.plot(dt*np.arange(len(q_NN_list)-1), per_NN, label='peri_NN')
    plt.plot(dt*np.arange(len(q_SAV_list)-1),per_SAV, label='peri_SAV')
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Perimeter")
    plt.legend()
    tip_NN = NN_dic['tip_position']
    tip_SAV = SAV_dic['tip_position']
    plt.subplot(2, 2, 4)
    plt.plot(dt*jnp.arange(1, len(tip_NN)+1), tip_NN, label='Tip NN')
    plt.plot(dt*jnp.arange(1, len(tip_SAV)+1), tip_SAV, label='Tip SAV')
    plt.xlabel("Time")
    plt.ylabel("Tip Position")
    plt.legend()
    plt.savefig(f"{save_path}", bbox_inches='tight')


    timelist =dt*np.arange(len(q_SAV_list)) 
    rel_freq = len(timelist)//4

    for i in range(1, 5): 
        it = i*rel_freq if i !=4 else -1
        print(f"===============at T={timelist[it]} ==========")
        print(f"Rel of Energy NN/SAV is {(q_NN_list[it]-q_SAV_list[it])/q_SAV_list[it]:.2%}  at {timelist[it]}")
        print(f"Rel of Ratio-Solid NN/SAV is {(ratio_NN[it]-ratio_SAV[it])/ratio_SAV[it]:.2%} at {timelist[it]}")
        print(f"Rel of Perimeter NN/SAV is {(per_NN[it]-per_SAV[it])/per_SAV[it]:.2%} at {timelist[it]}")
        print(f"Rel of Tip position NN/SAV is {(tip_NN[it]-tip_SAV[it])/tip_SAV[it]:.2%} at {timelist[it]}")