import time
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib


### One gem
def showSAV(phi_u_init, 
                    dt, 
                    solver, 
                    Delta=-0.3,
                    freq = 10, 
                    delay = 0,
                    save_path=None,
):

    phi_u_SAV = phi_u_init
    q_SAV = solver.energy(phi_u_SAV[0], phi_u_SAV[1])
    # Regularisation
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

    U_fields = []


    time_NN = 0
    time_SAV = 0

    fig, ax_list = plt.subplots(2, 5, figsize=(20, 7.5))
    matplotlib.rcParams.update({'font.size': 32})

    for i in range(4): 
        if i == 0 : 
            T = 0
            phi_SAV = phi_u_SAV[0]
        else:
            for _ in range(freq):
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
        T = dt*freq*i
        im = ax_list[0, i].imshow(phi_u_SAV[0], origin='lower', cmap="jet")
        ax_list[0, i].set_xticks([])
        ax_list[0, i].set_yticks([])
        ax_list[0, i].set_title(f"at T={T:.0f}")
        im = ax_list[1, i].imshow(phi_u_SAV[1], origin='lower', vmin=Delta, vmax=0)
        ax_list[1, i].set_xticks([])
        ax_list[1, i].set_yticks([])
        U_fields.append(phi_u_SAV[1])

    fig.text(-0.01, 0.70, r"$\phi$", va="center", ha="left", fontsize=48)
    fig.text(-0.01, 0.23, "U", va="center", ha="left", fontsize=48)
    fig.text(-0.03, 0.5, "Reference (SAV)", va="center", ha="center", fontsize=42, rotation=90)
    #fig.text(-0.03, 0.5, "SAV", va="center", ha="center", fontsize=48, rotation=90)
    #fig.text(-0.07, 0.5, "Reference", va="center", ha="center", fontsize=48, rotation=90)


    T = dt*freq*4
    for _ in range(freq):
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
    im_last = ax_list[0, 4].imshow(phi_u_SAV[0], origin='lower', cmap="jet")
    ax_list[0, 4].set_xticks([])
    ax_list[0, 4].set_yticks([])
    ax_list[0, 4].set_title(f"at T={T:.0f}")
    # Add colorbar without shrinking the image
    fig.colorbar(im_last, ax=ax_list[0, 4], fraction=0.046, pad=0.015)

    im_last_u = ax_list[1, 4].imshow(phi_u_SAV[1], origin='lower', vmin=Delta, vmax=0)
    U_fields.append(phi_u_SAV[1])
    ax_list[1, 4].set_xticks([])
    ax_list[1, 4].set_yticks([])
    fig.colorbar(im_last_u, ax=ax_list[1, 4], fraction=0.046, pad=0.015)
    plt.tight_layout()
#    plt.subplot(6, 1, 6)
#    plt.plot(dt*np.arange(len(q_checkpoint)), q_checkpoint, label='q_NN')
#    plt.plot(dt*np.arange(len(q_checkpoint)), q_SAV_checkpoint, label='q_SAV')
#    plt.legend()
#    plt.xlabel("Time")
#    plt.ylabel("Energy")
#    print(f"Computetime NN: {time_NN}| SAV {time_SAV}")
    if save_path:
        plt.savefig(f"{save_path}", bbox_inches='tight')
    return U_fields

def showNN(model, phi_u_init, 
                    model_name,
                    dt, 
                    solver, 
                    Delta=-0.3,
                    freq = 10, 
                    delay = 0,
                    train_name = 'DeepRitzSplit',
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
        


    time_NN = 0
    phi_u = phi_u_SAV
    q_SAV = solver.energy(phi_u_SAV[0], phi_u_SAV[1])
    q_checkpoint = [solver.energy(phi_u[0], phi_u[1])]


    time_NN = 0
    import matplotlib
    fig, ax_list = plt.subplots(2, 5, figsize=(20, 7.5))
    matplotlib.rcParams.update({'font.size': 32})

    for i in range(4): 
        if i ==0:
            ax_list[0, 0].axis("off")
            ax_list[1, 0].axis("off")
        if i > 0:
            for _ in range(freq):
                time_in = time.time()
                phi_u = model.compute_next(phi_u)
                time_out = time.time()
                q = solver.energy(phi_u[0], phi_u[1])
                q_checkpoint.append(q)
                time_NN += time_out - time_in
            T = dt*freq*i
            im = ax_list[0, i].imshow(phi_u[0], origin='lower', cmap="jet")
            ax_list[0, i].set_xticks([])
            ax_list[0, i].set_yticks([])
            ax_list[0, i].set_title(f"at T={T:.0f}")
            im = ax_list[1, i].imshow(phi_u[1], origin='lower')
            ax_list[1, i].set_xticks([])
            ax_list[1, i].set_yticks([])

    fig.text(-0.01, 0.70, r"$\phi$", va="center", ha="left", fontsize=48)
    fig.text(-0.01, 0.23, "U", va="center", ha="left", fontsize=48)
    fig.text(-0.03, 0.48, train_name+ r" $-$ "+ f"{model_name}", va="center", ha="center", fontsize=42, rotation=90)

    T = dt*freq*4
    for _ in range(freq):
        time_in = time.time()
        phi_u = model.compute_next(phi_u)
        time_out = time.time()
        q = solver.energy(phi_u[0], phi_u[1])
        q_checkpoint.append(q)
        time_NN += time_out - time_in
    im_last = ax_list[0, 4].imshow(phi_u[0], origin='lower', cmap="jet")
    ax_list[0, 4].set_xticks([])
    ax_list[0, 4].set_yticks([])
    ax_list[0, 4].set_title(f"at T={T:.0f}")
    # Add colorbar without shrinking the image
    fig.colorbar(im_last, ax=ax_list[0, 4], fraction=0.046, pad=0.015)

    im_last_u = ax_list[1, 4].imshow(phi_u[1], origin='lower', vmin=Delta, vmax=0)
    ax_list[1, 4].set_xticks([])
    ax_list[1, 4].set_yticks([])
    fig.colorbar(im_last_u, ax=ax_list[1, 4], fraction=0.046, pad=0.015)

    plt.tight_layout()
    if save_path:
        plt.savefig(f"{save_path}", bbox_inches='tight')
#plt.tight_layout()
#savefig_path  =model_path.replace(".eqx", "-benchmark-one-gem-NN-phi.png")
#plt.savefig(savefig_path,bbox_inches='tight')

#    plt.subplot(6, 5, 1+i)
#    plt.imshow(phi_u[0], extent=[x_lb, x_rb, y_lb, y_ub], origin='lower', cmap='jet' )
#    plt.title(f"phi_NN at T={T}")
#    plt.colorbar()
#    plt.subplot(6, 5, 6 + i)
#    plt.imshow(phi_u[1], extent=[x_lb, x_rb, y_lb, y_ub], origin='lower', )
#    plt.title(f"U at T={T}")
#    plt.colorbar()
#    plt.subplot(6, 5, 11 + i)
#    plt.imshow(phi_u_SAV[0], extent=[x_lb, x_rb, y_lb, y_ub], origin='lower', cmap = 'jet')
#    plt.title(f"phi_SAV at T={T}")
#    plt.colorbar()
#    plt.subplot(6, 5, 16 + i)
#    plt.imshow(phi_u_SAV[1], extent=[x_lb, x_rb, y_lb, y_ub], origin='lower', )
#    plt.title(f"U_SAV at T={T}")
#    plt.colorbar()
#    plt.subplot(6, 5, 21 + i)
#    plt.imshow(abs(phi_u_SAV[0] - phi_u[0]), extent=[x_lb, x_rb, y_lb, y_ub], origin='lower', cmap='rainbow')
#    plt.title(f"abs diff at T={T}")
#    plt.colorbar()
# plt.subplot(6, 1, 6)
# plt.plot(dt*np.arange(len(q_checkpoint)), q_checkpoint, label='q_NN')
# plt.plot(dt*np.arange(len(q_checkpoint)), q_SAV_checkpoint, label='q_SAV')
# plt.legend()
# plt.xlabel("Time")
# plt.ylabel("Energy")
# print(f"Computetime NN: {time_NN}| SAV {time_SAV}")
# def runSAV(solver, freq=10, delay=0): 

### Multi-gems figures
def showShort(model1, model2, phi_u_init, 
                    dt, 
                    solver, 
                    model_name1 = "RDNO",
                    model_name2 = "UNet",
                    freq = 10, 
                    delay = 0,
                    Delta = -0.3,
                    cluster_name = None,
                    save_path = None,
):

    # Track U max
    u_maxs = [0, 0, 0]

    # regularize phi_u_init
    q_init = solver.energy(phi_u_init[0], phi_u_init[1])

    for _ in range(delay):
        phi_init, u_init, q_init = solver.compute_next(1, phi_u_init[0], phi_u_init[1], q_init,
                                phi_u_init[0], phi_u_init[1],
                                phi_u_init[0], phi_u_init[1],
                                dt
                            )
        phi_u_init = jnp.stack([phi_init, u_init], axis =0)
        
    phi_u_SAV = phi_u_init
    q_SAV = solver.energy(phi_u_SAV[0], phi_u_SAV[1])
    q_checkpoint = [q_SAV]
    q_SAV_checkpoint = [q_SAV]

    # Draw initial condition
    import matplotlib
    fig, ax_list = plt.subplots(2, 4, figsize=(17, 7.5))
    matplotlib.rcParams.update({'font.size': 32})

    T = 0
    im = ax_list[0, 0].imshow(phi_u_SAV[0], origin='lower', cmap="jet")
    ax_list[0, 0].set_xticks([])
    ax_list[0, 0].set_yticks([])
    ax_list[0, 0].set_title(f"at T={T:.0f}")
    im = ax_list[1, 0].imshow(phi_u_SAV[1], origin='lower', vmin=Delta, vmax=0)
    ax_list[1, 0].set_xticks([])
    ax_list[1, 0].set_yticks([])

    # SAV
    time_SAV =0
    T = dt*freq
    for _ in range(freq):
        time_in = time.time()
        phi_SAV, u_SAV, q_SAV = solver.compute_next(1, phi_u_SAV[0], phi_u_SAV[1], q_SAV,
                                phi_u_SAV[0], phi_u_SAV[1],
                                phi_u_SAV[0], phi_u_SAV[1],
                                dt
                            )
        u_maxs[0] = max(u_maxs[0], jnp.max(u_SAV))
        phi_u_SAV = jnp.stack([phi_SAV, u_SAV], axis =0)
        time_out = time.time()
        q_SAV_checkpoint.append(q_SAV)
        time_SAV += time_out - time_in
    im_last = ax_list[0, 1].imshow(phi_u_SAV[0], origin='lower', cmap="jet")
    ax_list[0, 1].set_xticks([])
    ax_list[0, 1].set_yticks([])
    ax_list[0, 1].set_title(f"SAV at T={T:.0f}")
    # Add colorbar without shrinking the image
    #fig.colorbar(im_last, ax=ax_list[0, 1], fraction=0.046, pad=0.015)
    im_last_u = ax_list[1, 1].imshow(phi_u_SAV[1], origin='lower', vmin= Delta, vmax=0)
    ax_list[1, 1].set_xticks([])
    ax_list[1, 1].set_yticks([])
    #fig.colorbar(im_last_u, ax=ax_list[1, 1], fraction=0.046, pad=0.015)

        
    # NN 1
    time_NN = 0
    phi_u = phi_u_init
    q_checkpoint = [solver.energy(phi_u[0], phi_u[1])]


    T = dt*freq
    for _ in range(freq):
        time_in = time.time()
        phi_u = model1.compute_next(phi_u)
        u_maxs[1] = max(u_maxs[1], jnp.max(phi_u[1]))
        time_out = time.time()
        q = solver.energy(phi_u[0], phi_u[1])
        q_checkpoint.append(q)
        time_NN += time_out - time_in
    im_last = ax_list[0,2].imshow(phi_u[0], origin='lower', cmap="jet")
    ax_list[0, 2].set_xticks([])
    ax_list[0, 2].set_yticks([])
    #ax_list[0, 2].set_title(f" {model_name1} at T={T:.0f}")
    ax_list[0, 2].set_title(f" {model_name1}")
    # Add colorbar without shrinking the image
    #fig.colorbar(im_last, ax=ax_list[0,2], fraction=0.046, pad=0.015)

    im_last_u = ax_list[1,2].imshow(phi_u[1], origin='lower', vmin=Delta, vmax=0)
    ax_list[1, 2].set_xticks([])
    ax_list[1, 2].set_yticks([])
    #fig.colorbar(im_last_u, ax=ax_list[1,2], fraction=0.046, pad=0.015)

    # NN 2
    time_NN = 0
    phi_u = phi_u_init
    q_checkpoint = [solver.energy(phi_u[0], phi_u[1])]


    T = dt*freq
    for _ in range(freq):
        time_in = time.time()
        phi_u = model2.compute_next(phi_u)
        u_maxs[2] = max(u_maxs[2], jnp.max(phi_u[1]))
        time_out = time.time()
        q = solver.energy(phi_u[0], phi_u[1])
        q_checkpoint.append(q)
        time_NN += time_out - time_in
    im_last = ax_list[0,3].imshow(phi_u[0], origin='lower', cmap="jet")
    ax_list[0, 3].set_xticks([])
    ax_list[0, 3].set_yticks([])
    ax_list[0, 3].set_title(f" {model_name2}")
    # Add colorbar without shrinking the image
    fig.colorbar(im_last, ax=ax_list[0,3], fraction=0.046, pad=0.015)

    im_last_u = ax_list[1,3].imshow(phi_u[1], origin='lower', vmin=Delta, vmax=0)
    ax_list[1, 3].set_xticks([])
    ax_list[1, 3].set_yticks([])
    fig.colorbar(im_last_u, ax=ax_list[1,3], fraction=0.046, pad=0.015)

    fig.text(0.02, 0.70, r"$\phi$", va="center", ha="left", fontsize=48)
    fig.text(0.02, 0.23, "U", va="center", ha="left", fontsize=48)
    if cluster_name:
        fig.text(-0.03, 0.5, cluster_name, va="center", ha="center", fontsize=60, rotation=90)
    plt.tight_layout()
    if save_path:
        plt.savefig(f"{save_path}", bbox_inches='tight')
    print(f"Max Temperature {u_maxs}")



def showShortUdiff(model1, model2, phi_u_init, 
                    dt, 
                    solver, 
                    model_name1 = "RDNO",
                    model_name2 = "UNet",
                    freq = 10, 
                    delay = 0,
                    Delta = -0.3,
                    cluster_name = None,
                    save_path = None,
):

    # Track U max
    u_maxs = [0, 0, 0]

    # regularize phi_u_init
    q_init = solver.energy(phi_u_init[0], phi_u_init[1])

    for _ in range(delay):
        phi_init, u_init, q_init = solver.compute_next(1, phi_u_init[0], phi_u_init[1], q_init,
                                phi_u_init[0], phi_u_init[1],
                                phi_u_init[0], phi_u_init[1],
                                dt
                            )
        phi_u_init = jnp.stack([phi_init, u_init], axis =0)
        
    phi_u_SAV = phi_u_init
    q_SAV = solver.energy(phi_u_SAV[0], phi_u_SAV[1])
    q_checkpoint = [q_SAV]
    q_SAV_checkpoint = [q_SAV]

    matplotlib.rcParams.update({'font.size':20})
    # Draw initial condition
    fig, ax_list = plt.subplots(1, 4, figsize=(12, 3))

    T = 0
    im = ax_list[0].imshow(phi_u_SAV[1], origin='lower', vmin=Delta, vmax=0)
    ax_list[0].set_xticks([])
    ax_list[0].set_yticks([])
    ax_list[0].set_title(f"at T={T:.0f}")

    # SAV
    time_SAV =0
    T = dt*freq
    for _ in range(freq):
        time_in = time.time()
        phi_SAV, u_SAV, q_SAV = solver.compute_next(1, phi_u_SAV[0], phi_u_SAV[1], q_SAV,
                                phi_u_SAV[0], phi_u_SAV[1],
                                phi_u_SAV[0], phi_u_SAV[1],
                                dt
                            )
        u_maxs[0] = max(u_maxs[0], jnp.max(u_SAV))
        phi_u_SAV = jnp.stack([phi_SAV, u_SAV], axis =0)
        time_out = time.time()
        q_SAV_checkpoint.append(q_SAV)
        time_SAV += time_out - time_in
    im_last_u = ax_list[1].imshow(phi_u_SAV[1], origin='lower', vmin= Delta, vmax=0)
    ax_list[1].set_xticks([])
    ax_list[1].set_yticks([])
    ax_list[1].set_title(f"SAV at T={T:.0f}")
    # Add colorbar without shrinking the image
    #fig.colorbar(im_last, ax=ax_list[0, 1], fraction=0.046, pad=0.015)
    #fig.colorbar(im_last_u, ax=ax_list[1, 1], fraction=0.046, pad=0.015)

        
    # NN 1
    time_NN = 0
    phi_u = phi_u_init
    q_checkpoint = [solver.energy(phi_u[0], phi_u[1])]


    T = dt*freq
    for _ in range(freq):
        time_in = time.time()
        phi_u = model1.compute_next(phi_u)
        u_maxs[1] = max(u_maxs[1], jnp.max(phi_u[1]))
        time_out = time.time()
        q = solver.energy(phi_u[0], phi_u[1])
        q_checkpoint.append(q)
        time_NN += time_out - time_in
    im_last_u = ax_list[2].imshow(phi_u[1]-phi_u_SAV[1], origin='lower', vmin=0, vmax=0.1)
    ax_list[2].set_xticks([])
    ax_list[2].set_yticks([])
    ax_list[2].set_title(r"${U_{RDNO}}$-"+r"$U_{SAV}$")
    # Add colorbar without shrinking the image
    #fig.colorbar(im_last, ax=ax_list[2], fraction=0.046, pad=0.015)


    # NN 2
    time_NN = 0
    phi_u = phi_u_init
    q_checkpoint = [solver.energy(phi_u[0], phi_u[1])]


    T = dt*freq
    for _ in range(freq):
        time_in = time.time()
        phi_u = model2.compute_next(phi_u)
        u_maxs[2] = max(u_maxs[2], jnp.max(phi_u[1]))
        time_out = time.time()
        q = solver.energy(phi_u[0], phi_u[1])
        q_checkpoint.append(q)
        time_NN += time_out - time_in
    im_last_u = ax_list[3].imshow(phi_u[1]-phi_u_SAV[1], origin='lower', vmin=0.0, vmax=0.1)
    ax_list[3].set_xticks([])
    ax_list[3].set_yticks([])
    ax_list[3].set_title(r"${U_{UNet}}$-"+r"$U_{SAV}$")
    # Add colorbar without shrinking the image
    fig.colorbar(im_last_u, ax=ax_list[3], fraction=0.046, pad=0.015)

#    fig.text(0.02, 0.70, r"$\phi$", va="center", ha="left", fontsize=48)
    fig.text(0.00, 0.5, r"${U_{{diff}}}$", va="center", ha="left", fontsize=28, rotation=90)
    if cluster_name:
        fig.text(-0.03, 0.5, cluster_name, va="center", ha="center", fontsize=36, rotation=90)
    plt.tight_layout()
    if save_path:
        plt.savefig(f"{save_path}", bbox_inches='tight')
    print(f"Max Temperature {u_maxs}")