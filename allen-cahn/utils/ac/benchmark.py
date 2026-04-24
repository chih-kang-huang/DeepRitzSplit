import time
import matplotlib 
import matplotlib.pyplot as plt

def eval_num(f_init, dt, DR_next, energy, freq = [0, 10, 20, 30, 40], r= 90, title_name = None, entete =None, savepath = None): 
    u_num = f_init
    energy_num = [energy(u_num)]
    u_sols = [f_init]

    time_in = time.time()
    matplotlib.rcParams.update({'font.size': 20})

    fig, ax_list = plt.subplots(1, 5, figsize=(15, 3))

    # Plot first 4 images (no colorbar)
    for i in range(4):
        T= dt*freq[i]
        im = ax_list[i].imshow(u_num, cmap='rainbow', origin='lower', vmin=-1, vmax=1)
        ax_list[i].set_xticks([])
        ax_list[i].set_yticks([])
        ax_list[i].set_title(f"T={T:.2e}")

        freqdiff = freq[i+1]-freq[i]
        for _ in range(freqdiff):
            u_num = DR_next(u_num)
            energy_num.append(energy(u_num))
            u_sols.append(u_num)
        
    time_out = time.time()
    # Plot last image + colorbar
    T= dt*freq[4] 
    im_last = ax_list[4].imshow(u_num, cmap='rainbow', origin='lower', vmin=-1, vmax=1)
    ax_list[4].set_xticks([])
    ax_list[4].set_yticks([])
    ax_list[4].set_title(f"T={T:.2e}")

    # Add colorbar without shrinking the image
    fig.colorbar(im_last, ax=ax_list[4], fraction=0.046, pad=0.04)

    if title_name: 
        fig.text(0.00, 0.5, title_name, va="center", ha="center", fontsize=32, rotation=r)
    if entete:
        fig.text(-0.04, 0.5, entete, va="center", ha="center", fontsize=28, rotation=r)
    plt.tight_layout()

    if savepath:
        plt.savefig(savepath, bbox_inches='tight')
    print(f"Eval time : {time_out-time_in}")
    return u_sols

def eval_NN(f_init, model, dt, energy, freq = [0, 10, 20, 30, 40], title_name = None, method_name="DeepRitzSplit", savepath = None): 

    time_in = time.time()
    u_num = f_init
    energy_num = [energy(u_num)]
    u_sols = [f_init]

    matplotlib.rcParams.update({'font.size': 20})

    fig, ax_list = plt.subplots(1, 5, figsize=(15, 3))

    # Plot first 4 images (no colorbar)
    for i in range(4):
        T= dt*freq[i]
        im = ax_list[i].imshow(u_num, cmap='rainbow', origin='lower', vmin=-1, vmax=1)
        ax_list[i].set_xticks([])
        ax_list[i].set_yticks([])
        ax_list[i].set_title(f"T={T:.2e}")
        freqdiff = freq[i+1]-freq[i]
        for _ in range(freqdiff):
            u_num = model(u_num)
            energy_num.append(energy(u_num))
            u_sols.append(u_num)
    time_out =  time.time()
    # Plot last image + colorbar
    T= dt*freq[4] 
    im_last = ax_list[4].imshow(u_num, cmap='rainbow', origin='lower', vmin=-1, vmax=1)
    ax_list[4].set_xticks([])
    ax_list[4].set_yticks([])
    ax_list[4].set_title(f"T={T:.2e}")

    # Add colorbar without shrinking the image
    fig.colorbar(im_last, ax=ax_list[4], fraction=0.046, pad=0.04)

    if title_name: 
        fig.text(-0.04, 0.5, method_name, va="center", ha="center", fontsize=28, rotation=90)
        fig.text(0.00, 0.5, title_name, va="center", ha="center", fontsize=32, rotation=90)
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, bbox_inches='tight')
    print(f"Eval time : {time_out - time_in}")
    return u_sols