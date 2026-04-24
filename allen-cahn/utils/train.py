from tqdm import trange 
import numpy as np
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import sys 



def train_phys(
        model, 
        h5dl, 
        step,
        opt_state,
        n_sample, n_phys = 32, n_epoch = 100,
        n_log = 10,
        n_save = 20,
        model_path = None, 
        *,
        key
    ):

    pbar = trange(n_epoch, dynamic_ncols=True, file=sys.stdout, leave=True, position=0)
    best = float('inf')
    idx_train = np.arange(n_sample)
    n_batch = n_sample //n_phys 
    loss_history = []
    early_stopper = 0

    for epoch in pbar:
        batch_loss = 0
        key, train_key = jr.split(key)
        idx_phys = jr.choice(train_key, idx_train, shape=(n_batch, n_phys), replace=False)
        idx_phys = idx_phys.sort(axis=-1)
        #numpy_dl = NumpyLoader(h5dl, batch_size = n_phys, num_workers=8, pin_memory=True, shuffle=True, collate_fn=numpy_collate)

        for batch in range(n_batch):
        #    dl_batch = next(iter(numpy_dl))
        #    phi_u_inits = dl_batch[:, 0]
        #    phi_u_nexts = dl_batch[:, 1]

            phi_u_inits = h5dl[idx_phys[batch], 0]
            # phi_u_nexts = h5dl[idx_phys[batch], 1]
            #u_exact_data = u_fdm[idx_data[batch], 1]



            if (batch != (n_batch -1)) or (epoch <= 5/6*n_epoch):
                model, opt_state, weighted_loss = step(model, opt_state, phi_u_inits)
            #    #if (epoch +1 ) % (n_epoch//6) == 0: 
            #    #    key, r3key = jr.split(key)
            #    #    coloc_x, coloc_y = r3_resampling(model, r3key)
            else: 
                # Save best performed hyper-parameters
                model_new, opt_state, weighted_loss = step(model, opt_state, phi_u_inits)

            batch_loss += weighted_loss

        batch_loss /= n_batch
        loss_history.append(batch_loss.item())


        ### Early Stopper

        # Res = loss_history[-1] - loss_history[-2]
        # if (epoch> 1/3*n_epoch) and (Res >=0): 
        #     if abs(abs)

        #     if model_path:
        #         eqx.tree_serialise_leaves(f"{model_path}.eqx", model)
        #     print(f"Early Stopping at Epoch {epoch} | Train Loss {loss_history[-1]}")
        #     return loss_history


        if (epoch > 5/6*n_epoch):
            if (batch_loss <= best):
                best_model = model
                best_epoch = epoch
                best = batch_loss

            model = model_new

            if (model_path) and (n_save) and (epoch+1 % n_save == 0):
                eqx.tree_serialise_leaves(f"{model_path}.eqx", model)



        # Logging
        if (epoch+1) % n_log == 0:
        #    #L_t, W = residus_and_weights(model, tol)
            pbar.set_postfix({"Epoch" : epoch , 'Loss': batch_loss,}) 
        #    #if min(W) >=0.99:
        #    #    break


    # Save hyperparameters
    if 'best_model' in locals():
        model = best_model
    if model_path:
        eqx.tree_serialise_leaves(f"{model_path}.eqx", model)
    print(f"Best Epoch {best_epoch} | Train Loss {loss_history[best_epoch]}")
    return loss_history, model 


def train_data(
        model, 
        h5dl, 
        step,
        opt_state,
        n_sample, n_phys = 32, n_epoch = 100,
        n_log = 10,
        n_save = 20,
        model_path = None, 
        *,
        key
    ):

    pbar = trange(n_epoch, dynamic_ncols=True, file=sys.stdout, leave=True, position=0)
    best = float('inf')
    idx_train = np.arange(n_sample)
    n_batch = n_sample //n_phys 
    loss_history = []

    for epoch in pbar:
        batch_loss = 0
        key, train_key = jr.split(key)
        idx_data = jr.choice(train_key, idx_train, shape=(n_batch, n_phys), replace=False)
        idx_data = idx_data.sort(axis=-1)

        for batch in range(n_batch):
            phi_u_inits = h5dl[idx_data[batch], 0]
            phi_u_nexts = h5dl[idx_data[batch], 1]

            if (batch != (n_batch -1)) or (epoch <= 5/6*n_epoch):
                model, opt_state, weighted_loss = step(model, opt_state, phi_u_inits, phi_u_nexts)
            else: 
                model_new, opt_state, weighted_loss = step(model, opt_state, phi_u_inits, phi_u_nexts)

            batch_loss += weighted_loss

        batch_loss /= n_batch
        loss_history.append(batch_loss.item())

        if (epoch > 5/6*n_epoch):
            if (batch_loss <= best):
                best_model = model
                best_epoch = epoch
                best = batch_loss

            model = model_new

            if (model_path) and (n_save) and (epoch+1 % n_save == 0):
                eqx.tree_serialise_leaves(f"{model_path}.eqx", model)

        # Logging
        if (epoch+1) % n_log == 0:
            pbar.set_postfix({"Epoch" : epoch , 'Loss': batch_loss,}) 

    # Save hyperparameters
    if 'best_model' in locals():
        model = best_model
    if model_path:
        eqx.tree_serialise_leaves(f"{model_path}.eqx", model)
    print(f"Best Epoch {best_epoch} | Train Loss {loss_history[best_epoch]}")
    return loss_history, model 
