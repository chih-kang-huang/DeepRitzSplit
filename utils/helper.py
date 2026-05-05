import os, sys
from tqdm import tqdm, trange
from tqdm.utils import _unicode, disp_len
import numpy as np
import h5py
from jax.tree_util import tree_map
from torch.utils.data import Dataset, DataLoader, default_collate
import jax
import jax.numpy as jnp
import equinox as eqx
import torch
import random
import optax


# For reproducibility purpose
def seed_everything(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def tqdm_slurm(_TQDM_STATUS_EVERY_N = 5):
    if "SLURM_JOB_ID" in os.environ:
        def status_printer(self, file):
            """
            Manage the printing and in-place updating of a line of characters.
            Note that if the string is longer than a line, then in-place
            updating may not work (it will print a new line at each refresh).
            """
            self._status_printer_counter = 0
            fp = file
            fp_flush = getattr(fp, 'flush', lambda: None)  # pragma: no cover
            if fp in (sys.stderr, sys.stdout):
                getattr(sys.stderr, 'flush', lambda: None)()
                getattr(sys.stdout, 'flush', lambda: None)()

            def fp_write(s):
                fp.write(_unicode(s))
                fp_flush()

            last_len = [0]

            def print_status(s):
                self._status_printer_counter += 1
                if self._status_printer_counter % _TQDM_STATUS_EVERY_N == 0:
                    len_s = disp_len(s)
                    # This is where we've removed the \r for clearer output
                    fp_write(s + (' ' * max(last_len[0] - len_s, 0)) + '\n')
                    last_len[0] = len_s

            return print_status
        tqdm.status_printer = status_printer

class H5Dataset(Dataset):
   def __init__(self, path, name):
      self.file_path = path
      self.dataset = None
      self.name = name
      with h5py.File(self.file_path, 'r') as file:
         self.dataset_len = len(file[name])
   def __getitem__(self, index):
      if self.dataset is None:
          self.dataset = h5py.File(self.file_path, 'r')[self.name]
      return self.dataset[index]
   def __len__(self):
      return self.dataset_len

class MultiH5Dataset(Dataset):
    def __init__(self, h5_paths, dataset_name='data'):
        self.datasets = []
        self.indices = []
        self.name = dataset_name

        # Open all files and map indices
        for path in h5_paths:
            h5f = h5py.File(path, 'r')
            data = h5f[dataset_name]
            self.datasets.append((h5f, data))
            self.indices.extend([(len(self.datasets) - 1, i) for i in range(data.shape[0])])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        dataset_idx, local_idx = self.indices[idx]
        h5f, data = self.datasets[dataset_idx]
        sample = data[local_idx]
        return sample

    def __del__(self):
        for h5f, _ in self.datasets:
            h5f.close()

def numpy_collate(batch):
    return tree_map(np.asarray, default_collate(batch))

def jax_numpy_collate(batch):
    return tree_map(jnp.asarray, default_collate(batch))


class NumpyLoader(DataLoader):
  def __init__(self, dataset, batch_size=32,
                shuffle=True, sampler=None,
                batch_sampler=None, num_workers=0,
                pin_memory=False, drop_last=False,
                collate_fn=jax_numpy_collate, 
                timeout=0, worker_init_fn=None):
    #super(self.__class__, self).__init__(dataset,
    super().__init__(dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        collate_fn=collate_fn, 
        pin_memory=pin_memory,
        drop_last=drop_last,
        timeout=timeout,
        worker_init_fn=worker_init_fn)


def count_parameters(model: eqx.Module):
    return sum(p.size for p in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array)))

## Ne marche pas
# from torch.utils.data import Sampler
# import torch
# 
# class SeededRandomSampler(Sampler):
#     def __init__(self, data_source, seed):
#         self.data_source = data_source
#         self.seed = seed
#         self.epoch = 0
# 
#     def __iter__(self):
#         g = torch.Generator()
#         g.manual_seed(self.seed + self.epoch)
#         indices = torch.randperm(len(self.data_source), generator=g).tolist()
#         return iter(indices)
# 
#     def __len__(self):
#         return len(self.data_source)
# 
#     def set_epoch(self, epoch):
#         self.epoch = epoch
# 
# sampler = SeededRandomSampler(data_source=h5dl, seed=42)
# loader = NumpyLoader(dataset=h5dl , shuffle=False, batch_size=n_phys, pin_memory=True, drop_last=True, num_workers = 0, sampler=sampler)


# num_workder = 1 + collate_fn = numpy_collate not faster
def train_phys(
        model, 
        dataset, 
        step,
        opt_state,
        n_phys = 32, n_epoch = 100,
        check = 10,
        n_log = 10,
        n_save = 20,
        num_workers = 0,
        collate_fn=jax_numpy_collate, 
        lr = 1e-3,
        model_path = None, 
        *,
        key
    ):

    n_sample = len(dataset)
    pbar = trange(n_epoch, dynamic_ncols=True, file=sys.stdout, leave=True, position=0)
    best = float('inf')
    loss_history = []
    early_stopper = 0
    n_batch = n_sample // n_phys

    sampler = torch.utils.data.RandomSampler(dataset, replacement=False)
    loader = NumpyLoader(dataset=dataset , shuffle=False, batch_size=n_phys, pin_memory=True, 
                         drop_last=True, num_workers = num_workers, 
                         sampler=sampler, collate_fn=collate_fn)

    for epoch in pbar:
        #sampler.set_epoch(epoch)
        batch_loss = 0
        #key, train_key = jr.split(key)

        for batch, phi_u_inits in enumerate(loader):

            if (batch != (n_batch -1)) or (epoch <=check):
                model, opt_state, weighted_loss = step(model, opt_state, phi_u_inits)
            else: 
                # Save best performed hyper-parameters
                model_new, opt_state, weighted_loss = step(model, opt_state, phi_u_inits)

            batch_loss += weighted_loss

        batch_loss /= n_batch
        loss_history.append(batch_loss.item())

        if (epoch > check):
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