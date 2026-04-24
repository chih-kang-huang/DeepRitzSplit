import os, sys
from tqdm import tqdm
from tqdm.utils import _unicode, disp_len
import numpy as np
import h5py
import equinox as eqx 
import jax
from jax.tree_util import tree_map
from torch.utils.data import Dataset, DataLoader, default_collate


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

def numpy_collate(batch):
    return tree_map(np.asarray, default_collate(batch))

#class H5Dataset(data.Dataset):
#    def __init__(self, h5_path):
#        self.h5_file = h5py.File(h5_path, "r")
#
#    def __getitem__(self, index):
#        return (
#            self.h5_file["u_sols"][index]
#        )
#
#    def __len__(self):
#        return self.h5_file["u_sols"].size

class NumpyLoader(DataLoader):
  def __init__(self, dataset, batch_size=1,
                shuffle=False, sampler=None,
                batch_sampler=None, num_workers=0,
                pin_memory=False, drop_last=False,
                collate_fn=None, 
                timeout=0, worker_init_fn=None):
    super(self.__class__, self).__init__(dataset,
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
