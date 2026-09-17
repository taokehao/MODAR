import os
import sys
import time
import torch
import hydra
import argparse
import warnings
import chemparse
import numpy as np
from pathlib import Path
from p_tqdm import p_map
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch
from hydra import initialize_config_dir
from torch_geometric.data import DataLoader
from scipy.signal import find_peaks
from numpy.polynomial import Polynomial
from pymatgen.io.cif import CifWriter
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure

sys.path.append('.')
from app.model.flow import CSPFlow
from scripts.eval_utils import chemical_symbols, lattices_to_params_shape, get_crystals_list, lattice_params_to_matrix_torch
warnings.filterwarnings('ignore')


def subtract_polynomial_background(continue_x, continue_y, poly_order, smoothing_window = 3):
    
    smoothed_data = np.convolve(continue_y, np.ones(smoothing_window) / smoothing_window, mode='same')
    p = Polynomial.fit(continue_x, smoothed_data, poly_order)
    background = p(continue_x)
    intensity_corrected = smoothed_data - background
    intensity_corrected = np.maximum(0, intensity_corrected)
    
    return intensity_corrected/np.max(intensity_corrected)*100


def search_peaks(continue_x, continue_y, intensity_threshold = 3, min_peak_width = 3, prominence=0.05):
    prominence = (np.max(continue_y) - np.min(continue_y)) * prominence
    peaks_indices, _ = find_peaks(continue_y, height=intensity_threshold, width=min_peak_width, prominence=prominence)
    pxrd_x = continue_x[peaks_indices]
    pxrd_y = continue_y[peaks_indices]
    pxrd_y = pxrd_y/np.max(pxrd_y)*100
    return pxrd_x, pxrd_y


class SampleDataset(Dataset):

    def __init__(self, formula, xrd_path, cell, num_evals):
        super().__init__()
        self.formula = formula
        cell = torch.Tensor(np.array(cell))
        self.lattice_matrix = lattice_params_to_matrix_torch(cell)
        xrd = np.loadtxt(xrd_path)
        continue_x = xrd[:, 0]
        continue_y = xrd[:, 1]
        continue_y_delete_bg = subtract_polynomial_background(continue_x, continue_y, poly_order=6)
        self.pxrd_x, self.pxrd_y = search_peaks(continue_x, continue_y_delete_bg)
        self.num_evals = num_evals
        self.get_structure()

    def get_structure(self):
        self.composition = chemparse.parse_formula(self.formula)
        chem_list = []
        for elem in self.composition:
            num_int = int(self.composition[elem])
            chem_list.extend([chemical_symbols.index(elem)] * num_int)
        self.chem_list = chem_list

    def __len__(self) -> int:
        return self.num_evals

    def __getitem__(self, index):
        return Data(
            atom_types=torch.LongTensor(self.chem_list),
            lattice_matrix=torch.Tensor(self.lattice_matrix).view(1,3,3),
            pxrd_x = torch.Tensor(self.pxrd_x).view(-1, 1),
            pxrd_y = torch.Tensor(self.pxrd_y).view(-1, 1),
            peak_num = len(self.pxrd_x),
            num_atoms=len(self.chem_list),
            num_nodes=len(self.chem_list),
        )

def load_model(model_path, label=-1):
    with initialize_config_dir(str(model_path/'.hydra')):
        
        cfg = hydra.compose(config_name='configL')
        ckpt = list(sorted(model_path.glob('*.ckpt')))
        print(str(ckpt[label]))
        model = CSPFlow.load_from_checkpoint(str(ckpt[label]), **cfg.model)
        model.eval()

        return model
        


def flow(loader, model, step_lr = 5, infer_timesteps=200):

    frac_coords = []
    num_atoms = []
    atom_types = []
    lattices = []
    for idx, batch in enumerate(loader):

        batch.to(model.device)
        outputs = model.sample(batch, step_lr = step_lr, infer_timesteps=infer_timesteps)
        frac_coords.append(outputs['frac_coords'].detach().cpu())
        num_atoms.append(outputs['num_atoms'].detach().cpu())
        atom_types.append(outputs['atom_types'].detach().cpu())
        lattices.append(outputs['lattices'].detach().cpu())

    frac_coords = torch.cat(frac_coords, dim=0)
    num_atoms = torch.cat(num_atoms, dim=0)
    atom_types = torch.cat(atom_types, dim=0)
    lattices = torch.cat(lattices, dim=0)
    lengths, angles = lattices_to_params_shape(lattices)

    return (
        frac_coords, atom_types, lattices, lengths, angles, num_atoms
    )


def get_pymatgen(crystal_array):
    frac_coords = crystal_array['frac_coords']
    atom_types = crystal_array['atom_types']
    lengths = crystal_array['lengths']
    angles = crystal_array['angles']
    try:
        structure = Structure(
            lattice=Lattice.from_parameters(
                *(lengths.tolist() + angles.tolist())),
            species=atom_types, coords=frac_coords, coords_are_cartesian=False)
        return structure
    except:
        return None

def main(args):
    startime = time.time()
    model_path = Path(args.model_path)
    
    diff_model = load_model(model_path, label=args.label)
    tar_dir = os.path.join(args.save_path, args.formula)

    os.makedirs(tar_dir, exist_ok=True)
    test_set = SampleDataset(args.formula, args.xrd_path, args.cell, args.num_evals)
    test_loader = DataLoader(test_set, batch_size = min(256, args.num_evals))
    
    (frac_coords, atom_types, lattices, lengths, angles, num_atoms) = flow(test_loader, diff_model)

    crystal_list = get_crystals_list(frac_coords, atom_types, lengths, angles, num_atoms)
    strcuture_list = p_map(get_pymatgen, crystal_list)

    for i,structure in enumerate(strcuture_list):
        tar_file = os.path.join(tar_dir, f"{args.formula}_{i+1}.cif")
        if structure is not None:
            writer = CifWriter(structure)
            writer.write_file(tar_file)
        else:
            print(f"{i+1} Error Structure.")
    endtime = time.time()
    print('use time %s s.' %(endtime-startime))
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--xrd_path', required=True)
    parser.add_argument('--formula', required=True)
    parser.add_argument('--cell', required=True, nargs=6, type=float)
    parser.add_argument('--save_path', required=True)
    parser.add_argument('--num_evals', default=1, type=int)
    parser.add_argument('--label', default=-1, type=int)
    args = parser.parse_args()
    main(args)