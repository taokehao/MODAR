import torch
import numpy as np
from omegaconf import ValueNode
from torch.utils.data import Dataset
from torch_geometric.data import Data
from app.data.utils import MultiLMDBDataset, idx_atom

def augment_spectrum(two_theta, intensity, noise_level=0.05, shift_range=0.1, scale_range=(0.8, 1.2)):
    # add Gaussian noise
    noise = np.random.normal(0, noise_level * np.max(intensity), size=intensity.shape)
    intensity_aug = intensity + noise

    # peak shift
    shift = np.random.uniform(-shift_range, shift_range, size=two_theta.shape)
    two_theta_aug = two_theta + shift

    # scale
    scale_factor = np.random.uniform(*scale_range)
    intensity_aug *= scale_factor
    intensity_aug = np.where(intensity_aug < 0, intensity, intensity_aug)

    # norm
    max_intensity = np.max(intensity_aug)
    intensity_aug = intensity_aug / max_intensity * 100
    
    return two_theta_aug, intensity_aug


class CrystDataset_pxrd(Dataset):
    def __init__(self, path: ValueNode, gzip: ValueNode, conventional: ValueNode, xrd_augment: ValueNode):
        self.cached_data = MultiLMDBDataset(path, gzip=gzip, key_to_id=False)
        self.conventional = conventional
        self.xrd_augment = xrd_augment
        
    def __len__(self):
        return len(self.cached_data)

    def __getitem__(self, index):
        data_dict, _ = self.cached_data[index]
        if self.conventional:
            atom_types = data_dict['c_atom_type']
            atom_types = np.array([idx_atom.get(i, 0) for i in atom_types])
            atom_pos = data_dict['c_atom_pos']
            lattice_matrix = data_dict['c_lattice_matrix']
        else:
            atom_types = data_dict['p_atom_type']
            atom_types = np.array([idx_atom.get(i, 0) for i in atom_types])
            atom_pos = data_dict['p_atom_pos']
            lattice_matrix = data_dict['p_lattice_matrix']

        if self.xrd_augment:
            pxrd_x = np.array(data_dict['pxrd_x'])
            pxrd_y = np.array(data_dict['pxrd_y'])
            mask = pxrd_y > 5
            pxrd_x, pxrd_y = pxrd_x[mask], pxrd_y[mask]
            pxrd_x_aug, pxrd_y_aug = augment_spectrum(pxrd_x, pxrd_y)
            mask_aug = pxrd_y > 5
            pxrd_x_final, pxrd_y_final = pxrd_x_aug[mask_aug], pxrd_y_aug[mask_aug]
        else:
            pxrd_x = np.array(data_dict['pxrd_x'])
            pxrd_y = np.array(data_dict['pxrd_y'])
            mask = pxrd_y > 5
            pxrd_x_final = pxrd_x[mask]
            pxrd_y_final = pxrd_y[mask]

        data = Data(
            frac_coords=torch.Tensor(atom_pos),
            atom_types=torch.LongTensor(atom_types),
            lattice_matrix=torch.Tensor(lattice_matrix).view(1,3,3),
            num_atoms=len(atom_types),
            num_nodes=len(atom_types),
            pxrd_x = torch.Tensor(pxrd_x_final).view(-1, 1),
            pxrd_y = torch.Tensor(pxrd_y_final).view(-1, 1),
            peak_num = len(pxrd_x_final),
        )

        return data

