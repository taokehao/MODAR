import wandb
import os
import lmdb
import gzip
import json
import pickle
import numpy as np
from functools import lru_cache
from typing import List
from omegaconf import DictConfig
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import Callback, EarlyStopping, LearningRateMonitor, ModelCheckpoint


def find_lmdb_files(folder_path):
    lmdb_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".lmdb"):
                lmdb_files.append(os.path.join(root, file))
    return lmdb_files

def check_path_type(path):
    if os.path.isfile(path):
        return [path]
    elif os.path.isdir(path):
        return find_lmdb_files(path)
    else:
        raise"Does not exist"

class MultiLMDBDataset:
    def __init__(self, db_path_folder, key_to_id=True, gzip=True):
        self.db_paths = check_path_type(db_path_folder)
        self.key_to_id = key_to_id
        self.gzip = gzip
        self.envs = []
        self.keys = []
        self._load_databases()

    def _load_databases(self):
        for i, db_path in enumerate(self.db_paths):
            assert os.path.isfile(db_path), f"{db_path} not found"
            env = self.connect_db(db_path)
            with env.begin() as txn:
                keys = list(txn.cursor().iternext(values=False))
            self.envs.append(env)
            self.keys.extend([(i, key) for key in keys])
    
    def connect_db(self, lmdb_path):
        return lmdb.open(
            lmdb_path,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=256,
        )
    def __len__(self):
        return len(self.keys)
    
    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        env_idx, key = self.keys[idx]
        datapoint_pickled = self.envs[env_idx].begin().get(key)
        if self.gzip:
            datapoint_pickled = gzip.decompress(datapoint_pickled)
        data = pickle.loads(datapoint_pickled)
        if self.key_to_id:
            data["id"] = int.from_bytes(key, "big")
        return data, key
    

def build_callbacks(cfg: DictConfig) -> List[Callback]:
    callbacks: List[Callback] = []

    if "model_checkpoints" in cfg.logging:
        callbacks.append(
            ModelCheckpoint(**cfg.logging.model_checkpoints)
        )

        callbacks.append(
            ModelCheckpoint(
                dirpath=cfg.logging.model_checkpoints.dirpath,
                filename='last_one'
            )
        )

        
    if "lr_monitor" in cfg.logging:
        callbacks.append(
            LearningRateMonitor(
                logging_interval=cfg.logging.lr_monitor.logging_interval,
                log_momentum=cfg.logging.lr_monitor.log_momentum,
            )
        )

    # if "early_stopping" in cfg.logging:
    #     callbacks.append(
    #         EarlyStopping(
    #             monitor=cfg.logging.early_stopping.monitor,
    #             mode=cfg.logging.early_stopping.mode,
    #             patience=cfg.logging.early_stopping.patience,
    #             verbose=cfg.logging.early_stopping.verbose,
    #         )
    #     )


    return callbacks

def Wandb_logger(cfg: DictConfig):
    wandb_logger = None
    if "wandb" in cfg.logging:
        wandb_config = cfg.logging.wandb
        wandb_logger = WandbLogger(
            **wandb_config,
            settings=wandb.Settings(start_method="fork"),
            tags=cfg.core.tags,
        )
    
    return wandb_logger





atomic_masses = {
    "X": 0,
    "H": 1.008,
    "He": 4.0026,
    "Li": 6.94,
    "Be": 9.0122,
    "B": 10.81,
    "C": 12.011,
    "N": 14.007,
    "O": 15.999,
    "F": 18.998,
    "Ne": 20.180,
    "Na": 22.990,
    "Mg": 24.305,
    "Al": 26.982,
    "Si": 28.085,
    "P": 30.974,
    "S": 32.06,
    "Cl": 35.45,
    "Ar": 39.95,
    "K": 39.098,
    "Ca": 40.078,
    "Sc": 44.956,
    "Ti": 47.867,
    "V": 50.942,
    "Cr": 51.996,
    "Mn": 54.938,
    "Fe": 55.845,
    "Co": 58.933,
    "Ni": 58.693,
    "Cu": 63.546,
    "Zn": 65.38,
    "Ga": 69.723,
    "Ge": 72.630,
    "As": 74.922,
    "Se": 78.971,
    "Br": 79.904,
    "Kr": 83.798,
    "Rb": 85.468,
    "Sr": 87.62,
    "Y": 88.906,
    "Zr": 91.224,
    "Nb": 92.906,
    "Mo": 95.95,
    "Tc": 98,  # Technetium is synthetic and has no stable isotopes
    "Ru": 101.07,
    "Rh": 102.91,
    "Pd": 106.42,
    "Ag": 107.87,
    "Cd": 112.41,
    "In": 114.82,
    "Sn": 118.71,
    "Sb": 121.76,
    "Te": 127.60,
    "I": 126.90,
    "Xe": 131.29,
    "Cs": 132.91,
    "Ba": 137.33,
    "La": 138.91,
    "Ce": 140.12,
    "Pr": 140.91,
    "Nd": 144.24,
    "Pm": 145,  # Promethium is synthetic and has no stable isotopes
    "Sm": 150.36,
    "Eu": 151.96,
    "Gd": 157.25,
    "Tb": 158.93,
    "Dy": 162.50,
    "Ho": 164.93,
    "Er": 167.26,
    "Tm": 168.93,
    "Yb": 173.05,
    "Lu": 174.97,
    "Hf": 178.49,
    "Ta": 180.95,
    "W": 183.84,
    "Re": 186.21,
    "Os": 190.23,
    "Ir": 192.22,
    "Pt": 195.08,
    "Au": 196.97,
    "Hg": 200.59,
    "Tl": 204.38,
    "Pb": 207.2,
    "Bi": 208.98,
    "Po": 209,  # Polonium is synthetic and has no stable isotopes
    "At": 210,  # Astatine is synthetic and has no stable isotopes
    "Rn": 222,  # Radon is synthetic and has no stable isotopes
    "Fr": 223,  # Francium is synthetic and has no stable isotopes
    "Ra": 226,  # Radium is synthetic and has no stable isotopes
    "Ac": 227,  # Actinium is synthetic and has no stable isotopes
    "Th": 232.04,
    "Pa": 231.04,
    "U": 238.03,
    "Np": 237,  # Neptunium is synthetic and has no stable isotopes
    "Pu": 244,  # Plutonium is synthetic and has no stable isotopes
    "Am": 243,  # Americium is synthetic and has no stable isotopes
    "Cm": 247,  # Curium is synthetic and has no stable isotopes
    "Bk": 247,  # Berkelium is synthetic and has no stable isotopes
    "Cf": 251,  # Californium is synthetic and has no stable isotopes
    "Es": 252,  # Einsteinium is synthetic and has no stable isotopes
    "Fm": 257,  # Fermium is synthetic and has no stable isotopes
    "Md": 258,  # Mendelevium is synthetic and has no stable isotopes
    "No": 259,  # Nobelium is synthetic and has no stable isotopes
    "Lr": 266,  # Lawrencium is synthetic and has no stable isotopes
    "Rf": 267,  # Rutherfordium is synthetic and has no stable isotopes
    "Db": 270,  # Dubnium is synthetic and has no stable isotopes
    "Sg": 271,  # Seaborgium is synthetic and has no stable isotopes
    "Bh": 270,  # Bohrium is synthetic and has no stable isotopes
    "Hs": 277,  # Hassium is synthetic and has no stable isotopes
    "Mt": 276,  # Meitnerium is synthetic and has no stable isotopes
    "Ds": 281,  # Darmstadtium is synthetic and has no stable isotopes
    "Rg": 282,  # Roentgenium is synthetic and has no stable isotopes
    "Cn": 285,  # Copernicium is synthetic and has no stable isotopes
    "Nh": 286,  # Nihonium is synthetic and has no stable isotopes
    "Fl": 289,  # Flerovium is synthetic and has no stable isotopes
    "Mc": 290,  # Moscovium is synthetic and has no stable isotopes
    "Lv": 293,  # Livermorium is synthetic and has no stable isotopes
    "Ts": 294,  # Tennessine is synthetic and has no stable isotopes
    "Og": 294,  # Oganesson is synthetic and has no stable isotopes
}

idx_atom = {k: idx for idx, (k,v) in enumerate(atomic_masses.items())}




class AtomInitializer(object):
    """
    Base class for intializing the vector representation for atoms.

    !!! Use one AtomInitializer per dataset !!!
    """
    def __init__(self, atom_types):
        self.atom_types = set(atom_types)
        self._embedding = {}

    def get_atom_fea(self, atom_type):
        assert atom_type in self.atom_types
        return self._embedding[atom_type]

    def load_state_dict(self, state_dict):
        self._embedding = state_dict
        self.atom_types = set(self._embedding.keys())
        self._decodedict = {idx: atom_type for atom_type, idx in
                            self._embedding.items()}

    def state_dict(self):
        return self._embedding

    def decode(self, idx):
        if not hasattr(self, '_decodedict'):
            self._decodedict = {idx: atom_type for atom_type, idx in
                                self._embedding.items()}
        return self._decodedict[idx]

class AtomCustomJSONInitializer(AtomInitializer):
    """
    Initialize atom feature vectors using a JSON file, which is a python
    dictionary mapping from element number to a list representing the
    feature vector of the element.

    Parameters
    ----------

    elem_embedding_file: str
        The path to the .json file
    """
    def __init__(self, elem_embedding_file):
        with open(elem_embedding_file) as f:
            elem_embedding = json.load(f)
        elem_embedding = {int(key): value for key, value
                          in elem_embedding.items()}
        atom_types = set(elem_embedding.keys())
        super(AtomCustomJSONInitializer, self).__init__(atom_types)
        for key, value in elem_embedding.items():
            self._embedding[key] = np.array(value, dtype=float)

