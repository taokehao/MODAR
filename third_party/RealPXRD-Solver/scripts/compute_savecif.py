import torch
import numpy as np
import argparse
from pymatgen.core.structure import Structure
from pymatgen.core.composition import Composition
from pymatgen.core.lattice import Lattice
from pymatgen.analysis.structure_matcher import StructureMatcher
from collections import Counter
from eval_utils import smact_validity, structure_validity, get_crystals_list_w_ids, get_crystals_true_list_w_id
from p_tqdm import p_map
import warnings
import os
warnings.filterwarnings('ignore')


num_cpus = 16

def get_crystal_array_list(file_path, batch_idx=0):
    data = torch.load(file_path, map_location='cpu')
    if batch_idx == -1:
        batch_size = data['frac_coords'].shape[0]
        crys_array_list = []
        for i in range(batch_size):
            tmp_crys_array_list = get_crystals_list_w_ids(
                data['frac_coords'][i],
                data['atom_types'][i],
                data['lengths'][i],
                data['angles'][i],
                data['num_atoms'][i],
                data['input_data_batch'].id)
            crys_array_list.append(tmp_crys_array_list)     
    else:
        crys_array_list = get_crystals_list_w_ids(
            data['frac_coords'][batch_idx],
            data['atom_types'][batch_idx],
            data['lengths'][batch_idx],
            data['angles'][batch_idx],
            data['num_atoms'][batch_idx],
            data['input_data_batch'].id)

    if 'input_data_batch' in data:
        batch = data['input_data_batch']
        if isinstance(batch, dict):
            true_crystal_array_list = get_crystals_true_list_w_id(
                batch['frac_coords'], batch['atom_types'], batch['lattice_matrix'], batch['num_atoms'], batch['id'])
        else:
            true_crystal_array_list = get_crystals_true_list_w_id(
                batch.frac_coords, batch.atom_types, batch.lattice_matrix, batch.num_atoms, batch.id)
    else:
        true_crystal_array_list = None

    return crys_array_list, true_crystal_array_list



class Crystal_truth(object):

    def __init__(self, crys_array_dict, path):
        self.id = crys_array_dict['id']
        self.frac_coords = crys_array_dict['frac_coords']
        self.atom_types = crys_array_dict['atom_types']
        self.lengths = crys_array_dict['lengths']
        self.angles = crys_array_dict['angles']
        self.formula = crys_array_dict['atom_types']
        self.dict = crys_array_dict
        self.path = path
        
        self.get_structure()
        self.get_composition()
        self.get_validity()
        


    def get_structure(self):
        if min(self.lengths.tolist()) < 0:
            self.constructed = False
            self.invalid_reason = 'non_positive_lattice'
        if np.isnan(self.lengths).any() or np.isnan(self.angles).any() or  np.isnan(self.frac_coords).any():
            self.constructed = False
            self.invalid_reason = 'nan_value'            
        else:
            try:
                self.structure = Structure(
                    lattice=Lattice.from_parameters(
                        *(self.lengths.tolist() + self.angles.tolist())),
                    species=self.atom_types, coords=self.frac_coords, coords_are_cartesian=False)
                self.constructed = True
                self.structure.to(filename=f"{self.path}/{self.id}_truth.cif")
            except Exception:
                self.constructed = False
                self.invalid_reason = 'construction_raises_exception'
            if self.structure.volume < 0.1:
                self.constructed = False
                self.invalid_reason = 'unrealistically_small_lattice'

    def get_composition(self):
        elem_counter = Counter(self.atom_types)
        composition = [(elem, elem_counter[elem])
                       for elem in sorted(elem_counter.keys())]
        elems, counts = list(zip(*composition))
        counts = np.array(counts)
        counts = counts / np.gcd.reduce(counts)
        self.elems = elems
        self.comps = tuple(counts.astype('int').tolist())

    def get_validity(self):
        self.comp_valid = smact_validity(self.elems, self.comps)
        if self.constructed:
            self.struct_valid = structure_validity(self.structure)
        else:
            self.struct_valid = False
        self.valid = self.comp_valid and self.struct_valid

class Crystal_predict(object):

    def __init__(self, crys_array_dict, path):
        self.id = crys_array_dict['id']
        self.frac_coords = crys_array_dict['frac_coords']
        self.atom_types = crys_array_dict['atom_types']
        self.lengths = crys_array_dict['lengths']
        self.angles = crys_array_dict['angles']
        self.formula = crys_array_dict['atom_types']
        self.dict = crys_array_dict
        self.path = path
        
        self.get_structure()
        self.get_composition()
        self.get_validity()
        


    def get_structure(self):
        if min(self.lengths.tolist()) < 0:
            self.constructed = False
            self.invalid_reason = 'non_positive_lattice'
        if np.isnan(self.lengths).any() or np.isnan(self.angles).any() or  np.isnan(self.frac_coords).any():
            self.constructed = False
            self.invalid_reason = 'nan_value'            
        else:
            try:
                self.structure = Structure(
                    lattice=Lattice.from_parameters(
                        *(self.lengths.tolist() + self.angles.tolist())),
                    species=self.atom_types, coords=self.frac_coords, coords_are_cartesian=False)
                self.constructed = True
            except Exception:
                self.constructed = False
                self.invalid_reason = 'construction_raises_exception'
            if self.structure.volume < 0.1:
                self.constructed = False
                self.invalid_reason = 'unrealistically_small_lattice'

    def get_composition(self):
        elem_counter = Counter(self.atom_types)
        composition = [(elem, elem_counter[elem])
                       for elem in sorted(elem_counter.keys())]
        elems, counts = list(zip(*composition))
        counts = np.array(counts)
        counts = counts / np.gcd.reduce(counts)
        self.elems = elems
        self.comps = tuple(counts.astype('int').tolist())

    def get_validity(self):
        self.comp_valid = smact_validity(self.elems, self.comps)
        if self.constructed:
            self.struct_valid = structure_validity(self.structure)
        else:
            self.struct_valid = False
        self.valid = self.comp_valid and self.struct_valid

class RecEvalBatch(object):

    def __init__(self, pred_crys, gt_crys, path, stol=0.5, angle_tol=10, ltol=0.3):
        self.matcher = StructureMatcher(
            stol=stol, angle_tol=angle_tol, ltol=ltol)
        self.preds = pred_crys
        self.gts = gt_crys
        self.path = path
        self.batch_size = len(self.preds)

    def get_match_rate_and_rms(self):
        def process_one(pred, gt, is_valid):
            if not is_valid:
                return None
            try:
                rms_dist = self.matcher.get_rms_dist(
                    pred.structure, gt.structure)
                rms_dist = None if rms_dist is None else rms_dist[0]
                # if rms_dist is not None:
                #     print(gt.structure.formula)
                #     print(rms_dist)
                return rms_dist
            except Exception:
                return None

        rms_dists = []
        self.all_rms_dis = np.zeros((self.batch_size, len(self.gts)))
        for i in range(len(self.preds[0])):
            tmp_rms_dists = []
            tmp_struct = []
            for j in range(self.batch_size):
                rmsd = process_one(self.preds[j][i], self.gts[i], self.preds[j][i].valid and self.gts[i].valid)
                self.all_rms_dis[j][i] = rmsd
                if rmsd is not None:
                    tmp_rms_dists.append(rmsd)
                    tmp_struct.append(self.preds[j][i])
            if len(tmp_rms_dists) == 0:
                rms_dists.append(None)
                pred_struct = self.preds[0][i].structure    # the None structure
                id = self.preds[0][i].id
                pred_struct.to(filename=f"{self.path}/{id}_predict_None.cif")
            else:
                min_index = np.argmin(tmp_rms_dists)
                rms_dists.append(tmp_rms_dists[min_index])
                rms = round(tmp_rms_dists[min_index], 4)
                pred_struct = self.preds[min_index][i].structure
                id = tmp_struct[min_index].id
                pred_struct.to(filename=f"{self.path}/{id}_predict_{rms}.cif")
        rms_dists = np.array(rms_dists)
        match_rate = sum(rms_dists != None) / len(self.preds[0])
        mean_rms_dist = rms_dists[rms_dists != None].mean()
        return {'match_rate': match_rate,
                'rms_dist': mean_rms_dist}    

    def get_metrics(self):
        metrics = {}
        metrics.update(self.get_match_rate_and_rms())
        return metrics


class RecEval(object):

    def __init__(self, pred_crys, gt_crys, path, stol=0.5, angle_tol=10, ltol=0.3):
        assert len(pred_crys) == len(gt_crys)
        self.matcher = StructureMatcher(stol=stol, angle_tol=angle_tol, ltol=ltol)
        self.preds = pred_crys
        self.path = path
        self.gts = gt_crys

    def get_match_rate_and_rms(self):
        def process_one(pred, gt, is_valid):
            if not is_valid:
                pred_struct = pred.structure    # the None structure
                id = pred.id
                pred_struct.to(filename=f"{self.path}/{id}_predict_None.cif")
                return None
            try:
                rms_dist = self.matcher.get_rms_dist(pred.structure, gt.structure)
                rms_dist = None if rms_dist is None else rms_dist[0]
                if rms_dist is not None:
                    rms = round(rms_dist, 4)
                    pred_struct = pred.structure
                    id = pred.id
                    pred_struct.to(filename=f"{self.path}/{id}_predict_{rms}.cif")
                else:
                    pred_struct = pred.structure    # the None structure
                    id = pred.id
                    pred_struct.to(filename=f"{self.path}/{id}_predict_None.cif")
                return rms_dist
            except Exception:
                pred_struct = pred.structure    # the None structure
                id = pred.id
                pred_struct.to(filename=f"{self.path}/{id}_predict_None.cif")
                return None
        validity = [c1.valid and c2.valid for c1,c2 in zip(self.preds, self.gts)]

        rms_dists = []
        for i in range(len(self.preds)):
            rms_dists.append(process_one(self.preds[i], self.gts[i], validity[i]))
        rms_dists = np.array(rms_dists)
        match_rate = sum(rms_dists != None) / len(self.preds)
        mean_rms_dist = rms_dists[rms_dists != None].mean()
        return {'match_rate': match_rate, 'rms_dist': mean_rms_dist}     

    def get_metrics(self):
        metrics = {}
        metrics.update(self.get_match_rate_and_rms())
        return metrics
    

def get_metrics(recon_file_path, save_path, multi_eval):
    all_metrics = {}
    batch_idx = -1 if multi_eval else 0
    crys_array_list, true_crystal_array_list = get_crystal_array_list(recon_file_path, batch_idx = batch_idx)

    gt_crys = p_map(lambda x, y: Crystal_truth(x, y), true_crystal_array_list, [save_path]*len(true_crystal_array_list), num_cpus=num_cpus)
    if multi_eval:
        pred_crys = []
        for i in range(len(crys_array_list)):
            # print(f"Processing batch {i}")
            pred_crys.append(p_map(lambda x, y: Crystal_predict(x, y), crys_array_list[i], [save_path]*len(true_crystal_array_list), num_cpus=num_cpus))
    else:
        pred_crys = p_map(lambda x, y: Crystal_predict(x, y), crys_array_list, [save_path]*len(true_crystal_array_list), num_cpus=num_cpus)
        

    if multi_eval:
        rec_evaluator = RecEvalBatch(pred_crys, gt_crys, save_path)
    else:
        rec_evaluator = RecEval(pred_crys, gt_crys, save_path)

    recon_metrics = rec_evaluator.get_metrics()

    all_metrics.update(recon_metrics)

    if multi_eval:
        print('sample %s: ' %len(crys_array_list))
    else:
        print('sample 1: ')
    print(all_metrics)



#### main

def main(args):
    
    print(args.pt_path)
    save_path = args.pt_path.rsplit('/', 1)[0] + '/' + args.save_path
    os.makedirs(save_path, exist_ok=True)
    get_metrics(args.pt_path, save_path, args.multi_eval)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pt_path', required=True)
    parser.add_argument('--save_path', required=True)
    parser.add_argument('--multi_eval', action='store_true')
    args = parser.parse_args()
    main(args)