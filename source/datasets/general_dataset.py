import binascii
import glob
import os
import pickle
from collections import defaultdict
from multiprocessing import Pool
import random
import copy
import torch.nn.functional as F
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import MolFromSmiles, AddHs
from torch_geometric.data import Dataset, HeteroData
from torch_geometric.transforms import BaseTransform
from tqdm import tqdm
from rdkit.Chem import RemoveAllHs


from source.datasets.process_mols import read_molecule, get_lig_graph_with_matching, generate_conformer, moad_extract_receptor_structure
from source.utils.diffusion_utils import modify_conformer, set_time
from source.utils import so3, torus


class NoiseTransform(BaseTransform):
    def __init__(self, t_to_sigma, no_torsion, all_atom, alpha=1, beta=1,
                 include_miscellaneous_atoms=False, time_independent=False, rmsd_cutoff=0,
                 minimum_t=0, sampling_mixing_coeff=0):
        self.t_to_sigma = t_to_sigma
        self.no_torsion = no_torsion
        self.all_atom = all_atom
        self.include_miscellaneous_atoms = include_miscellaneous_atoms
        self.minimum_t = minimum_t
        self.mixing_coeff = sampling_mixing_coeff
        self.alpha = alpha
        self.beta = beta
        self.rmsd_cutoff = rmsd_cutoff
        self.time_independent = time_independent

    def __call__(self, data):
        t_tr, t_rot, t_tor, t = self.get_time()
        return self.apply_noise(data, t_tr, t_rot, t_tor, t)

    def get_time(self):
        if self.time_independent:
            t = np.random.beta(self.alpha, self.beta)
            t_tr, t_rot, t_tor = t,t,t
        else:
            t = None
            if self.mixing_coeff == 0:
                t = np.random.beta(self.alpha, self.beta)
                t = self.minimum_t + t * (1 - self.minimum_t)
            else:
                choice = np.random.binomial(1, self.mixing_coeff)
                t1 = np.random.beta(self.alpha, self.beta)
                t1 = t1 * self.minimum_t
                t2 = np.random.beta(self.alpha, self.beta)
                t2 = self.minimum_t + t2 * (1 - self.minimum_t)
                t = choice * t1 + (1 - choice) * t2

            t_tr, t_rot, t_tor = t,t,t
        return t_tr, t_rot, t_tor, t

    def apply_noise(self, data, t_tr, t_rot, t_tor, t, tr_update=None, rot_update=None, torsion_updates=None):
        if not torch.is_tensor(data['ligand'].pos):
            data['ligand'].pos = random.choice(data['ligand'].pos)

        if self.time_independent:
            orig_complex_graph = copy.deepcopy(data)

        tr_sigma, rot_sigma, tor_sigma = self.t_to_sigma(t_tr, t_rot, t_tor)

        if self.time_independent:
            set_time(data, 0, 0, 0, 0, 1, self.all_atom, device=None,
                     include_miscellaneous_atoms=self.include_miscellaneous_atoms)
        else:
            set_time(data, t, t_tr, t_rot, t_tor, 1, self.all_atom, device=None,
                     include_miscellaneous_atoms=self.include_miscellaneous_atoms)

        tr_update = torch.normal(mean=0, std=tr_sigma, size=(1, 3)) if tr_update is None else tr_update
        rot_update = so3.sample_vec(eps=rot_sigma) if rot_update is None else rot_update
        torsion_updates = np.random.normal(loc=0.0, scale=tor_sigma, size=data['ligand'].edge_mask.sum()) \
            if torsion_updates is None else torsion_updates
        torsion_updates = None if self.no_torsion else torsion_updates
        try:
            modify_conformer(data, tr_update, torch.from_numpy(rot_update).float(), torsion_updates)
        except Exception as e:
            print("failed modify conformer")
            print(e)

        if self.time_independent:
            if self.no_torsion:
                orig_complex_graph['ligand'].orig_pos = (orig_complex_graph['ligand'].pos.cpu().numpy() +
                                                         orig_complex_graph.original_center.cpu().numpy())

            filterHs = torch.not_equal(data['ligand'].x[:, 0], 0).cpu().numpy()
            if isinstance(orig_complex_graph['ligand'].orig_pos, list):
                orig_complex_graph['ligand'].orig_pos = orig_complex_graph['ligand'].orig_pos[0]
            ligand_pos = data['ligand'].pos.cpu().numpy()[filterHs]
            orig_ligand_pos = orig_complex_graph['ligand'].orig_pos[filterHs] - orig_complex_graph.original_center.cpu().numpy()
            rmsd = np.sqrt(((ligand_pos - orig_ligand_pos) ** 2).sum(axis=1).mean(axis=0))
            data.y = torch.tensor(rmsd < self.rmsd_cutoff).float().unsqueeze(0)
            data.atom_y = data.y
            return data

        data.tr_score = -tr_update / tr_sigma ** 2
        data.rot_score = torch.from_numpy(so3.score_vec(vec=rot_update, eps=rot_sigma)).float().unsqueeze(0)
        data.tor_score = None if self.no_torsion else torch.from_numpy(torus.score(torsion_updates, tor_sigma)).float()
        data.tor_sigma_edge = None if self.no_torsion else np.ones(data['ligand'].edge_mask.sum()) * tor_sigma

        if data['ligand'].pos.shape[0] == 1:
            # if the ligand is a single atom, the rotational score is always 0
            data.rot_score = data.rot_score * 0

        set_time(data, t, t_tr, t_rot, t_tor, 1, self.all_atom, device=None,
                 include_miscellaneous_atoms=self.include_miscellaneous_atoms)
        return data


class GeneralDataset(Dataset):
    def __init__(self, root, transform=None, cache_path='data/cache', split_path='data/', limit_complexes=0,
                 receptor_radius=30, num_workers=1, c_alpha_max_neighbors=None, popsize=15, maxiter=15,
                 matching=True, keep_original=False,  remove_hs=False, num_conformers=1, all_atoms=False,
                 atom_radius=5, atom_max_neighbors=None, require_ligand=False,
                 include_miscellaneous_atoms=False,
                 protein_path_list=None, ligand_descriptions=None, keep_local_structures=False,
                 protein_file="protein_processed", ligand_file="ligand",
                 knn_only_graph=False, matching_tries=1, dataset='PDBBind'):

        super(GeneralDataset, self).__init__(root, transform)
        self.pdbbind_dir = root
        self.include_miscellaneous_atoms = include_miscellaneous_atoms
        self.split_path = split_path
        self.limit_complexes = limit_complexes
        self.receptor_radius = receptor_radius
        self.num_workers = num_workers
        self.c_alpha_max_neighbors = c_alpha_max_neighbors
        self.remove_hs = remove_hs
        self.require_ligand = require_ligand
        self.protein_path_list = protein_path_list
        self.ligand_descriptions = ligand_descriptions
        self.keep_local_structures = keep_local_structures
        self.protein_file = protein_file
        self.fixed_knn_radius_graph = True
        self.knn_only_graph = knn_only_graph
        self.matching_tries = matching_tries
        self.ligand_file = ligand_file
        self.dataset = dataset
        assert knn_only_graph or (not all_atoms)
        self.all_atoms = all_atoms
        if matching or protein_path_list is not None and ligand_descriptions is not None:
            cache_path += '_torsion'

        self.full_cache_path = os.path.join(cache_path, f'{dataset}3_limit{self.limit_complexes}'
                                                        f'_INDEX{os.path.splitext(os.path.basename(self.split_path))[0]}'
                                                        f'_recRad{self.receptor_radius}_recMax{self.c_alpha_max_neighbors}'
                                            + (''if not matching or num_conformers == 1 else f'_confs{num_conformers}')
                                            + '_full'
                                            + ('' if not keep_local_structures else f'_keptLocalStruct')
                                            + ('' if protein_path_list is None or ligand_descriptions is None else str(binascii.crc32(''.join(ligand_descriptions + protein_path_list).encode())))
                                            + ('' if protein_file == "protein_processed" else '_' + protein_file)
                                            + ('' if not self.fixed_knn_radius_graph else (f'_fixedKNN' if not self.knn_only_graph else '_fixedKNNonly'))
                                            + ('' if self.matching_tries == 1 else f'_tries{matching_tries}'))
        self.popsize, self.maxiter = popsize, maxiter
        self.matching, self.keep_original = matching, keep_original
        self.num_conformers = num_conformers

        self.atom_radius, self.atom_max_neighbors = atom_radius, atom_max_neighbors

        if not self.check_all_complexes():
            os.makedirs(self.full_cache_path, exist_ok=True)
            if protein_path_list is None or ligand_descriptions is None:
                self.preprocessing()
            else:
                self.inference_preprocessing()

        self.complex_graphs, self.rdkit_ligands = self.collect_all_complexes()
        print_statistics(self.complex_graphs)
        list_names = [complex['name'] for complex in self.complex_graphs]
        with open(os.path.join(self.full_cache_path, f'pdbbind_{os.path.splitext(os.path.basename(self.split_path))[0][:3]}_names.txt'), 'w') as f:
            f.write('\n'.join(list_names))

    def len(self):
        return len(self.complex_graphs)

    def get(self, idx):
        complex_graph = copy.deepcopy(self.complex_graphs[idx])
        if self.require_ligand:
            complex_graph.mol = RemoveAllHs(copy.deepcopy(self.rdkit_ligands[idx]))

        for a in ['random_coords', 'coords', 'seq', 'sequence', 'mask', 'rmsd_matching', 'cluster', 'orig_seq', 'to_keep', 'chain_ids']:
            if hasattr(complex_graph, a):
                delattr(complex_graph, a)
            if hasattr(complex_graph['receptor'], a):
                delattr(complex_graph['receptor'], a)

        return complex_graph

    def preprocessing(self):
        print(f'Processing data from [{self.split_path}] and saving it to [{self.full_cache_path}]')

        complex_names_all = glob.glob(self.split_path + "/*")
        complex_names_all.sort()

        if self.limit_complexes is not None and self.limit_complexes != 0:
            complex_names_all = complex_names_all[:self.limit_complexes]
        print(f'Loading {len(complex_names_all)} complexes.')

        # running preprocessing in parallel on multiple workers and saving the progress every processed complex
        list_indices = list(range(len(complex_names_all)))
        # random.shuffle(list_indices)
        for i in list_indices:
            if os.path.exists(os.path.join(self.full_cache_path, f"heterographs{i}.pkl")):
                continue
            patch_names = glob.glob(complex_names_all[i] + '/extracted_amino/*')
            complex_graphs, rdkit_ligands = [], []
            if self.num_workers > 1:
                p = Pool(self.num_workers, maxtasksperchild=1)
                p.__enter__()
            with tqdm(total=len(patch_names), desc=f'Loading patches from complex {i+1}/{len(complex_names_all)}') as pbar:
                map_fn = p.imap_unordered if self.num_workers > 1 else map
                for t in map_fn(self.get_patch, zip(patch_names, [None] * len(patch_names), [None] * len(patch_names))):
                    complex_graphs.extend(t[0])
                    rdkit_ligands.extend(t[1])
                    pbar.update()

            if self.num_workers > 1:
                p.__exit__(None, None, None)

            with open(os.path.join(self.full_cache_path, f"heterographs{i}.pkl"), 'wb') as f:
                pickle.dump((complex_graphs), f)
            with open(os.path.join(self.full_cache_path, f"rdkit_ligands{i}.pkl"), 'wb') as f:
                pickle.dump((rdkit_ligands), f)

    def inference_preprocessing(self):
        ligands_list = []
        print('Reading molecules and generating local structures with RDKit')
        for ligand_description in tqdm(self.ligand_descriptions):
            mol = read_molecule(ligand_description, remove_hs=False, sanitize=True)
            if not self.keep_local_structures:
                mol.RemoveAllConformers()
                mol = AddHs(mol)
                generate_conformer(mol)
            ligands_list.append(mol)

        print('Generating graphs for ligands and proteins')
        # running preprocessing in parallel on multiple workers and saving the progress every 1000 complexes
        list_indices = list(range(len(self.protein_path_list)//1000+1))
        random.shuffle(list_indices)
        for i in list_indices:
            if os.path.exists(os.path.join(self.full_cache_path, f"heterographs{i}.pkl")):
                continue
            protein_paths_chunk = self.protein_path_list[1000*i:1000*(i+1)]
            ligand_description_chunk = self.ligand_descriptions[1000*i:1000*(i+1)]
            ligands_chunk = ligands_list[1000 * i:1000 * (i + 1)]
            complex_graphs, rdkit_ligands = [], []
            if self.num_workers > 1:
                p = Pool(self.num_workers, maxtasksperchild=1)
                p.__enter__()
            with tqdm(total=len(protein_paths_chunk), desc=f'loading complexes {i}/{len(protein_paths_chunk)//1000+1}') as pbar:
                map_fn = p.imap_unordered if self.num_workers > 1 else map
                for t in map_fn(self.get_patch, zip(protein_paths_chunk, ligands_chunk, ligand_description_chunk)):
                    complex_graphs.extend(t[0])
                    rdkit_ligands.extend(t[1])
                    pbar.update()

            if self.num_workers > 1:
                p.__exit__(None, None, None)

            with open(os.path.join(self.full_cache_path, f"heterographs{i}.pkl"), 'wb') as f:
                pickle.dump((complex_graphs), f)
            with open(os.path.join(self.full_cache_path, f"rdkit_ligands{i}.pkl"), 'wb') as f:
                pickle.dump((rdkit_ligands), f)

    def check_all_complexes(self):
        if os.path.exists(os.path.join(self.full_cache_path, f"heterographs.pkl")):
            return True

        complex_names_all = glob.glob(self.split_path + "/*")
        complex_names_all.sort()
        if self.limit_complexes is not None and self.limit_complexes != 0:
            complex_names_all = complex_names_all[:self.limit_complexes]
        for i in range(len(complex_names_all)):
            if not os.path.exists(os.path.join(self.full_cache_path, f"heterographs{i}.pkl")):
                return False
        return True

    def collect_all_complexes(self):
        print('Collecting all complexes from cache', self.full_cache_path)
        if os.path.exists(os.path.join(self.full_cache_path, f"heterographs.pkl")):
            with open(os.path.join(self.full_cache_path, "heterographs.pkl"), 'rb') as f:
                patch_graphs = pickle.load(f)
            if self.require_ligand:
                with open(os.path.join(self.full_cache_path, "rdkit_ligands.pkl"), 'rb') as f:
                    rdkit_ligands = pickle.load(f)
            else:
                rdkit_ligands = None
            return patch_graphs, rdkit_ligands

        complex_names_all = glob.glob(self.split_path + "/*")
        complex_names_all.sort()

        if self.limit_complexes is not None and self.limit_complexes != 0:
            complex_names_all = complex_names_all[:self.limit_complexes]
        patch_graphs_all = []
        for i in range(len(complex_names_all)):
            with open(os.path.join(self.full_cache_path, f"heterographs{i}.pkl"), 'rb') as f:
                print(i)
                l = pickle.load(f)
                patch_graphs_all.extend(l)

        rdkit_ligands_all = []
        for i in range(len(complex_names_all)):
            with open(os.path.join(self.full_cache_path, f"rdkit_ligands{i}.pkl"), 'rb') as f:
                l = pickle.load(f)
                rdkit_ligands_all.extend(l)

        return patch_graphs_all, rdkit_ligands_all

    def get_patch(self, par):
        name, ligand, ligand_description = par
        if not os.path.exists(name) and ligand is None:
            print("Folder not found", name)
            return [], []

        try:
            lig = read_mol(name, remove_hs=False)

            patch_graph = HeteroData()
            patch_graph['name'] = f"{name.split('/')[-3]}<>{name.split('/')[-1].split('.')[0]}"
            get_lig_graph_with_matching(lig, patch_graph, self.popsize, self.maxiter, self.matching, self.keep_original,
                                        self.num_conformers, remove_hs=self.remove_hs, tries=self.matching_tries)
            ed_path_list = name.split('/')
            ed_path_list[-2] = 'extracted_point_clouds'

            if len(ed_path_list[-1].split('.')[-2]) == 1:
                ed_path_list[-1] = f"{ed_path_list[-1].split('.')[0]}.{ed_path_list[-1].split('.')[-2]}.csv"
            else:
                ed_path_list[-1] = f"{ed_path_list[-1].split('.')[0]}.csv"

            moad_extract_receptor_structure(path='/'.join(ed_path_list),
                                            complex_graph=patch_graph,
                                            neighbor_cutoff=self.receptor_radius,
                                            max_neighbors=self.c_alpha_max_neighbors,
                                            knn_only_graph=self.knn_only_graph)

        except Exception as e:
            print(f"Skipping {name.split('/')[-3]}<>{name.split('/')[-1].split('.')[0]} because of the error:")
            print(e)
            return [], []

        protein_center = torch.mean(patch_graph['patch_ed'].pos, dim=0, keepdim=True)
        patch_graph['patch_ed'].pos -= protein_center

        if (not self.matching) or self.num_conformers == 1:
            patch_graph['ligand'].pos -= protein_center
        else:
            for p in patch_graph['ligand'].pos:
                p -= protein_center

        patch_graph.original_center = protein_center
        patch_graph['protein_name'] = f"{name.split('/')[-3]}"
        return [patch_graph], [lig]


def print_statistics(patch_graphs):
    statistics = ([], [], [], [], [], [])
    patch_sizes = []

    for patch_graph in patch_graphs:
        lig_pos = patch_graph['ligand'].pos if torch.is_tensor(patch_graph['ligand'].pos) else patch_graph['ligand'].pos[0]
        patch_sizes.append(patch_graph['patch_ed'].pos.shape[0])
        amino_center = torch.mean(lig_pos, dim=0)
        radius_amino = torch.max(
            torch.linalg.vector_norm(lig_pos - amino_center.unsqueeze(0), dim=1))
        distance_center = torch.linalg.vector_norm(amino_center)
        statistics[0].extend(patch_sizes)
        statistics[1].append(radius_amino)
        statistics[2].append(distance_center)
        if "rmsd_matching" in patch_graph:
            statistics[3].append(patch_graph.rmsd_matching)
        else:
            statistics[3].append(0)
        statistics[4].append(int(patch_graph.random_coords) if "random_coords" in patch_graph else -1)
        if "random_coords" in patch_graph and patch_graph.random_coords and "rmsd_matching" in patch_graph:
            statistics[5].append(patch_graph.rmsd_matching)

    if len(statistics[5]) == 0:
        statistics[5].append(-1)
    name = ['Num patch nodes', 'Radius amino', 'Distance patch-amino',
            'RMSD matching', 'Random coordinates', 'Random RMSD matching']
    print('Number of patches: ', len(patch_graphs))
    for i in range(len(name)):
        array = np.asarray(statistics[i])
        print(f"{name[i]}: mean {np.mean(array)}, std {np.std(array)}, max {np.max(array)}")

    return


def read_mol(name, remove_hs=False):
    lig = read_molecule(name, remove_hs=remove_hs, sanitize=True)
    if lig is None:
        raise Exception('Could not read molecule')
    return lig


# TODO: check is this is used

# def read_mols(pdbbind_dir, name, remove_hs=False):
#     ligs = []
#     for file in os.listdir(os.path.join(pdbbind_dir, name)):
#         if file.endswith(".sdf") and 'rdkit' not in file:
#             lig = read_molecule(os.path.join(pdbbind_dir, name, file), remove_hs=remove_hs, sanitize=True)
#             if lig is None and os.path.exists(os.path.join(pdbbind_dir, name, file[:-4] + ".mol2")):  # read mol2 file if sdf file cannot be sanitized
#                 print('Using the .sdf file failed. We found a .mol2 file instead and are trying to use that.')
#                 lig = read_molecule(os.path.join(pdbbind_dir, name, file[:-4] + ".mol2"), remove_hs=remove_hs, sanitize=True)
#             if lig is not None:
#                 ligs.append(lig)
#     return ligs
