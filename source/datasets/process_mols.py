import copy
import warnings
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem import AllChem, GetPeriodicTable, RemoveHs
from rdkit.Geometry import Point3D
from torch import cdist
from torch_cluster import knn_graph

import torch.nn.functional as F

from source.datasets.conformer_matching import get_torsion_angles, optimize_rotatable_bonds
from source.utils.torsion import get_transformation_mask

#TODO: remove redundant imports and functions
#TODO: remove unused functions
#TODO: add docstrings to functions
#TODO: rename modified functions to avoid conflicts

periodic_table = GetPeriodicTable()
allowable_features = {
    'possible_atomic_num_list': list(range(1, 119)) + ['misc'],
    'possible_chirality_list': [
        'CHI_UNSPECIFIED',
        'CHI_TETRAHEDRAL_CW',
        'CHI_TETRAHEDRAL_CCW',
        'CHI_OTHER'
    ],
    'possible_degree_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc'],
    'possible_numring_list': [0, 1, 2, 3, 4, 5, 6, 'misc'],
    'possible_implicit_valence_list': [0, 1, 2, 3, 4, 5, 6, 'misc'],
    'possible_formal_charge_list': [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 'misc'],
    'possible_numH_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
    'possible_number_radical_e_list': [0, 1, 2, 3, 4, 'misc'],
    'possible_hybridization_list': [
        'SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'misc'
    ],
    'possible_is_aromatic_list': [False, True],
    'possible_is_in_ring3_list': [False, True],
    'possible_is_in_ring4_list': [False, True],
    'possible_is_in_ring5_list': [False, True],
    'possible_is_in_ring6_list': [False, True],
    'possible_is_in_ring7_list': [False, True],
    'possible_is_in_ring8_list': [False, True],
    'possible_amino_acids': ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET',
                             'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL', 'HIP', 'HIE', 'TPO', 'HID', 'LEV', 'MEU',
                             'PTR', 'GLV', 'CYT', 'SEP', 'HIZ', 'CYM', 'GLM', 'ASQ', 'TYS', 'CYX', 'GLZ', 'misc'],
    'possible_atom_type_2': ['C*', 'CA', 'CB', 'CD', 'CE', 'CG', 'CH', 'CZ', 'N*', 'ND', 'NE', 'NH', 'NZ', 'O*', 'OD',
                             'OE', 'OG', 'OH', 'OX', 'S*', 'SD', 'SG', 'misc'],
    'possible_atom_type_3': ['C', 'CA', 'CB', 'CD', 'CD1', 'CD2', 'CE', 'CE1', 'CE2', 'CE3', 'CG', 'CG1', 'CG2', 'CH2',
                             'CZ', 'CZ2', 'CZ3', 'N', 'ND1', 'ND2', 'NE', 'NE1', 'NE2', 'NH1', 'NH2', 'NZ', 'O', 'OD1',
                             'OD2', 'OE1', 'OE2', 'OG', 'OG1', 'OH', 'OXT', 'SD', 'SG', 'misc'],
}
bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}

lig_feature_dims = (list(map(len, [
    allowable_features['possible_atomic_num_list'],
    allowable_features['possible_chirality_list'],
    allowable_features['possible_degree_list'],
    allowable_features['possible_formal_charge_list'],
    allowable_features['possible_implicit_valence_list'],
    allowable_features['possible_numH_list'],
    allowable_features['possible_number_radical_e_list'],
    allowable_features['possible_hybridization_list'],
    allowable_features['possible_is_aromatic_list'],
    allowable_features['possible_numring_list'],
    allowable_features['possible_is_in_ring3_list'],
    allowable_features['possible_is_in_ring4_list'],
    allowable_features['possible_is_in_ring5_list'],
    allowable_features['possible_is_in_ring6_list'],
    allowable_features['possible_is_in_ring7_list'],
    allowable_features['possible_is_in_ring8_list'],
])), 0)  # number of scalar features

rec_atom_feature_dims = (list(map(len, [
    allowable_features['possible_amino_acids'],
    allowable_features['possible_atomic_num_list'],
    allowable_features['possible_atom_type_2'],
    allowable_features['possible_atom_type_3'],
])), 0)

rec_residue_feature_dims = (list(map(len, [
    allowable_features['possible_amino_acids']
])), 0)


def lig_atom_featurizer(mol):
    ring_info = mol.GetRingInfo()
    atom_features_list = []
    for idx, atom in enumerate(mol.GetAtoms()):
        chiral_tag = str(atom.GetChiralTag())
        if chiral_tag in ['CHI_SQUAREPLANAR', 'CHI_TRIGONALBIPYRAMIDAL', 'CHI_OCTAHEDRAL']:
            chiral_tag = 'CHI_OTHER'

        atom_features_list.append([
            safe_index(allowable_features['possible_atomic_num_list'], atom.GetAtomicNum()),
            allowable_features['possible_chirality_list'].index(str(chiral_tag)),
            safe_index(allowable_features['possible_degree_list'], atom.GetTotalDegree()),
            safe_index(allowable_features['possible_formal_charge_list'], atom.GetFormalCharge()),
            safe_index(allowable_features['possible_implicit_valence_list'], atom.GetImplicitValence()),
            safe_index(allowable_features['possible_numH_list'], atom.GetTotalNumHs()),
            safe_index(allowable_features['possible_number_radical_e_list'], atom.GetNumRadicalElectrons()),
            safe_index(allowable_features['possible_hybridization_list'], str(atom.GetHybridization())),
            allowable_features['possible_is_aromatic_list'].index(atom.GetIsAromatic()),
            safe_index(allowable_features['possible_numring_list'], ring_info.NumAtomRings(idx)),
            allowable_features['possible_is_in_ring3_list'].index(ring_info.IsAtomInRingOfSize(idx, 3)),
            allowable_features['possible_is_in_ring4_list'].index(ring_info.IsAtomInRingOfSize(idx, 4)),
            allowable_features['possible_is_in_ring5_list'].index(ring_info.IsAtomInRingOfSize(idx, 5)),
            allowable_features['possible_is_in_ring6_list'].index(ring_info.IsAtomInRingOfSize(idx, 6)),
            allowable_features['possible_is_in_ring7_list'].index(ring_info.IsAtomInRingOfSize(idx, 7)),
            allowable_features['possible_is_in_ring8_list'].index(ring_info.IsAtomInRingOfSize(idx, 8)),
            # g_charge if not np.isnan(g_charge) and not np.isinf(g_charge) else 0.
        ])
    return torch.tensor(atom_features_list)


def safe_index(l, e):
    """Return index of element e in list l. If e is not present, return the last index."""
    return l.index(e) if e in l else len(l) - 1


def moad_extract_receptor_structure(path, complex_graph, neighbor_cutoff=20, max_neighbors=None,
                                    knn_only_graph=False):
    # load the electron density dataframe

    data = pd.read_csv(path)
    all_coords = np.array(data[['x', 'y', 'z']].values)

    complex_graph['patch_ed'].num_nodes = len(all_coords)
    complex_graph['patch_ed'].x = torch.tensor(data['i'].values.reshape(-1, 1), dtype=torch.float32)

    new_extract_receptor_structure(all_coords, complex_graph, neighbor_cutoff=neighbor_cutoff,
                                   max_neighbors=max_neighbors, knn_only_graph=knn_only_graph)


def new_extract_receptor_structure(all_coords, complex_graph, neighbor_cutoff=20, max_neighbors=None,
                                   knn_only_graph=False):

    # Build the k-NN graph
    coords = torch.tensor(all_coords, dtype=torch.float)
    if len(coords) > 4000:
        raise ValueError(f'The receptor is too large {len(coords)}')
    if knn_only_graph:
        edge_index = knn_graph(coords, k=max_neighbors if max_neighbors else 32)
    else:
        distances = cdist(coords, coords)
        src_list = []
        dst_list = []
        for i in range(len(coords)):
            dst = list(np.where(distances[i, :] < neighbor_cutoff)[0])
            dst.remove(i)
            max_neighbors = max_neighbors if max_neighbors else 32
            if max_neighbors is not None and len(dst) > max_neighbors:
                dst = list(np.argsort(distances[i, :]))[1: max_neighbors + 1]
            if len(dst) == 0:
                dst = list(np.argsort(distances[i, :]))[1:2]  # choose second because first is i itself
                print(
                    f'The cutoff {neighbor_cutoff} was too small for one atom such that it had no neighbors. '
                    f'So we connected it to the closest other atom')
            assert i not in dst
            src = [i] * len(dst)
            src_list.extend(src)
            dst_list.extend(dst)
        edge_index = torch.from_numpy(np.asarray([dst_list, src_list]))

    complex_graph['patch_ed'].pos = coords
    # complex_graph['receptor'].side_chain_vecs = side_chain_vecs.float()
    complex_graph['patch_ed', 'patch_contact', 'patch_ed'].edge_index = edge_index

    return


def get_lig_graph(mol, patch_graph):
    atom_feats = lig_atom_featurizer(mol)

    row, col, edge_type = [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        edge_type += 2 * [bonds[bond.GetBondType()]] if bond.GetBondType() != BT.UNSPECIFIED else [0, 0]

    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_type = torch.tensor(edge_type, dtype=torch.long)
    edge_attr = F.one_hot(edge_type, num_classes=len(bonds)).to(torch.float)

    patch_graph['ligand'].x = atom_feats
    patch_graph['ligand', 'lig_bond', 'ligand'].edge_index = edge_index
    patch_graph['ligand', 'lig_bond', 'ligand'].edge_attr = edge_attr

    if mol.GetNumConformers() > 0:
        lig_coords = torch.from_numpy(mol.GetConformer().GetPositions()).float()
        patch_graph['ligand'].pos = lig_coords

    return


def generate_conformer(mol):
    ps = AllChem.ETKDGv2()
    failures, id = 0, -1
    while failures < 3 and id == -1:
        if failures > 0:
            print(f'rdkit coords could not be generated. trying again {failures}.')
        id = AllChem.EmbedMolecule(mol, ps)
        failures += 1
    if id == -1:
        print('rdkit coords could not be generated without using random coords. using random coords now.')
        ps.useRandomCoords = True
        AllChem.EmbedMolecule(mol, ps)
        AllChem.MMFFOptimizeMolecule(mol, confId=0)
        return True
    # else:
    #    AllChem.MMFFOptimizeMolecule(mol, confId=0)
    return False


def get_lig_graph_with_matching(mol_, patch_graph, popsize, maxiter, matching, keep_original, num_conformers,
                                remove_hs, tries=10, skip_matching=False):
    if matching:
        mol_maybe_noh = copy.deepcopy(mol_)
        if remove_hs:
            mol_maybe_noh = RemoveHs(mol_maybe_noh, sanitize=True)
            mol_maybe_noh = AllChem.RemoveAllHs(mol_maybe_noh)
        if keep_original:
            positions = []
            for conf in mol_maybe_noh.GetConformers():
                positions.append(conf.GetPositions())
            patch_graph['ligand'].orig_pos = np.asarray(positions) if len(positions) > 1 else positions[0]

        rotatable_bonds = get_torsion_angles(mol_maybe_noh)

        # if not rotatable_bonds: print("no_rotatable_bonds but still using it")

        for i in range(num_conformers):
            mols, rmsds = [], []
            for _ in range(tries):
                mol_rdkit = copy.deepcopy(mol_)

                mol_rdkit.RemoveAllConformers()
                mol_rdkit = AllChem.AddHs(mol_rdkit)
                generate_conformer(mol_rdkit)
                if remove_hs:
                    mol_rdkit = RemoveHs(mol_rdkit, sanitize=True)
                mol_rdkit = AllChem.RemoveAllHs(mol_rdkit)
                mol = AllChem.RemoveAllHs(copy.deepcopy(mol_maybe_noh))
                if rotatable_bonds and not skip_matching:
                    optimize_rotatable_bonds(mol_rdkit, mol, rotatable_bonds, popsize=popsize, maxiter=maxiter)
                mol.AddConformer(mol_rdkit.GetConformer())
                rms_list = []
                AllChem.AlignMolConformers(mol, RMSlist=rms_list)
                mol_rdkit.RemoveAllConformers()
                mol_rdkit.AddConformer(mol.GetConformers()[1])
                mols.append(mol_rdkit)
                rmsds.append(rms_list[0])

            # select molecule with lowest rmsd
            # print("mean std min max", np.mean(rmsds), np.std(rmsds), np.min(rmsds), np.max(rmsds))
            mol_rdkit = mols[np.argmin(rmsds)]
            if i == 0:
                patch_graph.rmsd_matching = min(rmsds)
                get_lig_graph(mol_rdkit, patch_graph)
            else:
                if torch.is_tensor(patch_graph['ligand'].pos):
                    patch_graph['ligand'].pos = [patch_graph['ligand'].pos]
                patch_graph['ligand'].pos.append(torch.from_numpy(mol_rdkit.GetConformer().GetPositions()).float())

    else:  # no matching
        patch_graph.rmsd_matching = 0
        if remove_hs:
            mol_ = RemoveHs(mol_)
        get_lig_graph(mol_, patch_graph)

    edge_mask, mask_rotate = get_transformation_mask(patch_graph)
    patch_graph['ligand'].edge_mask = torch.tensor(edge_mask)
    patch_graph['ligand'].mask_rotate = mask_rotate

    return


def write_mol_with_coords(mol, new_coords, path):
    w = Chem.SDWriter(path)
    conf = mol.GetConformer()
    for i in range(mol.GetNumAtoms()):
        x, y, z = new_coords.astype(np.double)[i]
        conf.SetAtomPosition(i, Point3D(x, y, z))
    w.write(mol)
    w.close()


def read_molecule(molecule_file, sanitize=False, calc_charges=False, remove_hs=False):

    if molecule_file.endswith('.pdb'):
        mol = Chem.MolFromPDBFile(molecule_file, sanitize=False, removeHs=False)
    else:
        return ValueError('Expect the format of the molecule_file to be '
                          '.pdb, got {}'.format(molecule_file))

    try:
        if sanitize or calc_charges:
            Chem.SanitizeMol(mol)

        if calc_charges:
            # Compute Gasteiger charges on the molecule.
            try:
                AllChem.ComputeGasteigerCharges(mol)
            except:
                warnings.warn('Unable to compute charges for the molecule.')

        if remove_hs:
            mol = Chem.RemoveHs(mol, sanitize=sanitize)
    except:
        return None

    return mol
