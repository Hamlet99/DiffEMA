import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from scipy.stats import beta

from source.utils.geometry import axis_angle_to_matrix, rigid_transform_Kabsch_3D_torch, rigid_transform_Kabsch_3D_torch_batch
from source.utils.torsion import modify_conformer_torsion_angles, modify_conformer_torsion_angles_batch


def t_to_sigma(t_tr, t_rot, t_tor, args):
    tr_sigma = args.tr_sigma_min ** (1 - t_tr) * args.tr_sigma_max**t_tr
    rot_sigma = args.rot_sigma_min ** (1 - t_rot) * args.rot_sigma_max**t_rot
    tor_sigma = args.tor_sigma_min ** (1 - t_tor) * args.tor_sigma_max**t_tor
    return tr_sigma, rot_sigma, tor_sigma


def modify_conformer(data, tr_update, rot_update, torsion_updates, pivot=None):
    lig_center = torch.mean(data['ligand'].pos, dim=0, keepdim=True)
    rot_mat = axis_angle_to_matrix(rot_update.squeeze())
    rigid_new_pos = (data['ligand'].pos - lig_center) @ rot_mat.T + tr_update + lig_center

    if torsion_updates is not None:
        flexible_new_pos = modify_conformer_torsion_angles(rigid_new_pos,
                                                           data['ligand', 'ligand'].edge_index.T[
                                                               data['ligand'].edge_mask],
                                                           data['ligand'].mask_rotate if isinstance(
                                                               data['ligand'].mask_rotate, np.ndarray) else
                                                           data['ligand'].mask_rotate[0],
                                                           torsion_updates).to(rigid_new_pos.device)
        if pivot is None:
            R, t = rigid_transform_Kabsch_3D_torch(flexible_new_pos.T, rigid_new_pos.T)
            aligned_flexible_pos = flexible_new_pos @ R.T + t.T
        else:
            R1, t1 = rigid_transform_Kabsch_3D_torch(pivot.T, rigid_new_pos.T)
            R2, t2 = rigid_transform_Kabsch_3D_torch(flexible_new_pos.T, pivot.T)

            aligned_flexible_pos = (flexible_new_pos @ R2.T + t2.T) @ R1.T + t1.T

        data['ligand'].pos = aligned_flexible_pos
    else:
        data['ligand'].pos = rigid_new_pos
    return data


def set_time(patch_graphs, t, t_tr, t_rot, t_tor, batchsize, all_atoms, device, include_miscellaneous_atoms=False):
    patch_graphs['ligand'].node_t = {
        'tr': t_tr * torch.ones(patch_graphs['ligand'].num_nodes).to(device),
        'rot': t_rot * torch.ones(patch_graphs['ligand'].num_nodes).to(device),
        'tor': t_tor * torch.ones(patch_graphs['ligand'].num_nodes).to(device)}
    patch_graphs['patch_ed'].node_t = {
        'tr': t_tr * torch.ones(patch_graphs['patch_ed'].num_nodes).to(device),
        'rot': t_rot * torch.ones(patch_graphs['patch_ed'].num_nodes).to(device),
        'tor': t_tor * torch.ones(patch_graphs['patch_ed'].num_nodes).to(device)}
    patch_graphs.complex_t = {'tr': t_tr * torch.ones(batchsize).to(device),
                              'rot': t_rot * torch.ones(batchsize).to(device),
                              'tor': t_tor * torch.ones(batchsize).to(device)}
