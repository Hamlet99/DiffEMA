import torch
from torch_geometric.data import Dataset

from source.datasets.dataloader import DataLoader, DataListLoader
from source.datasets.general_dataset import NoiseTransform, PDBBind


def construct_loader(args, t_to_sigma, device):
    val_dataset2 = None
    transform = NoiseTransform(t_to_sigma=t_to_sigma, no_torsion=args.no_torsion,
                               all_atom=args.all_atoms, alpha=args.sampling_alpha, beta=args.sampling_beta,
                               include_miscellaneous_atoms=False,)

    common_args = {'transform': transform,
                   'limit_complexes': args.limit_complexes,
                   'receptor_radius': args.receptor_radius,
                   'c_alpha_max_neighbors': args.c_alpha_max_neighbors,
                   'remove_hs': args.remove_hs,
                   'matching': not args.no_torsion,
                   'popsize': args.matching_popsize,
                   'maxiter': args.matching_maxiter,
                   'num_workers': args.num_workers,
                   'all_atoms': args.all_atoms,
                   'atom_radius': args.atom_radius,
                   'atom_max_neighbors': args.atom_max_neighbors,
                   'knn_only_graph': False if not hasattr(args, 'not_knn_only_graph') else not args.not_knn_only_graph,
                   'include_miscellaneous_atoms': False if not hasattr(args, 'include_miscellaneous_atoms') else args.include_miscellaneous_atoms,
                   'matching_tries': args.matching_tries}

    train_dataset = PDBBind(cache_path=args.cache_path, split_path=args.split_train, keep_original=True,
                            num_conformers=args.num_conformers, root=args.pdbbind_dir, protein_file=args.protein_file,
                            **common_args)

    val_dataset = PDBBind(cache_path=args.cache_path, split_path=args.split_val, keep_original=True,
                          root=args.pdbbind_dir, protein_file=args.protein_file, require_ligand=True, **common_args)

    loader_class = DataListLoader if torch.cuda.is_available() else DataLoader

    train_loader = loader_class(dataset=train_dataset, batch_size=args.batch_size,
                                num_workers=args.num_dataloader_workers, shuffle=True, pin_memory=args.pin_memory,
                                drop_last=args.dataloader_drop_last)

    val_loader = loader_class(dataset=val_dataset, batch_size=args.batch_size, num_workers=args.num_dataloader_workers,
                              shuffle=False, pin_memory=args.pin_memory, drop_last=args.dataloader_drop_last)

    return train_loader, val_loader, val_dataset2, train_dataset, val_dataset

