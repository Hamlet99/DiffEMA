from argparse import ArgumentParser, FileType


def parse_train_args():
    # General arguments
    parser = ArgumentParser()
    parser.add_argument('--config', type=FileType(mode='r'), default=None)
    parser.add_argument('--log_dir', type=str, default='workdir/test_score',
                        help='Folder in which to save model and logs')
    parser.add_argument('--restart_dir', type=str, help='Folder of previous training model from which to restart')
    parser.add_argument('--restart_ckpt', type=str, default='last_model', help='')
    parser.add_argument('--pretrain_dir', type=str, help='Folder of pretrained model from which to restart')
    parser.add_argument('--pretrain_ckpt', type=str, help='')
    parser.add_argument('--freeze_params', type=int, default=0, help='')
    parser.add_argument('--cache_path', type=str, default='data/cache',
                        help='Folder from where to load/restore cached dataset')
    parser.add_argument('--pdbbind_dir', type=str, default='data/PDBBind_processed/',
                        help='Folder containing original structures')
    parser.add_argument('--dataset', type=str, default='pdbbind', help='Folder containing original structures')  # remove
    parser.add_argument('--split_train', type=str, default='training_data_preprocessed/train',
                        help='Path of file defining the split')
    parser.add_argument('--split_val', type=str, default='training_data_preprocessed/val',
                        help='Path of file defining the split')
    parser.add_argument('--split_test', type=str, default='training_data_preprocessed/test',
                        help='Path of file defining the split')
    parser.add_argument('--test_sigma_intervals', action='store_true', default=False,
                        help='Whether to log loss per noise interval')
    parser.add_argument('--val_inference_freq', type=int, default=5,
                        help='Frequency of epochs for which to run expensive inference on val data')
    parser.add_argument('--save_model_freq', type=int, default=None, help='')
    parser.add_argument('--inference_samples', type=int, default=1, help='')
    parser.add_argument('--train_inference_freq', type=int, default=None,
                        help='Frequency of epochs for which to run expensive inference on train data')
    parser.add_argument('--inference_steps', type=int, default=20,
                        help='Number of denoising steps for inference on val')
    parser.add_argument('--num_inference_complexes', type=int, default=2,  # optimize
                        help='Number of complexes for which inference is run every val/train_inference_freq epochs (None will run it on all)')
    parser.add_argument('--inference_earlystop_metric', type=str, default='valinf_min_rmsds_lt2',
                        help='This is the metric that is addionally used when val_inference_freq is not None')
    parser.add_argument('--inference_secondary_metric', type=str, default=None, help='')
    parser.add_argument('--inference_earlystop_goal', type=str, default='max',
                        help='Whether to maximize or minimize metric')
    parser.add_argument('--wandb', action='store_true', default=False, help='')
    parser.add_argument('--project', type=str, default='DiffEMA', help='')
    parser.add_argument('--run_name', type=str, default='test_run', help='')
    parser.add_argument('--cudnn_benchmark', action='store_true', default=False,
                        help='CUDA optimization parameter for faster training')
    parser.add_argument('--num_dataloader_workers', type=int, default=0, help='Number of workers for dataloader')
    parser.add_argument('--pin_memory', action='store_true', default=False, help='pin_memory arg of dataloader')
    parser.add_argument('--dataloader_drop_last', action='store_true', default=False,
                        help='drop_last arg of dataloader')

    # Training arguments
    parser.add_argument('--n_epochs', type=int, default=400, help='Number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--scheduler', type=str, default=None, help='LR scheduler')
    parser.add_argument('--scheduler_patience', type=int, default=20, help='Patience of the LR scheduler')
    parser.add_argument('--lr_start_factor', type=float, default=0.001, help='')
    parser.add_argument('--warmup_dur', type=int, default=4, help='')
    parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate')
    parser.add_argument('--restart_lr', type=float, default=None,
                        help='If this is not none, the lr of the optimizer will be overwritten with this value when restarting from a checkpoint.')
    parser.add_argument('--w_decay', type=float, default=0.0, help='Weight decay added to loss')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of workers for preprocessing')
    parser.add_argument('--use_ema', action='store_true', default=False,
                        help='Whether or not to use ema for the model weights')
    parser.add_argument('--ema_rate', type=float, default=0.999,
                        help='decay rate for the exponential moving average model parameters ')

    # Dataset
    parser.add_argument('--limit_complexes', type=int, default=2,
                        help='If positive, the number of training and validation complexes is capped')  # TODO change
    parser.add_argument('--all_atoms', action='store_true', default=False, help='Whether to use the all atoms model')  # remove
    parser.add_argument('--receptor_radius', type=float, default=1, help='Cutoff on distances for receptor edges')
    parser.add_argument('--c_alpha_max_neighbors', type=int, default=10,
                        help='Maximum number of neighbors for each residue')
    parser.add_argument('--atom_radius', type=float, default=1, help='Cutoff on distances for atom connections')  # remove
    parser.add_argument('--atom_max_neighbors', type=int, default=8,
                        help='Maximum number of atom neighbours for receptor')  # remove
    parser.add_argument('--matching_popsize', type=int, default=20,
                        help='Differential evolution popsize parameter in matching')
    parser.add_argument('--matching_maxiter', type=int, default=20,
                        help='Differential evolution maxiter parameter in matching')
    parser.add_argument('--matching_tries', type=int, default=1, help='')
    parser.add_argument('--remove_hs', action='store_true', default=True, help='remove Hs')
    parser.add_argument('--num_conformers', type=int, default=1, help='Number of conformers to match to each ligand')
    parser.add_argument('--moad_esm_embeddings_path', type=str, default=None,
                        help='If this is set then the LM embeddings at that path will be used for the receptor features')  # remove
    parser.add_argument('--pdbbind_esm_embeddings_path', type=str, default=None,
                        help='If this is set then the LM embeddings at that path will be used for the receptor features')  # remove
    parser.add_argument('--moad_esm_embeddings_sequences_path', type=str, default=None, help='')  # remove
    parser.add_argument('--esm_embeddings_model', type=str, default=None, help='')  # remove
    parser.add_argument('--not_fixed_knn_radius_graph', action='store_true', default=False,
                        help='Use knn graph and radius graph with closest neighbors instead of random ones as with radius_graph')
    parser.add_argument('--not_knn_only_graph', action='store_true', default=False,
                        help='Use knn graph only and not restrict to a specific radius')
    parser.add_argument('--train_multiplicity', type=int, default=1, help='')
    parser.add_argument('--val_multiplicity', type=int, default=1, help='')
    parser.add_argument('--max_receptor_size', type=int, default=None, help='')  # remove
    parser.add_argument('--remove_promiscuous_targets', type=int, default=None, help='')  # remove
    parser.add_argument('--unroll_clusters', action='store_true', default=False, help='')
    parser.add_argument('--enforce_timesplit', action='store_true', default=False, help='')
    parser.add_argument('--merge_clusters', type=int, default=1, help='')
    parser.add_argument('--triple_training', action='store_true', default=False, help='')
    parser.add_argument('--crop_beyond', type=float, default=20, help='')  # remove

    # Diffusion
    parser.add_argument('--tr_weight', type=float, default=0.33, help='Weight of translation loss')
    parser.add_argument('--rot_weight', type=float, default=0.33, help='Weight of rotation loss')
    parser.add_argument('--tor_weight', type=float, default=0.33, help='Weight of torsional loss')
    parser.add_argument('--confidence_weight', type=float, default=0.33, help='Weight of confidence loss')
    parser.add_argument('--rot_sigma_min', type=float, default=0.1, help='Minimum sigma for rotational component')
    parser.add_argument('--rot_sigma_max', type=float, default=1.65, help='Maximum sigma for rotational component')
    parser.add_argument('--tr_sigma_min', type=float, default=0.1, help='Minimum sigma for translational component')
    parser.add_argument('--tr_sigma_max', type=float, default=30, help='Maximum sigma for translational component')
    parser.add_argument('--tor_sigma_min', type=float, default=0.0314, help='Minimum sigma for torsional component')
    parser.add_argument('--tor_sigma_max', type=float, default=3.14, help='Maximum sigma for torsional component')
    parser.add_argument('--no_torsion', action='store_true', default=False, help='If set only rigid matching')
    parser.add_argument('--sampling_alpha', type=float, default=1,
                        help='Alpha parameter of beta distribution for sampling t')
    parser.add_argument('--sampling_beta', type=float, default=1,
                        help='Beta parameter of beta distribution for sampling t')
    parser.add_argument('--bootstrap_alpha', type=float, default=1,
                        help='Alpha parameter of beta distribution for sampling t in bootstrapping')
    parser.add_argument('--bootstrap_beta', type=float, default=1,
                        help='Beta parameter of beta distribution for sampling t in bootstrapping')
    parser.add_argument('--bootstrap_tmin', type=float, default=0, help='')

    # Model
    parser.add_argument('--num_conv_layers', type=int, default=2, help='Number of interaction layers')
    parser.add_argument('--max_radius', type=float, default=2.0, help='Radius cutoff for geometric graph')
    parser.add_argument('--scale_by_sigma', action='store_true', default=True, help='Whether to normalise the score')
    parser.add_argument('--norm_by_sigma', action='store_true', default=False, help='Whether to normalise the score')
    parser.add_argument('--ns', type=int, default=16, help='Number of hidden features per node of order 0')
    parser.add_argument('--nv', type=int, default=4, help='Number of hidden features per node of order >0')
    parser.add_argument('--distance_embed_dim', type=int, default=32, help='Embedding size for the distance')
    parser.add_argument('--cross_distance_embed_dim', type=int, default=32,
                        help='Embeddings size for the cross distance')
    parser.add_argument('--no_batch_norm', action='store_true', default=False, help='If set, it removes the batch norm')
    parser.add_argument('--use_second_order_repr', action='store_true', default=False,
                        help='Whether to use only up to first order representations or also second')
    parser.add_argument('--cross_max_distance', type=float, default=80,
                        help='Maximum cross distance in case not dynamic')
    parser.add_argument('--dynamic_max_cross', action='store_true', default=False,
                        help='Whether to use the dynamic distance cutoff')
    parser.add_argument('--dropout', type=float, default=0.0, help='MLP dropout')
    parser.add_argument('--smooth_edges', action='store_true', default=False,
                        help='Whether to apply additional smoothing weight to edges')
    parser.add_argument('--odd_parity', action='store_true', default=False,
                        help='Whether to impose odd parity in output')
    parser.add_argument('--embedding_type', type=str, default="sinusoidal", help='Type of diffusion time embedding')
    parser.add_argument('--sigma_embed_dim', type=int, default=32, help='Size of the embedding of the diffusion time')
    parser.add_argument('--embedding_scale', type=int, default=1000, help='Parameter of the diffusion time embedding')
    parser.add_argument('--use_old_atom_encoder', action='store_true', default=False,
                        help='option to use old atom encoder for backward compatibility')  # remove
    parser.add_argument('--depthwise_convolution', action='store_true', default=False, help='')

    parser.add_argument('--protein_file', type=str, default='protein_processed', help='')
    parser.add_argument('--no_aminoacid_identities', action='store_true', default=False, help='')
    parser.add_argument('--sh_lmax', type=int, default=2, help='Size of the embedding of the diffusion time')
    parser.add_argument('--no_differentiate_convolutions', action='store_true', default=False, help='')
    parser.add_argument('--tp_weights_layers', type=int, default=2, help='')
    parser.add_argument('--num_prot_emb_layers', type=int, default=0, help='')
    parser.add_argument('--reduce_pseudoscalars', action='store_true', default=False, help='')
    parser.add_argument('--embed_also_ligand', action='store_true', default=True, help='')
    parser.add_argument('--sidechain_loss_weight', type=float, default=0, help='')
    parser.add_argument('--backbone_loss_weight', type=float, default=0, help='')

    args = parser.parse_args()

    # _____ DiffEMA ______

    parser.add_argument('--patch_sampling_radius', default=7,
                        help="Radius of the patch sampling in Angstrom from the CA atom")
    parser.add_argument('--output_directory', default=None,
                        help="Directory where the preprocessed files will be saved")
    parser.add_argument('--point_cloud_sigma_level', default=1.0, help="Sigma level for the point cloud")

    assert (not args.dynamic_max_cross) or (args.tr_sigma_max * 3 + 20 < args.cross_max_distance)
    assert args.esm_embeddings_model is None or args.esm_embeddings_path is None  # remove
    return args
