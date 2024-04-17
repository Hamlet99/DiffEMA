class DiffusionUtils:
    @staticmethod
    def t_to_sigma(t_tr, t_rot, t_tor, args):
        tr_sigma = args.tr_sigma_min ** (1 - t_tr) * args.tr_sigma_max**t_tr
        rot_sigma = args.rot_sigma_min ** (1 - t_rot) * args.rot_sigma_max**t_rot
        tor_sigma = args.tor_sigma_min ** (1 - t_tor) * args.tor_sigma_max**t_tor
        return tr_sigma, rot_sigma, tor_sigma
