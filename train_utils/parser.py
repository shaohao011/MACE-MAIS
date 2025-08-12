import argparse

def get_args():
    config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
    parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                        help='YAML config file specifying default arguments')
    parser = argparse.ArgumentParser(description='LLD-MMRI 2023 Training')
    # Dataset parameters
    parser.add_argument('--data_dir', default='', type=str)
    parser.add_argument('--train_anno_file', default='', type=str)
    parser.add_argument('--val_anno_file', default='', type=str)
    parser.add_argument('--train_transform_list', default=[
                                                        'random_crop', 
                                                        'z_flip', 
                                                        'x_flip', 
                                                        'y_flip', 
                                                        'rotation'], 
                                                        nargs='+', type=str)
    parser.add_argument('--val_transform_list', default=['center_crop'], nargs='+', type=str)
    parser.add_argument('--img_size', default=(16, 128, 128), type=int, nargs='+', help='input image size.')
    parser.add_argument('--img_patch_size', default=(14, 112, 112), type=int, nargs='+', help='cropped image size.')
    parser.add_argument('--flip_prob', default=0.5, type=float, help='Random flip prob (default: 0.5)')
    parser.add_argument('--reprob', type=float, default=0.25, help='Random erase prob (default: 0.25)')
    parser.add_argument('--rcprob', type=float, default=0.25, help='Random contrast prob (default: 0.25)')
    parser.add_argument('--angle', default=45, type=int)

    # Model parameters
    parser.add_argument('--model', default='mp_uniformer_small', type=str, metavar='MODEL',
                        help='Name of model to train (default: "resnet50"')
    parser.add_argument('--pretrained', action='store_true', default=False,
                        help='Start with pretrained version of specified network (if avail)')
    parser.add_argument('--initial-checkpoint', default='', type=str, metavar='PATH',
                        help='Initialize model from this checkpoint (default:]')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='Resume full model and optimizer state from checkpoint (default:]')
    parser.add_argument('--no-resume-opt', action='store_true', default=False,
                        help='prevent resume of optimizer state when resuming model')
    parser.add_argument('--out_channels', type=int, default=7, metavar='N',
                        help='number of label classes (Model default if]')
    parser.add_argument('--gp', default=None, type=str, metavar='POOL',
                        help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')
    parser.add_argument('--interpolation', default='', type=str, metavar='NAME',
                        help='Image resize interpolation type (overrides model)')
    parser.add_argument('-b', '--batch_size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 2)')
    parser.add_argument('-vb', '--validation-batch-size', type=int, default=None, metavar='N',
                        help='validation batch size override (default:]')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=None, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: None, use opt default)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='Optimizer momentum (default: 0.9)')
    parser.add_argument('--reg_weight', type=float, default=1e-5,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--clip-mode', type=str, default='norm',
                        help='Gradient clipping mode. One of ("norm", "value", "agc")')

    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "step"')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--lr-cycle-mul', type=float, default=1.0, metavar='MULT',
                        help='learning rate cycle len multiplier (default: 1.0)')
    parser.add_argument('--lr-cycle-decay', type=float, default=0.5, metavar='MULT',
                        help='amount to decay each learning rate cycle (default: 0.5)')
    parser.add_argument('--lr-cycle-limit', type=int, default=1, metavar='N',
                        help='learning rate cycle limit, cycles enabled if > 1')
    parser.add_argument('--lr-k-decay', type=float, default=1.0,
                        help='learning rate k-decay for cosine/poly (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--max_epochs', type=int, default=300, metavar='N',
                        help='number of epochs to train (default: 300)')
    parser.add_argument('--epoch-repeats', type=float, default=0., metavar='N',
                        help='epoch repeat multiplier (number of times to repeat dataset epoch per train epoch).')
    parser.add_argument('--start-epoch', default=None, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--decay-epochs', type=float, default=100, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Regularization parameters
    parser.add_argument('--bce-loss', action='store_true', default=False,
                        help='Enable BCE loss w/ Mixup/CutMix use.')
    parser.add_argument('--bce-target-thresh', type=float, default=None,
                        help='Threshold for binarizing softened BCE targets (default: None, disabled)')

    # Misc
    parser.add_argument('--random_seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')


    parser.add_argument('-j', '--num_workers', type=int, default=6, metavar='N',
                        help='how many training processes to use (default: 8)')

    parser.add_argument('--logdir', default='', type=str, metavar='PATH',
                        help='path to output folder (default: none, current dir)')
    parser.add_argument('--lrschedule', default='cosine_anneal', type=str, metavar='PATH',
                        help='path to output folder (default: none, current dir)')

    parser.add_argument('--optim_name', default='adamw', type=str, metavar='PATH',
                        help='path to output folder (default: none, current dir)')
    parser.add_argument('--optim_lr', default=3e-4, type=float, metavar='PATH',
                        help='path to output folder (default: none, current dir)')

    parser.add_argument('--eval-metric', default='f1', type=str, metavar='EVAL_METRIC',
                        help='Main metric (default: "f1"')
    parser.add_argument('--report-metrics', default=['acc', 'f1', 'recall', 'precision', 'kappa'], 
                        nargs='+', choices=['acc', 'f1', 'recall', 'precision', 'kappa'], 
                        type=str, help='All evaluation metrics')
    parser.add_argument('--pixelspacing', default=[1.0,1.0,1.0], 
                        nargs='+', 
                        type=float, help='All evaluation metrics')

    parser.add_argument("--local-rank", default=-1, type=int)
    parser.add_argument("--rank", default=0, type=int)

    # Data mixup configs
    parser.add_argument('--is_mixup', action='store_true', default=False,
                        help='unable mixup augmentation')
    parser.add_argument('--alpha', type=float, default=0.6,
                        help='param of beta distribution for data mixup')  

    # Channel cutout configs
    parser.add_argument("--cutcnum", default=1, type=int, help='numbers of random cut channels.')
    parser.add_argument('--cutcprob', type=float, default=0.25, help='Random channel-cut prob (default: 0.25)')
    parser.add_argument('--cutcmode', default='zeros', type=str)

    parser.add_argument("--in_channels", default=3, type=int, help='numbers of input channel.')
    parser.add_argument('--is-meanstd-norm', action='store_true', default=False,
                        help='If true, apply mean-std normalization when loading image.')

    parser.add_argument('--simpl-phase', default=[], nargs='+', type=str)
    parser.add_argument('--allow-missing-modality', action='store_true', default=False,
                        help='If true, pad missing series with zero-maps.')

    parser.add_argument('--distributed', action='store_true', default=False,
                        help='If true, pad missing series with zero-maps.')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='')

    parser.add_argument('--cache_dataset', action='store_true', default=False,
                        help='If true, pad missing series with zero-maps.')
    parser.add_argument('--model_inversion', action='store_true', default=False,
                        help='If true, pad missing series with zero-maps.')

    parser.add_argument("--max_mask_aug", default=0, type=int, help='if >0, apply mask (<=N) channel.')
    parser.add_argument("--prob_mask_aug", default=0.5, type=int, help='probability to apply mask augmentation.')

    parser.add_argument('--json_dir', default='./jsons', type=str)
    parser.add_argument('--concat_mask', action='store_true', default=False,)
    parser.add_argument('--initial_checkpoint', default='', type=str)
    parser.add_argument('--spatial_dim', default='', type=int)
    parser.add_argument('--loss_type', default='asl', type=str)
    parser.add_argument('--model_type', default='conv', type=str)
    parser.add_argument('--use_text_inputs', action='store_true', default=False,)
    parser.add_argument('--use_lora', action='store_true', default=False,)
    parser.add_argument('--pure_text_inputs', action='store_true', default=False,)
    parser.add_argument('--text_inputs_keys', default=['Imaging_Findings'], type=str, nargs='+', help='input image size.')
    

    parser.add_argument("--fold", default=0, type=int, help='probability to apply mask augmentation.')
    parser.add_argument("--n_folds", default=0, type=int, help='probability to apply mask augmentation.')
    parser.add_argument("--model_index", default=-1, type=int, help='probability to apply mask augmentation.')
    parser.add_argument("--llm_name", default="clinicalBERT", type=str, help='probability to apply mask augmentation.')
    parser.add_argument("--dataset_name", default="renji", type=str, help='probability to apply mask augmentation.')
    parser.add_argument('--dataset', default=[], type=str, nargs='+')
    parser.add_argument("--img_rpt_path", default="", type=str, help='probability to apply mask augmentation.')
    
    return parser