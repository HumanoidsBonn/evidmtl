# ------------------------------------------------------------------------------
# The code is from GLPDepth (https://github.com/vinvino02/GLPDepth).
# For non-commercial purpose only (research, evaluation etc).
# Modified by Zigang Geng (zigang@mail.ustc.edu.cn).
# ------------------------------------------------------------------------------

import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class BaseOptions():
    def __init__(self):
        pass

    def initialize(self):
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        # base configs
        parser.add_argument('--exp_name',   type=str, default='')
        parser.add_argument('--gpu_or_cpu',   type=str, default='gpu')
        parser.add_argument('--data_path',    type=str, default='/data/ssd1/')
        parser.add_argument('--dataset',      type=str, default='',
                            choices=['nyudepthv2', 'nyudepthv2labels40', 'nyudepthv2labels20scannet', 'nyudepthv2labels22scannet_mod', 'scannet_v2', 'kitti', 'imagepath', 'apolloscape', 'cityscapes'])
                        
        parser.add_argument('--num_classes', type = int, default= 20) 
                            

        parser.add_argument('--batch_size',   type=int, default=4) #24
        parser.add_argument('--workers',      type=int, default=4) #8
        
        # depth configs
        parser.add_argument('--max_depth',      type=float, default=10.0)
        parser.add_argument('--max_depth_eval', type=float, default=11.0)
        parser.add_argument('--min_depth_eval', type=float, default=0.0001)        
        parser.add_argument('--do_kb_crop',     type=int, default=1)
        parser.add_argument('--kitti_crop', type=str, default=None,
                            choices=['garg_crop', 'eigen_crop'])

        parser.add_argument('--backbone',   type=str, default='swin_base_v2')
        parser.add_argument('--pretrained',    type=str, default='')
        parser.add_argument('--window_size', nargs='+', type=int)
        parser.add_argument('--pretrain_window_size', nargs='+', type=int)
        parser.add_argument('--use_shift', nargs='+', type=str2bool)
        parser.add_argument('--depths', nargs='+', type=int)
        parser.add_argument('--drop_path_rate',     type=float, default=0.3)
        parser.add_argument('--use_checkpoint',   type=str2bool, default='False')
        parser.add_argument('--num_deconv',     type=int, default=3)
        parser.add_argument('--num_filters', nargs='+', type=int)
        parser.add_argument('--deconv_kernels', nargs='+', type=int)

        parser.add_argument('--shift_window_test', action='store_true')     
        parser.add_argument('--shift_size',  type=int, default=2)
        parser.add_argument('--flip_test', action='store_true')    

        parser.add_argument('--seg_evidence_type', type=str, default='softplus') 
        parser.add_argument('--depth_evidence_type', type=str, default='softplus') 
        parser.add_argument('--evidential_depth_loss', type=str, default='nig_nll',
                    choices=['nig_nll', 'evidential_silog', 'evidential_sumsquares', 'combined_silog_nll_loss'],
                    help="Choose the depth loss function: Negative Likelihood Loss or Evidential SiLog loss').")
        parser.add_argument('--vanilla_ckpt', type=str, default=None)
        parser.add_argument('--evidential_seg_loss', type=str, default='evidential_cross_entropy',
                            choices = ['evidential_cross_entropy', 'evidential_focal_cross_entropy', 'evidential_dice_cross_entropy' , 'evidential_classification_loss'],
                            help="Choose the segmentation loss function: 'evidential_cross_entropy' or 'evidential_classification_loss'.")
        parser.add_argument('--evidential_seg_reg_loss', type=str, default='kl_divergence')
        parser.add_argument('--evidential_depth_reg_loss', type=str, 
                            choices = ['nig_kl', 'nig_reg', 'nig_kl_reg', 'none'],
                            default='nig_kl_reg')
        parser.add_argument("--enable_kl_mu_prior", type=str2bool, default=False)
        
        return parser
