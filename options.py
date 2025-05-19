import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int, default=-1)
parser.add_argument('--epoch',       type=int,   default=81,   help='epoch number')
parser.add_argument('--lr',          type=float, default=1e-5,  help='learning rate')
parser.add_argument('--batchsize',   type=int,   default=6,    help='training batch size')
parser.add_argument('--trainsize',   type=int,   default=384,   help='training dataset size')
parser.add_argument('--clip',        type=float, default=0.5,   help='gradient clipping margin')
parser.add_argument('--lw',          type=float, default=0.001, help='weight')
parser.add_argument('--decay_rate',  type=float, default=0.1,   help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int,   default=40,    help='every n epochs decay learning rate')

#pretrained backbone parameters path
parser.add_argument('--load',        type=str,   default='swin_base_patch4_window12_384_22k.pth',  help='train from checkpoints')
parser.add_argument('--gpu_id',      type=str,   default='1',   help='train use gpu')

#dataset path of different task.
parser.add_argument('--rgb_label_root',      type=str, default='',           help='the training rgb images root')
parser.add_argument('--depth_label_root',    type=str, default='',         help='the training depth images root')
parser.add_argument('--gt_label_root',       type=str, default='',            help='the training gt images root')

#Evaluate dataset path during training
parser.add_argument('--val_rgb_root',        type=str, default='',      help='the test rgb images root')
parser.add_argument('--val_depth_root',      type=str, default='',    help='the test depth images root')
parser.add_argument('--val_gt_root',         type=str, default='',       help='the test gt images root')

parser.add_argument('--save_path',           type=str, default='',    help='the path to save models and logs')


#args for IHN
parser.add_argument('--IHNmodel', default='IHN.pth',help="restore checkpoint")
parser.add_argument('--iters_lev0', type=int, default=6)
parser.add_argument('--iters_lev1', type=int, default=0)
parser.add_argument('--mixed_precision', default=False, action='store_true',
                    help='use mixed precision')
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--gpuid', type=int, nargs='+', default=[0])
parser.add_argument('--savemat', type=str,  default='ggmap')
parser.add_argument('--savedict', type=str, default='resnpy')
parser.add_argument('--dataset', type=str, default='mscoco', help='dataset')
parser.add_argument('--lev0', default=True, action='store_true',
                    help='warp no')
parser.add_argument('--lev1', default=False, action='store_true',
                    help='warp once')
parser.add_argument('--weight', default=False, action='store_true',
                    help='weight')
parser.add_argument('--model_name_lev0', default='', help='specify model0 name')
parser.add_argument('--model_name_lev1', default='', help='specify model0 name')


opt = parser.parse_args()