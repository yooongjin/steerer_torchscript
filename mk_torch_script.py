import os
import pprint

from lib.utils.utils import create_logger, random_seed_setting
import torch.nn as nn
import torch
import argparse
from lib.models.build_counter import Baseline_Counter
from mmcv import Config, DictAction
from torch.nn import functional as F

def parse_args():
    parser = argparse.ArgumentParser(description='Test crowd counting network')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default="configs/QNRF_final.py",
                        type=str)
    parser.add_argument('--checkpoint',
                    help='experiment configure file name',
                    default="pretrained/QNRF_mae_77.8_mse_138.0.pth",
                    type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--launcher',
                        choices=['none', 'pytorch', 'slurm', 'mpi','torchrun'],
                        default='none',
                        help='job launcher')
        
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
             'in xxx=yyy format will be merged into config file. If the value to '
             'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
             'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
             'Note that the quotation marks are necessary and that no white space '
             'is allowed.')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args

class ScriptModule(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = torch.jit.trace(base_model, torch.zeros(1, 3, 224, 224).to(base_model.device))
        self.mean = nn.Parameter(torch.tensor([0.485, 0.456, 0.406]), requires_grad=False)
        self.std = nn.Parameter(torch.tensor([0.229, 0.224, 0.225]), requires_grad=False)
        

    def transfom(self, x):
        x = x.permute(0, 3, 1, 2)
        device = x.device
        mean = self.mean
        std = self.std
        divisor = 32
        
        h_res = x.shape[2] % divisor
        w_res = x.shape[3] % divisor
        
        hpad = divisor - h_res if h_res > 0 else 0
        wpad = divisor - w_res if w_res > 0 else 0

        pad_input = F.pad(x, pad=(0, wpad, 0, hpad))
        
        norm_input = pad_input / 255.0
        norm_input -= mean.view(1, 3, 1, 1)
        norm_input /= std.view(1, 3, 1, 1)
        
        return norm_input, torch.tensor(hpad, device=device), torch.tensor(wpad, device=device)
    
    def post_process(self, out, hpad, wpad):
        if hpad != 0 :
            out = out[:, :, :-hpad, :]
        if wpad != 0:
            out = out[:, :, :, :-wpad]
        return out
        
    def forward(self, x):
        # x : [b, h, w, c]
        x, hpad, wpad = self.transfom(x)
        result = self.base_model(x)
        ori_den = self.post_process(result, hpad, wpad)

        return ori_den


def main():
    args = parse_args()
    config = Config.fromfile(args.cfg)

    logger, final_output_dir = create_logger(
        config, args.cfg, 'test')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))
   
    random_seed_setting(config)

    # build model
    device = torch.device('cuda:{}'.format(args.local_rank))
    model = Baseline_Counter(config.network, config.dataset.den_factor, config.train.route_size,device)
    model.eval()
    

    if args.checkpoint:
        model_state_file = args.checkpoint
    elif config.test.model_file:
        model_state_file = config.test.model_file
    else:
        model_state_file = "/home/cho092871/Desktop/Networks/STEERER/exp/ASSEMBLE/MocHRBackbone_hrnet48/ASSEMBLE_final_2024-09-20-17-09/Ep_441_mae_59.899008520254604_mse_227.68671281631933.pth"
    
    logger.info('=> loading model from {}'.format(model_state_file))

    pretrained_dict = torch.load(model_state_file)
    model.load_state_dict(pretrained_dict)
    model = model.cuda()
    
    script_model = ScriptModule(model)
    script_model = script_model.cuda().eval()
    script_model = torch.jit.script(script_model)
    
    script_model.save('torchscript_steerer.pt')
    
   
    
if __name__ == '__main__':
    main()
