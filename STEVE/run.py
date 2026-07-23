import os

import warnings
from lib.metrics import test_metrics

warnings.filterwarnings('ignore')

import yaml
import argparse
import train

from lib.utils import get_project_path


def parse_cli_value(value):
    text = str(value).strip()
    lower = text.lower()
    if lower in {"true", "yes", "y", "on"}:
        return True
    if lower in {"false", "no", "n", "off"}:
        return False
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def parse_unknown_overrides(items):
    overrides = {}
    i = 0
    while i < len(items):
        item = items[i]
        if not item.startswith("--"):
            i += 1
            continue
        key = item[2:].replace("-", "_")
        if "=" in key:
            key, value = key.split("=", 1)
        elif i + 1 < len(items) and not items[i + 1].startswith("--"):
            value = items[i + 1]
            i += 1
        else:
            value = "true"
        overrides[key] = parse_cli_value(value)
        i += 1
    return overrides


def text2args(text,args):
    temp=text.split(",")
    args=argparse.Namespace()
    for s in temp:
        key,value=s.split("=")
        if '\'' in value:
            args[key] = value
        else:
            args[key] = int(value)
    return args


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='train',
                        type=str, help='the configuration to use')
    parser.add_argument('--config_filename', default='configs/NYCBike1.yaml',
                        type=str, help='the configuration to use')
    parser.add_argument('--lr', default=None,
                    type=float, help='init learning rate')
    parser.add_argument('--bs', default=None,
                    type=int, help='batch size')
    parser.add_argument('--d', default=None,
                        type=int, help='the dimition of encoder')
    parser.add_argument('--seed', default=None,
                    type=int, help='random seed')
    parser.add_argument('--lr_mode', default=None,
                    type=str, help='random seed')
    parser.add_argument('--max_epoch', default=None,
                    type=int, help='random seed')
    
    parser.add_argument('--ablation', default='all',
                    type=str, help='ablation study')
    
    args, unknown = parser.parse_known_args()

    args.config_filename = os.path.join(get_project_path(),args.config_filename)

    print(f'Starting experiment with configurations in {args.config_filename}...')
    configs = yaml.load(
        open(args.config_filename),
        Loader=yaml.FullLoader
    )
    if args.lr is not None:
        configs['lr_init']=args.lr

    if args.bs is not None:
        configs['batch_size']=args.bs
    
    if args.seed is not None:
        configs['seed']=args.seed

    if args.d is not None:
        configs['d_model']=args.d
    
    if args.lr_mode is not None:
        configs['lr_mode']=args.lr_mode
    
    if args.max_epoch is not None:
        configs['epochs']=args.max_epoch
    
    configs['ablation']=args.ablation
    overrides = parse_unknown_overrides(unknown)
    configs.update(overrides)
    has_fpem_config = any(str(key).startswith('fpem_') for key in configs.keys())
    configs.setdefault('model_impl', 'fpem' if has_fpem_config else 'steve_original')
    configs.setdefault('steve_prediction_mode', 'full')

    
    

    args = argparse.Namespace(**configs)

    args.graph_file = os.path.join(get_project_path(), args.graph_file)
    args.data_dir = os.path.join(get_project_path(), args.data_dir)

    if args.mode=="train":
        train.main(args)
    elif args.mode=="gat":
        # cross domain call
        pass
    elif args.mode=="test":
        # best_paths=[]
        # for best_path in best_paths:
        #     config_file_path=os.join(best_path,'run.log')
        #     config_file=open(config_file_path)
        #     config=config_file.readlines()
        #     config=config[55:-2]
        pass
            
