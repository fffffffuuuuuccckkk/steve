# coding=gbk
import os
import random
import torch
import numpy as np
from datetime import datetime

from lib.metrics import mae_torch

def masked_mae_loss(mask_value):
    def loss(preds, labels):
        mae = mae_torch(pred=preds, true=labels, mask_value=mask_value)
        return mae
    return loss

def init_seed(seed):
    '''
    Disable cudnn to maximize reproducibility
    '''
    torch.cuda.cudnn_enabled = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def disp(x, name):
    print(f'{name} shape: {x.shape}')

def get_model_params(model_list):
    model_parameters = []
    for m in model_list:
        if m != None:
            model_parameters += list(m.parameters())
    return model_parameters

def str2bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ('1', 'true', 'yes', 'y', 'on')


def safe_log_name(name):
    return ''.join(c if (c.isalnum() or c in ('-', '_', '.')) else '_' for c in str(name).strip())


def _has_checkpoint(log_dir):
    if not os.path.isdir(log_dir):
        return False
    names = os.listdir(log_dir)
    if 'last_model.pth' in names or 'best_model.pth' in names:
        return True
    return any(name.startswith('epoch') and name.endswith('.pth') for name in names)


def _checkpoint_epoch_from_name(name):
    if not (name.startswith('epoch') and name.endswith('.pth')):
        return -1
    try:
        return int(name[len('epoch'):-len('.pth')])
    except ValueError:
        return -1


def _checkpoint_progress(log_dir):
    progress = 0
    last_epoch_path = os.path.join(log_dir, 'last_epoch.txt')
    if os.path.isfile(last_epoch_path):
        try:
            with open(last_epoch_path, 'r') as f:
                progress = max(progress, int(f.read().strip()))
        except Exception:
            pass

    for name in os.listdir(log_dir):
        progress = max(progress, _checkpoint_epoch_from_name(name))
    return progress


def find_latest_exp_dir(args):
    exp_name = getattr(args, 'exp_name', None)
    if exp_name is None:
        exp_name = getattr(args, 'run_name', None)
    if exp_name is None or str(exp_name).strip() == '':
        return None

    current_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    exp_root = os.path.join(current_dir, 'experiments', args.dataset)
    if not os.path.isdir(exp_root):
        return None

    prefix = safe_log_name(exp_name) + '_'
    candidates = []
    for name in os.listdir(exp_root):
        path = os.path.join(exp_root, name)
        if name.startswith(prefix) and _has_checkpoint(path):
            candidates.append((_checkpoint_progress(path), os.path.getmtime(path), path))
    if not candidates:
        return None
    candidates.sort()
    return candidates[-1][2]


def get_log_dir(args):
    explicit_log_dir = getattr(args, 'log_dir', None)
    if explicit_log_dir is not None and str(explicit_log_dir).strip() != '':
        return os.path.abspath(str(explicit_log_dir))

    resume_dir = getattr(args, 'resume_dir', None)
    if str2bool(getattr(args, 'resume', False)):
        if resume_dir is not None and str(resume_dir).strip() != '':
            return os.path.abspath(str(resume_dir))
        latest_dir = find_latest_exp_dir(args)
        if latest_dir is not None:
            return latest_dir

    current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
    current_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    exp_name = getattr(args, 'exp_name', None)
    if exp_name is None:
        exp_name = getattr(args, 'run_name', None)
    if exp_name is not None and str(exp_name).strip() != '':
        safe_name = safe_log_name(exp_name)
        log_leaf = '{}_{}'.format(safe_name, current_time)
    else:
        log_leaf = current_time
    log_dir = os.path.join(current_dir, 'experiments', args.dataset, log_leaf)
    return log_dir 

def load_graph(adj_file, device='cpu'):
    '''Loading graph in form of edge index.'''
    graph = np.load(adj_file)['adj_mx']
    graph = torch.tensor(graph, device=device, dtype=torch.float)
    return graph

def dwa(L_old, L_new, T=2):
    '''
    L_old: list.
    '''
    L_old = torch.tensor(L_old, dtype=torch.float32)
    L_new = torch.tensor(L_new, dtype=torch.float32)
    N = len(L_new) # task number
    r =  L_old / L_new
    
    w = N * torch.softmax(r / T, dim=0)
    return w.numpy()

def get_project_path():
    project_path = os.path.join(
        os.path.dirname(__file__),
        "..",
    )
    project_path = project_path[:find_last(project_path,'STEVE')+5]
    return project_path

def find_last(search, target,start=0):
    loc = search.find(target,start)
    end_loc=loc
    while loc != -1:
        end_loc=loc
        start = loc+1
        loc = search.find(target,start)
    return end_loc
