import os
import glob
import traceback
from datetime import datetime
import json
import sys
sys.path.append('/home/zhangwt/StableST')

import warnings

from lib.metrics import test_metrics

warnings.filterwarnings('ignore')

import yaml
import argparse
import time
import torch

from lib.utils import (
    init_seed,
    get_model_params,
    load_graph, get_log_dir,
    str2bool,
)

from lib.dataloader import get_dataloader
from lib.logger import get_logger, PD_Stats
from lib.utils import dwa
import numpy as np
from models.our_model import StableST


def _checkpoint_epoch(path):
    name = os.path.basename(path)
    if name.startswith('epoch') and name.endswith('.pth'):
        try:
            return int(name[len('epoch'):-len('.pth')])
        except ValueError:
            return -1
    return -1


def _find_resume_checkpoint(log_dir):
    last_path = os.path.join(log_dir, 'last_model.pth')
    if os.path.isfile(last_path):
        return last_path

    epoch_paths = glob.glob(os.path.join(log_dir, 'epoch*.pth'))
    epoch_paths = [path for path in epoch_paths if _checkpoint_epoch(path) >= 0]
    if epoch_paths:
        epoch_paths.sort(key=lambda path: (_checkpoint_epoch(path), os.path.getmtime(path)))
        return epoch_paths[-1]

    best_path = os.path.join(log_dir, 'best_model.pth')
    if os.path.isfile(best_path):
        return best_path
    return None


class Trainer(object):
    def __init__(self, model, optimizer, dataloader, graph, lr_scheduler,args, graph2=None,load_state=None):
        super(Trainer, self).__init__()
        self.model = model
        self.optimizer = optimizer
        self.train_loader = dataloader['train']
        self.val_loader = dataloader['val']
        self.test_loader = dataloader['test']
        self.scaler = dataloader['scaler']
        self.graph = graph
        self.lr_scheduler=lr_scheduler
        self.args = args
        if graph2 != None:
            self.test_graph=graph2
        else:
            self.test_graph=graph

        self.train_per_epoch = len(self.train_loader)
        if self.val_loader != None:
            self.val_per_epoch = len(self.val_loader)

        # log
        args.log_dir = get_log_dir(args)
        if os.path.isdir(args.log_dir) == False and not args.debug:
            os.makedirs(args.log_dir, exist_ok=True)
        self.logger = get_logger(args.log_dir, name=args.log_dir, debug=args.debug)
        self.best_path = os.path.join(self.args.log_dir, 'best_model.pth')
        self.start_epoch = 1
        self.best_loss = float('inf')
        self.best_epoch = 0
        self.not_improved_count = 0
        self.resume_state = None
        self.resume_checkpoint_path = None

        # create a panda object to log loss and acc
        self.training_stats = PD_Stats(
            os.path.join(args.log_dir, 'stats.pkl'),
            ['epoch', 'train_loss', 'val_loss'],
        )
        self.logger.info('Experiment log path in: {}'.format(args.log_dir))
        self.logger.info('Experiment configs are: {}'.format(args))
        if str2bool(getattr(self.args, 'resume', False)):
            self._try_resume()

    def _try_resume(self):
        checkpoint_path = _find_resume_checkpoint(self.args.log_dir)
        if checkpoint_path is None:
            self.logger.info('Resume requested, but no checkpoint found in {}'.format(self.args.log_dir))
            return

        state_dict = torch.load(checkpoint_path, map_location=torch.device(self.args.device))
        self.model.load_state_dict(state_dict['model'])
        if 'optimizer' in state_dict and state_dict['optimizer'] is not None:
            self.optimizer.load_state_dict(state_dict['optimizer'])
        if 'lr_scheduler' in state_dict and state_dict['lr_scheduler'] is not None:
            try:
                self.lr_scheduler.load_state_dict(state_dict['lr_scheduler'])
            except Exception as exc:
                self.logger.info('Resume skipped lr_scheduler state: {}'.format(exc))

        checkpoint_epoch = int(state_dict.get('epoch', _checkpoint_epoch(checkpoint_path)))
        if checkpoint_epoch > 0:
            self.start_epoch = checkpoint_epoch + 1
        self.best_loss = float(state_dict.get('best_loss', self.best_loss))
        self.best_epoch = int(state_dict.get('best_epoch', self.best_epoch))
        self.not_improved_count = int(state_dict.get('not_improved_count', self.not_improved_count))
        self.resume_state = state_dict
        self.resume_checkpoint_path = checkpoint_path
        self.logger.info(
            'Resumed training from {} at epoch {}; next epoch {}'.format(
                checkpoint_path, checkpoint_epoch, self.start_epoch
            )
        )

    def train_epoch(self, epoch, loss_weights):
        self.model.train()
        p=epoch/self.args.epochs*1.0
        total_loss = 0
        total_sep_loss = np.zeros(3)
        lms=[]
        fpem_log_sums = {}
        fpem_log_count = 0
        t1=datetime.now()
        for batch_idx, (data, target, time_label,c) in enumerate(self.train_loader):
            

            self.optimizer.zero_grad()
            # input shape: n,l,v,c; graph shape: v,v;
            
            Z, H = self.model(data)  # nvc
            loss, sep_loss,lm = self.model.calculate_loss(Z, H, target, c, time_label, self.scaler, loss_weights,p,True)
            if type(lm) == int:
                lms.append(lm)
            else:
                lms.append(lm.item())
            # t2=datetime.now()
            assert not torch.isnan(loss)

            gc_handles = []
            try:
                primary_loss = None
                latest_outputs = getattr(self.model, "latest_fpem_outputs", {})
                if isinstance(latest_outputs, dict):
                    primary_loss = latest_outputs.get("primary_loss", None)
                if hasattr(self.model, "prepare_fpem_gc_pred_loss_only"):
                    self.model.prepare_fpem_gc_pred_loss_only(primary_loss)
                if hasattr(self.model, "register_fpem_grad_consensus_hooks"):
                    gc_handles = self.model.register_fpem_grad_consensus_hooks(epoch)
                loss.backward()
            finally:
                for handle in gc_handles:
                    handle.remove()
                if hasattr(self.model, "clear_fpem_gc_pred_loss_only"):
                    self.model.clear_fpem_gc_pred_loss_only()
            if getattr(self.model, "latest_fpem_logs", None):
                fpem_log_count += 1
                for key, value in self.model.latest_fpem_logs.items():
                    fpem_log_sums[key] = fpem_log_sums.get(key, 0.0) + float(value)
            # t3=datetime.now()
            # gradient clipping
            # import pdb
            # pdb.set_trace()
            # for param in self.model.parameters():
            #     print("param=%s, grad=%s" % (param.data, param.grad))
            
            # import pdb
            # pdb.set_trace()
            if self.args.grad_norm:
                torch.nn.utils.clip_grad_norm_(
                    get_model_params([self.model]),
                    self.args.max_grad_norm)
            # t4=datetime.now()
            self.optimizer.step()
            total_loss += loss.item()
            total_sep_loss += np.asarray(sep_loss, dtype=np.float64)
        t5=datetime.now()
        print(f"train_time:{t5-t1}")


        train_epoch_loss = total_loss / self.train_per_epoch
        total_sep_loss = total_sep_loss / self.train_per_epoch
        total_sep_loss = np.nan_to_num(total_sep_loss, nan=1.0, posinf=1.0, neginf=1.0)
        self.logger.info('*******Train Epoch {}: averaged Loss : {:.6f}'.format(epoch, train_epoch_loss))
        self.logger.info('*******Train Epoch {}: averaged lm : {:.6f}'.format(epoch, np.mean(lms)))
        if fpem_log_count > 0:
            fpem_logs = {key: value / fpem_log_count for key, value in fpem_log_sums.items()}
            self.logger.info('*******Train Epoch {}: fpem logs {}'.format(epoch, json.dumps(fpem_logs)))
        return train_epoch_loss, total_sep_loss

    def val_epoch(self, epoch, val_dataloader, loss_weights):
        self.model.eval()

        total_val_loss = 0
        total_sep_loss = np.zeros(3)
        with torch.no_grad():
            for batch_idx, (data, target,time_label,c) in enumerate(val_dataloader):
                Z, H = self.model(data)
                # c_hat=self.model.predict_con(data)
                loss, sep_loss,lm = self.model.calculate_loss(Z, H, target, c, time_label, self.scaler, loss_weights)
                # loss = self.model.pred_loss(repr1, repr1, target, self.scaler)
                if not torch.isnan(loss):
                    total_val_loss += loss.item()
                total_sep_loss += sep_loss
        val_loss = total_val_loss / len(val_dataloader)
        total_sep_loss = total_sep_loss /len(val_dataloader)
        self.logger.info('*******Val Epoch {}: averaged Loss : {:.6f} sep loss : {}'.format(epoch, val_loss, total_sep_loss))
        return val_loss

    def train(self):
        best_loss = self.best_loss
        best_epoch = self.best_epoch
        not_improved_count = self.not_improved_count
        last_save_dict = self.resume_state
        start_time = time.time()

        loss_tm1 = loss_t = np.ones(3)  # (1.0, 1.0, 1.0)
        if self.start_epoch > self.args.epochs:
            self.logger.info(
                'Resume checkpoint already reached target epochs: start_epoch={}, target={}'.format(
                    self.start_epoch, self.args.epochs
                )
            )

        for epoch in range(self.start_epoch, self.args.epochs + 1):
            # dwa mechanism to balance optimization speed for different tasks

            if self.args.use_dwa:
                loss_tm2 = loss_tm1
                loss_tm1 = loss_t
                if (epoch == 1) or (epoch == 2):
                    loss_weights = dwa(loss_tm1, loss_tm1, self.args.temp)
                else:
                    loss_weights = dwa(loss_tm1, loss_tm2, self.args.temp)
            else:
                loss_weights = np.ones(3)
            self.logger.info('loss weights: {}'.format(loss_weights))
            train_epoch_loss, loss_t = self.train_epoch(epoch, loss_weights)
            loss_t = np.nan_to_num(loss_t, nan=1.0, posinf=1.0, neginf=1.0)
            # train_epoch_loss = self.train_epoch(epoch, loss_weights)

            if train_epoch_loss > 1e6:
                self.logger.warning('Gradient explosion detected. Ending...')
                break

            val_dataloader = self.val_loader if self.val_loader != None else self.test_loader
            val_epoch_loss = self.val_epoch(epoch, val_dataloader, loss_weights)

            if epoch in [1,16,32,64,128]:
                save_dict = {
                    "epoch": epoch,
                    "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "lr_scheduler": self.lr_scheduler.state_dict(),
                    "best_loss": best_loss,
                    "best_epoch": best_epoch,
                    "not_improved_count": not_improved_count,
                }
                save_dir=os.path.join(self.args.log_dir,'epoch{}.pth'.format(epoch))
                self.logger.info('**************Current {} model saved to {}'.format(epoch,save_dir))
                torch.save(save_dict, save_dir)
                last_save_dict = save_dict

            if val_epoch_loss < best_loss:
                best_loss = val_epoch_loss
                best_epoch = epoch
                not_improved_count = 0
                # save the best state
                save_dict = {
                    "epoch": epoch,
                    "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "lr_scheduler": self.lr_scheduler.state_dict(),
                    "best_loss": best_loss,
                    "best_epoch": best_epoch,
                    "not_improved_count": not_improved_count,
                }

                if not self.args.debug:
                    self.logger.info('**************Current best model saved to {}'.format(self.best_path))
                    torch.save(save_dict, self.best_path)
                last_save_dict = save_dict
            else:
                not_improved_count += 1
            
            self.lr_scheduler.step(val_epoch_loss)

            save_dict = {
                "epoch": epoch,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "lr_scheduler": self.lr_scheduler.state_dict(),
                "best_loss": best_loss,
                "best_epoch": best_epoch,
                "not_improved_count": not_improved_count,
            }
            last_save_dict = save_dict
            if not self.args.debug:
                torch.save(save_dict, os.path.join(self.args.log_dir, 'last_model.pth'))
                with open(os.path.join(self.args.log_dir, 'last_epoch.txt'), 'w') as f:
                    f.write(str(epoch))

            # early stopping
            if self.args.early_stop and not_improved_count == self.args.early_stop_patience:
                self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                 "Training stops.".format(self.args.early_stop_patience))
                break


        training_time = time.time() - start_time
        self.logger.info("== Training finished.\n"
                         "Total training time: {:.2f} min\t"
                         "best loss: {:.4f}\t"
                         "best epoch: {}\t".format(
            (training_time / 60),
            best_loss,
            best_epoch))

        # test
        self.logger.info("load best model from {}".format(self.best_path))
        if self.args.debug and last_save_dict is not None:
            state_dict = last_save_dict
        elif os.path.isfile(self.best_path):
            state_dict = torch.load(self.best_path, map_location=torch.device(self.args.device))
        elif last_save_dict is not None:
            state_dict = last_save_dict
        else:
            raise RuntimeError('No checkpoint available for testing in {}'.format(self.args.log_dir))
        self.model.load_state_dict(state_dict['model'])
        self.logger.info("== Test results.")
        test_results = self.test(self.model, self.test_loader, self.scaler,
                                 self.test_graph, self.logger, self.args)
        results = {
            'best_val_loss': best_loss,
            'best_val_epoch': best_epoch,
            'test_results': test_results,
        }

        return results

    @staticmethod
    def test(model, dataloader, scaler, graph, logger, args):
        model.eval()
        y_pred = []
        y_true = []
        x=[]
        atts=[]
        Cs=[]
        Hs=[]
        start_time=time.time()
        with torch.no_grad():
            for batch_idx, (data, target, c) in enumerate(dataloader):
                # weather
                # if batch_idx!=10:
                #     continue
                x.append(data)
                output = model.forward_output(data, exog=c, training=False)
                pred_output = output["prediction"]
                pred_output = pred_output.squeeze(1)
                target = target.squeeze(1)
                y_true.append(target)
                y_pred.append(pred_output)
                atts.append(output["env_route_q"].cpu().detach())
                Cs.append(output["E_useful"].cpu().detach())
                Hs.append(output["Z_inv"].cpu().detach())
        
        y_true = scaler.inverse_transform(torch.cat(y_true, dim=0))
        y_pred = scaler.inverse_transform(torch.cat(y_pred, dim=0))

        x=torch.cat(x,dim=0)
        atts=torch.cat(atts,dim=0)
        Cs=torch.cat(Cs,dim=0)
        Hs=torch.cat(Hs,dim=0)

        end_time=time.time()
        print(start_time)
        print(end_time-start_time)
        logger.info(end_time-start_time)

        save_path=os.path.join(args.log_dir,'result.npz')
        np.savez(save_path,y_true=y_true.cpu().numpy(),y_pred=y_pred.cpu().numpy(),x=x.cpu().numpy(),atts=atts.cpu().numpy())
        rep_path=os.path.join(args.log_dir,'representation.npz')
        np.savez(rep_path,C=Cs.cpu().numpy(),H=Hs.cpu().numpy())

        test_results = []
        # inflow
        # mae, mape = test_metrics(y_pred[..., 0], y_true[..., 0])
        # logger.info("INFLOW, MAE: {:.2f}, MAPE: {:.4f}%".format(mae, mape * 100))
        # test_results.append([mae, mape])
        # outflow
        mae, mape = test_metrics(y_pred, y_true)
        logger.info("FLOW, MAE: {:.2f}, MAPE: {:.4f}%".format(mae, mape * 100))
        test_results.append([mae, mape])

        return np.stack(test_results, axis=0)


def make_one_hot(labels, classes):
    # labels=labels.to('cuda:1')
    labels = labels.unsqueeze(dim=-1)
    one_hot = torch.FloatTensor(labels.size()[0], classes).zero_().to(labels.device)
    target = one_hot.scatter_(1, labels.data, 1)
    return target

def main(args):

    # A,A2 = load_graph(args.graph_file, device=args.device)  # �ڽӾ���
    A = load_graph(args.graph_file, device=args.device)

    init_seed(args.seed)

    dataloader = get_dataloader(
        data_dir=args.data_dir,
        dataset=args.dataset,
        batch_size=args.batch_size,
        test_batch_size=args.test_batch_size,
        device=args.device
    )

    # current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
    # current_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    # log_dir = os.path.join(current_dir, 'experiments', 'NYCBike1', current_time)
    model = StableST(args=args, adj=A, in_channels=args.d_input, embed_size=args.d_model,
                T_dim=args.input_length, output_T_dim=1, output_dim=args.d_output,device=args.device).to(args.device)

    

    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=args.lr_init,
        eps=1.0e-8,
        weight_decay=0,
        amsgrad=False
    )
    lr_scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=args.lr_patience, verbose=True, threshold=0.0001, threshold_mode='rel', min_lr=0.000005, eps=1e-08)

    # start training
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        dataloader=dataloader,
        graph=A,
        graph2=A,
        lr_scheduler=lr_scheduler,
        args=args
    )

    results = None
    try:
        if args.mode == 'train':
            results = trainer.train() # best_eval_loss, best_epoch
        elif args.mode == 'test':
            # test
            state_dict = torch.load(
                args.best_path,
                map_location=torch.device(args.device)
            )
            model.load_state_dict(state_dict['model'])
            print("Load saved model")
            results = trainer.test(model, dataloader['test'], dataloader['scaler'],
                        A, trainer.logger, trainer.args)
        else:
            raise ValueError
    except Exception:
        trainer.logger.info(traceback.format_exc())
        raise

    trainer.logger.info("abulation is {}".format(args.ablation))
    trainer.logger.info("bank gradient!")
    trainer.logger.info("gamma {}".format(args.bank_gamma))
    trainer.logger.info("kw {}".format(args.kw))








