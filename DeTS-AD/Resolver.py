import os
import time
import math
import logging
import builtins
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from utils.utils import *
from model.loss_functions import sce_loss
from model.lr import PolynomialDecayLR
from model.Transformer import TfVar
from model.loss_functions import *
from data_process.data_loader import get_loader_segment
from sklearn.metrics import (precision_score,
                             recall_score,
                             f1_score,
                             auc,
                             roc_auc_score,
                             average_precision_score,
                             precision_recall_curve,
                             )

from metrics.metrics import *
from metrics import point_adjustment
from metrics import ts_metrics_enhanced

os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def adjust_learning_rate(optimizer, epoch, initial_lr, step_size=2, decay_factor=0.9):
    lr_adjust = {epoch: initial_lr * (decay_factor ** ((epoch - 1) // step_size))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print(f'Updating learning rate to {lr}')


class OneEarlyStopping:
    def __init__(self, patience=10, verbose=False, dataset_name='', delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.dataset = dataset_name

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
       

        torch.save(model.state_dict(), os.path.join(path, str(self.dataset) + f'_checkpoint.pth'))
        self.val_loss_min = val_loss


class ReSolver(object):
    DEFAULTS = {}

    def __init__(self, config):

        self.scheduler = None
        self.model = None
        self.optimizer = None
        self.__dict__.update(ReSolver.DEFAULTS, **config)

        self.train_time_per_epoch = 0.0
        self.test_time_per_epoch = 0.0
        self.train_loader, self.vali_loader = get_loader_segment(self.data_path,
                                                                 batch_size=self.batch_size,
                                                                 win_size=self.win_size,
                                                                 mode='train',
                                                                 dataset=self.dataset)

        self.test_loader = get_loader_segment(self.data_path,
                                              batch_size=self.batch_size,
                                              win_size=self.win_size,
                                              mode='test',
                                              dataset=self.dataset)

        self.entropy_loss = EntropyLoss()
        self.criterion = nn.MSELoss(reduction='none')

        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s - %(message)s')
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)

   
        if not any(isinstance(handler, logging.StreamHandler) for handler in self.logger.handlers):
            self.logger.addHandler(stream_handler)
   
            self._redirect_print_to_logger()

    def _redirect_print_to_logger(self):
        def print_to_logger(*args, **kwargs):
            message = " ".join(map(str, args))
            self.logger.info(message)
       
            builtins.print = print_to_logger

    def model_init(self, config):
        self.model = TfVar(config)
  
        if hasattr(self.model, 'module'):
            self.model.module.anomaly_score_method = getattr(self, 'anomaly_score_method', 'product')
            self.model.module.anomaly_score_alpha = getattr(self, 'anomaly_score_alpha', 0.5)
        else:
            self.model.anomaly_score_method = getattr(self, 'anomaly_score_method', 'product')
            self.model.anomaly_score_alpha = getattr(self, 'anomaly_score_alpha', 0.5)
            


        self.optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()),
                                           lr=self.peak_lr, weight_decay=self.weight_decay)
        self.scheduler = PolynomialDecayLR(self.optimizer,
                                           warmup_updates=self.warmup_epoch * self.batch_size,
                                           tot_updates=self.num_epochs * self.batch_size,
                                           lr=self.peak_lr,
                                           end_lr=self.end_lr,
                                           power=1.0)

        if torch.cuda.is_available():
            self.model = torch.nn.DataParallel(self.model, device_ids=[0], output_device=0).to(self.device)
          
            self.model.module.anomaly_score_method = getattr(self, 'anomaly_score_method', 'product')
            self.model.module.anomaly_score_alpha = getattr(self, 'anomaly_score_alpha', 0.5)

    def vali(self, vali_loader):
        self.model.eval()

        valid_loss_list = []
        valid_re_loss_list = []
        valid_intra_loss_list = []

        for i, (input_data, _) in enumerate(vali_loader):
            input_data = input_data.float().to(self.device)
            output_dict = self.model(input_data)

            output = output_dict['out']
            td_scores = output_dict['td_scores']  
            rd_scores = output_dict['rd_scores']  
            anomaly_scores = output_dict['anomaly_scores']  

            rec_loss = td_scores.mean()  
            attn_loss = rd_scores.mean() * self.alpha  

            loss = rec_loss + attn_loss

            valid_re_loss_list.append(rec_loss.detach().cpu().numpy())
            valid_intra_loss_list.append(attn_loss.detach().cpu().numpy())
            valid_loss_list.append(loss.detach().cpu().numpy())

        return np.average(valid_loss_list), np.average(valid_re_loss_list), np.average(valid_intra_loss_list)

    def train(self):

   
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)
        early_stopping = OneEarlyStopping(patience=self.patience, verbose=True, dataset_name=self.dataset)
        train_steps = len(self.train_loader)

        training_start_time = time_now = time.time()

        for epoch in tqdm(range(self.num_epochs)):
            iter_count = 0
            loss_list = []
            rec_loss_list = []
            intra_loss_list = []

       
            epoch_time = time.time()

            self.model.train()
            for i, (input_data, labels) in enumerate(self.train_loader):

                self.optimizer.zero_grad()
                iter_count += 1
                input_data = input_data.float().to(self.device)
                output_dict = self.model(input_data)

                output = output_dict['out']
                td_scores = output_dict['td_scores']  
                rd_scores = output_dict['rd_scores']  
                anomaly_scores = output_dict['anomaly_scores']  

                rec_loss = td_scores.mean()  
                attn_loss = rd_scores.mean() * self.alpha  

                loss = rec_loss + attn_loss

                loss_list.append(loss.detach().cpu().numpy())
                rec_loss_list.append(rec_loss.detach().cpu().numpy())
                intra_loss_list.append(attn_loss.detach().cpu().numpy())

                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.num_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                loss.backward()
                self.optimizer.step()


            train_loss = np.average(loss_list)
            train_rec_loss = np.average(rec_loss_list)
            train_intra_loss = np.average(intra_loss_list)
            valid_loss, valid_re_loss, valid_intra_loss = self.vali(self.vali_loader)

            print(
                f"Epoch: {epoch + 1}")
            early_stopping(valid_loss, self.model, self.model_save_path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        self.train_time_per_epoch = round((time.time() - training_start_time) / (epoch + 1), 3)

        return

    def test(self, config_setting):
        """
        Test the model with the test dataset
        """
        skip_loading = config_setting.get('skip_loading', False)
        
        if not skip_loading:
            model_name = os.path.join(config_setting['model_save_path'], f"{config_setting['dataset']}_checkpoint.pth")
            if os.path.exists(model_name):
                self.model.load_state_dict(torch.load(model_name))
                print(f"Successfully loaded model from {model_name}")
            else:
                print(f"Warning: No saved model found at {model_name}. Using initialized model.")
        else:
            print("Skipping loading weights as requested.")
            
        self.model.eval()



        criterion = nn.MSELoss(reduction='none')
        


        if self.threshold_setting == 'preset':
            train_scores = []
            for i, (input_data, labels) in enumerate(self.train_loader):
                input_data = input_data.float().to(self.device)
                output_dict = self.model(input_data, mode='test')
                anomaly_scores = output_dict['anomaly_scores'] 
                train_scores.append(anomaly_scores.detach().cpu().numpy())

            train_scores = np.concatenate(train_scores, axis=0).reshape(-1)

            valid_scores = []
            for i, (input_data, labels) in enumerate(self.vali_loader):
                input_data = input_data.float().to(self.device)
                output_dict = self.model(input_data, mode='test')
                anomaly_scores = output_dict['anomaly_scores']  
                valid_scores.append(anomaly_scores.detach().cpu().numpy())

            valid_scores = np.concatenate(valid_scores, axis=0).reshape(-1)
            combined_scores = np.concatenate([train_scores, valid_scores], axis=0)

            thresh = np.percentile(combined_scores, 100 - self.anomaly_ratio)
          

        test_window_labels = []
        test_window_energy = []

        test_labels = []
        test_scores = []

        start_time = time.time()

        for i, (input_data, labels) in enumerate(self.test_loader):
            input_data = input_data.float().to(self.device)
            output_dict = self.model(input_data, mode='test')
            
            anomaly_scores = output_dict['anomaly_scores']  
            scores = anomaly_scores.detach().cpu().numpy()
            
            test_scores.append(scores)
            test_labels.append(labels)

            test_window_energy.extend(scores.mean(axis=-1))
            test_window_labels.extend((labels.sum(axis=-1) > 1).numpy().astype(int))

        self.test_time_per_epoch = round(time.time() - start_time, 3)


        test_scores = np.concatenate(test_scores, axis=0)
        test_labels = np.concatenate(test_labels, axis=0)

        test_energy = test_scores.reshape(-1)
        test_labels = test_labels.reshape(-1)
        anomaly_state = False
        anomaly_count = 0

        if self.threshold_setting == 'optimal':
            anomaly_ratio = self.anomaly_ratio
            thresh = np.percentile(test_energy, 100 - anomaly_ratio)
         
            pred = (test_energy > thresh).astype(int)
            results = ts_metrics_enhanced(test_labels, point_adjustment(test_labels, test_energy), pred)
        else:
            results = {k: 0.0 for k in metric_list}

            results['thresh'] = 0.0

            pred = (test_energy > thresh).astype(int)

            gt = test_labels.astype(int)

            print(f"pred: {pred.shape}, gt: {gt.shape}")

            events = get_events(gt)

            _, _, _, _, _, _, threshold_setting_results = get_point_adjust_scores(gt, pred, test_energy, events)

            results = ts_metrics_enhanced(test_labels, point_adjustment(test_labels, test_energy), pred)

            results['pc_adjust'] = threshold_setting_results['pc_adjust']
            results['rc_adjust'] = threshold_setting_results['rc_adjust']
            results['f1_adjust'] = threshold_setting_results['f1_adjust']

        precision_adjust, recall_adjust, f_score_adjust = results['pc_adjust'], results['rc_adjust'], results['f1_adjust']
        results['thresh'] = thresh
        results['trt'] = self.train_time_per_epoch
        results['tst'] = self.test_time_per_epoch

        print('=' * 63)
        print(f"Dataset: {self.dataset} |adjusted_ Precision: {precision_adjust:.4f} | adjusted_Recall: {recall_adjust:.4f} | adjusted_F1: {f_score_adjust:.4f} ")

        return results


def get_point_adjust_scores(y_test, pred_labels, pred_scores, true_events):
    results = {
        "pc": 0.0,
        "rc": 0.0,
        "f1": 0.0,
        "acc_adjust": 0.0,
        "pc_adjust": 0.0,
        "rc_adjust": 0.0,
        "f1_adjust": 0.0,
        "mcc_adjust": 0.0,
        "prc": 0.0,
        "roc": 0.0,
        "apc": 0.0,
    }
    tp = 0
    fn = 0
    for true_event in true_events.keys():
        true_start, true_end = true_events[true_event]
        if pred_labels[true_start:true_end].sum() > 0:
            tp += (true_end - true_start)
        else:
            fn += (true_end - true_start)
    fp = np.sum(pred_labels) - np.sum(pred_labels * y_test)

    pc, rc, fscore = get_prec_rec_fscore(tp, fp, fn)

    tn = len(pred_labels) - (tp + fp + fn)

    avg_precision = average_precision_score(y_test, pred_scores)
    auc_roc = roc_auc_score(y_test, pred_scores)
    precision, recall, _ = precision_recall_curve(y_test, pred_scores)

    results['pc'] = round(precision_score(y_test, pred_labels, average='binary'), 4)
    results['rc'] = round(recall_score(y_test, pred_labels, average='binary'), 4)
    results['f1'] = round(f1_score(y_test, pred_labels, average='binary'), 4)

    results['f1_adjust'] = round(fscore, 4)
    results['pc_adjust'] = round(pc, 4)
    results['rc_adjust'] = round(rc, 4)

    results['acc_adjust'] = round((tp + tn) / len(y_test), 4)

    results["prc"] = round(auc(recall, precision), 4)
    results["roc"] = round(auc_roc, 4)
    results["apc"] = round(avg_precision, 4)

    return fp, fn, tp, pc, rc, fscore, results


def matthews_correlation_coefficient(TP, TN, FP, FN):
    numerator = TP * TN - FP * FN
    denominator = np.sqrt(TP + FP) * np.sqrt(TP + FN) * np.sqrt(TN + FP) * np.sqrt((TN + FN))

    if denominator < np.finfo(float).eps:
        return 0.0

    mcc = numerator / denominator
    return mcc


def get_f_score(pc, rc):
    if pc == 0 and rc == 0:
        f_score = 0
    else:
        f_score = 2 * (pc * rc) / (pc + rc)
    return f_score


def get_prec_rec_fscore(tp, fp, fn):
    if tp == 0:
        precision = 0
        recall = 0
    else:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
    fscore = get_f_score(precision, recall)
    return precision, recall, fscore


def get_events(y_test, outlier=1, normal=0, breaks=[]):
    events = dict()
    label_prev = normal
    event = 0 
    event_start = 0
    for tim, label in enumerate(y_test):
        if label == outlier:
            if label_prev == normal:
                event += 1
                event_start = tim
            elif tim in breaks:
    
                event_end = tim - 1
                events[event] = (event_start, event_end)
                event += 1
                event_start = tim
        else:
            if label_prev == outlier:
                event_end = tim - 1
                events[event] = (event_start, event_end)
        label_prev = label

    if label_prev == outlier:
     
        event_end = tim
        events[event] = (event_start, event_end)
    return events
