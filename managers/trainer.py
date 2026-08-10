

import statistics
import timeit
import os
import logging
import numpy as np
import time
import csv

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import dgl
from sklearn import metrics


class Trainer():
    def __init__(self, params, graph_classifier, train, valid_evaluator=None):
        self.graph_classifier = graph_classifier
        self.valid_evaluator = valid_evaluator
        self.params = params
        self.train_data = train
        self.temperature = 0.1
        self.updates_counter = 0
        self.validation_timeline_start_time = None
        self.validation_timeline_last_record_time = None
        self.validation_timeline_csv_path = None

        model_params = list(self.graph_classifier.parameters())
        logging.info('Total number of parameters: %d' % sum(map(lambda x: x.numel(), model_params)))

        if params.optimizer == "SGD":
            self.optimizer = optim.SGD(
                model_params,
                lr=params.lr,
                momentum=params.momentum,
                weight_decay=self.params.l2
            )
        if params.optimizer == "Adam":
            self.optimizer = optim.Adam(
                model_params,
                lr=params.lr,
                weight_decay=self.params.l2
            )

        self.criterion = nn.MarginRankingLoss(self.params.margin, reduction='mean')
        self.b_xent = nn.BCEWithLogitsLoss()
        self.reset_training_state()

    def reset_training_state(self):
        self.best_metric = 0
        self.last_metric = 0
        self.not_improved_count = 0

    def setup_validation_timeline_csv(self):
        self.validation_timeline_start_time = time.time()
        self.validation_timeline_last_record_time = self.validation_timeline_start_time

        csv_name = getattr(self.params, 'validation_timeline_csv', 'validation_auc_timeline.csv')
        self.validation_timeline_csv_path = os.path.join(self.params.exp_dir, csv_name)
        os.makedirs(os.path.dirname(self.validation_timeline_csv_path), exist_ok=True)

        with open(self.validation_timeline_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'elapsed_sec',
                'elapsed_min',
                'epoch',
                'updates',
                'auc',
                'auc_pr',
                'best_auc'
            ])
        logging.info('Validation AUC timeline CSV: %s', self.validation_timeline_csv_path)

    def record_validation_timeline(self, epoch, result):
        if self.validation_timeline_csv_path is None:
            return

        now = time.time()
        elapsed_sec = now - self.validation_timeline_start_time
        auc = result.get('auc', '')
        auc_pr = result.get('auc_pr', result.get('auc_pr_score', ''))

        with open(self.validation_timeline_csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                f'{elapsed_sec:.3f}',
                f'{elapsed_sec / 60.0:.3f}',
                epoch,
                self.updates_counter,
                auc,
                auc_pr,
                self.best_metric
            ])

        self.validation_timeline_last_record_time = now
        logging.info(
            'Recorded validation timeline: epoch=%s updates=%s elapsed_min=%.3f auc=%s auc_pr=%s',
            epoch, self.updates_counter, elapsed_sec / 60.0, str(auc), str(auc_pr)
        )

    def maybe_record_timed_validation(self, epoch, eval_result=None):
        if not self.valid_evaluator:
            return

        interval_sec = float(getattr(self.params, 'validation_timeline_interval_sec', 300.0))
        if interval_sec <= 0:
            return

        if self.validation_timeline_start_time is None:
            self.setup_validation_timeline_csv()

        now = time.time()
        if now - self.validation_timeline_last_record_time < interval_sec:
            return

        result = eval_result
        if result is None:
            tic = time.time()
            result = self.valid_evaluator.eval()
            logging.info(
                '\nTimed validation performance:' + str(result) + 'in ' + str(time.time() - tic)
            )

        self.record_validation_timeline(epoch, result)

    def get_cl_coef(self, epoch):
        return 0.2

    def train_epoch(self, epoch):
        total_loss = 0
        total_MI_loss = 0
        all_labels = []
        all_scores = []

        coef_cl = self.get_cl_coef(epoch)
        logging.info(f"Epoch {epoch} contrastive coefficient: {coef_cl}")

        print(self.params.collate_fn)
        dataloader = DataLoader(
            self.train_data,
            batch_size=self.params.batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=self.params.collate_fn_train
        )

        self.graph_classifier.train()
        model_params = list(self.graph_classifier.parameters())

        for b_idx, batch in enumerate(dataloader):

            data_pos, targets_pos, data_neg, targets_neg, data_cont1, data_cont2 =\
                self.params.move_batch_to_device_train(batch, self.params.device)


            self.optimizer.zero_grad()
            self.graph_classifier.train()

            score_pos, s_g_pos, _ = self.graph_classifier(data_pos, is_return_emb=True)
            score_neg = self.graph_classifier(data_neg)

            pos_scores = score_pos.view(-1)
            neg_scores = score_neg.view(pos_scores.shape[0], -1).mean(dim=1)
            target = torch.ones_like(pos_scores, device=self.params.device)
            loss = self.criterion(pos_scores, neg_scores, target)

            cl_loss = torch.tensor(0.0, device=self.params.device)

            if coef_cl > 0:

                _, s_g_cont1, _ = self.graph_classifier(data_cont1, is_return_emb=True, cor_graph=True)
                _, s_g_cont2, _ = self.graph_classifier(data_cont2, is_return_emb=True, cor_graph=True)

                cl_loss = self.graph_classifier.graph_cl_loss(s_g_cont1, s_g_cont2, temperature=self.temperature)

                print(f'epoch: {epoch}, supervised loss: {loss}, CL loss: {cl_loss}, coef_cl: {coef_cl}')
                loss = loss + coef_cl * cl_loss

            loss.backward()
            self.optimizer.step()
            self.updates_counter += 1

            with torch.no_grad():
                all_scores.extend(score_pos.view(-1).detach().cpu().tolist())
                all_labels.extend(targets_pos.view(-1).detach().cpu().tolist())
                all_scores.extend(score_neg.view(-1).detach().cpu().tolist())
                all_labels.extend(targets_neg.view(-1).detach().cpu().tolist())

                total_loss += loss.item()
                total_MI_loss += cl_loss.item()

            eval_result = None
            if self.valid_evaluator and self.params.eval_every_iter and self.updates_counter % self.params.eval_every_iter == 0:
                tic = time.time()
                eval_result = self.valid_evaluator.eval()
                logging.info('\nPerformance:' + str(eval_result) + 'in ' + str(time.time() - tic))

                if eval_result['auc'] >= self.best_metric:
                    self.save_classifier()
                    self.best_metric = eval_result['auc']
                    self.not_improved_count = 0
                else:
                    self.not_improved_count += 1
                    if self.not_improved_count > self.params.early_stop:
                        logging.info(
                            f"Validation performance didn\'t improve for {self.params.early_stop} epochs. Training stops."
                        )
                        break

                self.last_metric = eval_result['auc']

            self.maybe_record_timed_validation(epoch, eval_result)

        auc = metrics.roc_auc_score(all_labels, all_scores)
        auc_pr = metrics.average_precision_score(all_labels, all_scores)
        weight_norm = sum(map(lambda x: torch.norm(x), model_params))

        return total_loss, total_MI_loss, auc, auc_pr, weight_norm

    def train(self):
        self.reset_training_state()
        self.setup_validation_timeline_csv()

        for epoch in range(1, self.params.num_epochs + 1):
            time_start = time.time()
            loss, MI_loss, auc, auc_pr, weight_norm = self.train_epoch(epoch)
            time_elapsed = time.time() - time_start

            logging.info(
                f'Epoch {epoch} with loss: {loss}, MI loss: {MI_loss}, '
                f'training auc: {auc}, training auc_pr: {auc_pr}, '
                f'best validation AUC: {self.best_metric}, weight_norm: {weight_norm} '
                f'in {time_elapsed}'
            )

            if epoch % self.params.save_every == 0:
                torch.save(
                    self.graph_classifier,
                    os.path.join(self.params.exp_dir, 'graph_classifier_chk.pth')
                )

    def save_classifier(self):
        torch.save(
            self.graph_classifier,
            os.path.join(self.params.exp_dir, 'best_graph_classifier.pth')
        )
        logging.info('Better models found w.r.t accuracy. Saved it!')

