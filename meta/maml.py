import torch
import torch.nn as nn
from torch import optim
from util import evaluate_prediction
import random
import torch.nn.functional as F
import os
import numpy as np
from model.Loss import WeightBCEWithLogitsLoss
from torch.utils.checkpoint import checkpoint


class MAML(nn.Module):
    def __init__(self, args, model):
        super(MAML, self).__init__()
        self.args = args
        self.model = model
        self.query_node_num = args.query_node_num
        self.meta_lr = args.meta_lr
        self.update_lr = args.learning_rate
        self.num_shots = args.num_shots
        self.update_step = args.update_step
        self.update_step_test = args.update_step_test
        self.accumulation_steps=4
        self.meta_optimizer = optim.Adam(self.model.parameters(), lr = self.meta_lr, weight_decay=self.args.weight_decay)
        self.criterion = WeightBCEWithLogitsLoss()
        if self.args.cuda:
            self.model.to(self.args.device)
        self.task_cnt = 0

    def train_MAML(self, train_tasks):

        support_batches = [ task.get_support_batch(isMaml=True) for task in train_tasks]
        query_batches = [ task.get_query_batch(isMaml=True) for task in train_tasks]
        for epoch in range(self.args.epochs):
            # manually shuffle tasks
            tasks_batches = list(zip(support_batches, query_batches))
            random.shuffle(tasks_batches)
            support_batches, query_batches = zip(*tasks_batches)
            # execute one epoch maml training
            loss_q = self.train_loop(support_batches, query_batches,train_tasks,epoch)


    def train_loop(self, support_batches, query_batches,train_tasks,epoch):
        # Training one epoch
        num_tasks = len(support_batches)
        bce_losses_q = [0 for _ in range(self.update_step)]
        loss_out=0
        for i in range(num_tasks):
            support_batch = support_batches[i]
            query_batch = query_batches[i]
            task=train_tasks[i]

            fast_parameters = list(self.model.parameters())

            for weight in self.model.parameters():
                weight.fast = None

            # Train the i-th task by 'update_step' times
            for k in range(0, self.update_step):
                output, mask = self.model(support_batch.to(self.args.device))
                output = output.squeeze()
                loss = self.criterion(output, support_batch.y, mask)
                grad = torch.autograd.grad(loss, fast_parameters, create_graph=True, allow_unused=True)

                fast_parameters = list()
                # fast update the model's parameters
                for index, weight in enumerate(self.model.parameters()):
                    if weight.fast is None:
                        weight.fast = weight - self.update_lr * grad[index]
                    else:
                        #print(index, grad[index])
                        weight.fast = weight.fast - self.update_lr * grad[index]
                    fast_parameters.append(weight.fast)
                query_output, mask=self.model(query_batch.to(self.args.device))
                query_output = query_output.squeeze()
                loss_q = self.criterion(query_output, query_batch.y, mask)
            loss=loss_q/num_tasks
            loss.backward()
            loss_out=loss+loss_out
            del support_batch
            del query_batch
            torch.cuda.empty_cache()
        self.meta_optimizer.step()
        self.meta_optimizer.zero_grad()
        return loss_out.item()

    def evaluate_maml(self, test_tasks):
        support_batches = [task.get_support_batch(isMaml=True) for task in test_tasks]
        query_batches = [task.get_query_batch(isMaml=True) for task in test_tasks]
        self.model=self.model.cpu()

        num_tasks = len(support_batches)
        predict_all=[]
        targets_all=[]
        for i in range(num_tasks):
            support_batch = support_batches[i]
            query_batch = query_batches[i]
            fast_parameters = list(self.model.parameters())
            # Init the fast weight
            for weight in self.model.parameters():
                weight.fast = None
            self.model.zero_grad()
            for k in range(0, self.update_step_test):
                output, mask = self.model(support_batch)
                output = output.squeeze()
                loss = self.criterion(output, support_batch.y, mask)
                grad = torch.autograd.grad(loss, fast_parameters, create_graph=True, allow_unused=True)
                fast_parameters = list()
                for index, weight in enumerate(self.model.parameters()):
                    if weight.fast is None:
                        weight.fast = weight - self.update_lr * grad[index]
                    else:
                        weight.fast = weight.fast - self.update_lr * grad[index]
                    fast_parameters.append(weight.fast)
            # test the queries of the i-th task
            query_output,_ = self.model(query_batch)
            query_output = query_output.squeeze()
            pred = torch.sigmoid(query_output)
            pred = torch.where(pred > 0.5, 1, 0)
            pred, targets = pred.cpu().detach().numpy(), query_batch.y.cpu().detach().numpy()
            acc, precision, recall, f1 = evaluate_prediction(pred, targets)
            pred=pred.reshape(-1,1)
            targets=targets.reshape(-1,1)
            print("Test Task-{}: Acc={:.4f}, Pre={:.4f}, Recall={:.4f}, F1={:.4f}".format(i, acc, precision, recall, f1))
            if len(predict_all)==0:
                predict_all=pred
                target_all=targets
            else:
                predict_all = np.vstack((predict_all,pred))
                target_all = np.vstack((target_all,targets))
            torch.cuda.empty_cache()
            del support_batch
            del query_batch
        acc, precision, recall, f1 = evaluate_prediction(predict_all, target_all)
        print("MAML Test Result: Acc={:.4f}, Pre={:.4f}, Recall={:.4f}, F1={:.4f}".format(acc, precision, recall, f1))

    def save_checkpoint(self, checkpoint_name:str=""):
        # save state
        if not os.path.exists(self.args.save_model_dir):
            os.makedirs(self.args.save_model_dir)
        checkpoint_name = checkpoint_name if checkpoint_name \
            else '_'.join(["MAML", str(self.args.num_layers), self.args.gnn_type, self.args.embed_type,
                           str(self.args.train_ratio), str(self.query_node_num), str(self.args.epochs)]) + '.pt'
        checkpoint = os.path.join(self.args.save_model_dir, checkpoint_name)
        torch.save({'model': self.model.state_dict()}, checkpoint)
        print("Save MAML model checkpoint in: {}".format(checkpoint))


    def load_checkpoint(self):
        assert os.path.exists(self.args.load_model_dir), "Model load directory does not exist!"
        checkpoint_name = self.args.model_file_name if self.args.model_file_name \
            else '_'.join(["MAML", str(self.args.num_layers), self.args.gnn_type, self.args.embed_type,
                           str(self.args.train_ratio), str(self.query_node_num), str(self.args.epochs)]) + '.pt'
        checkpoint = os.path.join(self.args.load_model_dir, checkpoint_name)
        assert os.path.isfile(checkpoint), "model {} does not exist!".format(
            checkpoint_name)

        ckpt = torch.load(checkpoint)
        self.model.load_state_dict(ckpt['model'])
        print("Load MAML model checkpoint from: {}".format(checkpoint))

