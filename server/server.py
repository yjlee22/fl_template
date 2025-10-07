import os
import time
import numpy as np

import torch
from utils import *
from dataset import Dataset
from torch.utils import data
import torch.nn.functional as F

from torchmetrics.classification import Precision, Recall, F1Score, AUROC

class Server(object):
    def __init__(self, device, model_func, init_model, init_par_list, datasets, method, args):
        # Basic attributes
        self.args = args
        self.device = device
        self.datasets = datasets
        self.model_func = model_func
        self.max_vram = np.zeros((args.comm_rounds))
        self.server_model = init_model
        self.server_model_params_list = init_par_list 
        
        print("Initialize the Server      --->  {:s}".format(self.args.method))
        # Public storage for client parameters
        print("Initialize the Public Storage:")
        self.clients_params_list = init_par_list.repeat(args.total_client, 1)
        print("   Local Model Param List  --->  {:d} * {:d}".format(
                self.clients_params_list.shape[0], self.clients_params_list.shape[1]))
        
        self.clients_updated_params_list = torch.zeros((args.total_client, init_par_list.shape[0]))
        print(" Local Updated Param List  --->  {:d} * {:d}".format(
                self.clients_updated_params_list.shape[0], self.clients_updated_params_list.shape[1]))

        # Test performance buffer per round:
        # columns = [loss, acc(%), precision, recall, f1, auc]
        self.test_perf = np.zeros((self.args.comm_rounds, 6))
              
        self.time = np.zeros((args.comm_rounds))
        self.lr = self.args.local_learning_rate
        
        # Transfer vectors (placeholder if needed)
        self.comm_vecs = {
            'Params_list': None,
        }
        self.received_vecs = None
        self.Client = None
    
    def _activate_clients_(self, t):
        # Randomly select active clients each round (with replacement)
        return np.random.choice(
            range(self.args.total_client),
            max(int(self.args.active_ratio * self.args.total_client), 1),
            replace=True
        )
            
    def _lr_scheduler_(self):
        # Simple multiplicative LR decay per round
        self.lr *= self.args.lr_decay
        
    def _test_(self, t, selected_clients):
        # Evaluate on the test split and store metrics
        loss, acc, prec, rec, f1, auc = self._validate_((self.datasets.test_x, self.datasets.test_y))
        self.test_perf[t] = [loss, acc, prec, rec, f1, auc]
        print(
            "    Test    ----    Loss: {:.4f},   Acc: {:.4f},   P: {:.4f},   R: {:.4f},   F1: {:.4f},   AUC: {:.4f}"
            .format(loss, acc, prec, rec, f1, auc),
            flush=True
        )
        
    def _summary_(self):
        # Build summary directory
        if not self.args.non_iid:
            summary_root = f'{self.args.out_file}/summary/IID'
        else:
            summary_root = f'{self.args.out_file}/summary/{self.args.split_rule}_{self.args.split_coef}'
        if not os.path.exists(summary_root):
            os.makedirs(summary_root)
        
        # Compute indices for maxima of each metric (except loss)
        # Column map: 0=loss, 1=acc, 2=precision, 3=recall, 4=f1, 5=auc
        best_acc_val = np.max(self.test_perf[:, 1])
        best_acc_idx = int(np.argmax(self.test_perf[:, 1]))
        best_prec_val = np.max(self.test_perf[:, 2])
        best_prec_idx = int(np.argmax(self.test_perf[:, 2]))
        best_recall_val = np.max(self.test_perf[:, 3])
        best_recall_idx = int(np.argmax(self.test_perf[:, 3]))
        best_f1_val = np.max(self.test_perf[:, 4])
        best_f1_idx = int(np.argmax(self.test_perf[:, 4]))
        best_auc_val = np.max(self.test_perf[:, 5])
        best_auc_idx = int(np.argmax(self.test_perf[:, 5]))

        summary_file = summary_root + f'/{self.args.method}.txt'
        with open(summary_file, 'w') as f:
            f.write("##=============================================##\n")
            f.write("##                   Summary                   ##\n")
            f.write("##=============================================##\n")
            f.write("Communication round   --->   T = {:d}\n".format(self.args.comm_rounds))
            f.write("Average Time / round   --->   {:.2f}s \n".format(np.mean(self.time)))
            f.write("Top-1 Test Acc (T)    --->   {:.2f}% ({:d})".format(best_acc_val, best_acc_idx))
            # Append only the four Max lines as requested
            f.write("\n")
            f.write("Max Precision (T)     --->   {:.4f} ({:d})\n".format(best_prec_val, best_prec_idx))
            f.write("Max Recall (T)        --->   {:.4f} ({:d})\n".format(best_recall_val, best_recall_idx))
            f.write("Max F1 Score (T)      --->   {:.4f} ({:d})\n".format(best_f1_val, best_f1_idx))
            f.write("Max AUC (T)           --->   {:.4f} ({:d})\n".format(best_auc_val, best_auc_idx))
        
        # Mirror summary to stdout (same structure)
        print("##=============================================##")
        print("##                   Summary                   ##")
        print("##=============================================##")
        print("     Communication round   --->   T = {:d}       ".format(self.args.comm_rounds))
        print("    Average Time / round   --->   {:.2f}s        ".format(np.mean(self.time)))
        print("     Top-1 Test Acc (T)    --->   {:.2f}% ({:d}) ".format(best_acc_val, best_acc_idx))
        print("     Max Precision (T)     --->   {:.4f} ({:d})  ".format(best_prec_val, best_prec_idx))
        print("     Max Recall (T)        --->   {:.4f} ({:d})  ".format(best_recall_val, best_recall_idx))
        print("     Max F1 Score (T)      --->   {:.4f} ({:d})  ".format(best_f1_val, best_f1_idx))
        print("     Max AUC (T)           --->   {:.4f} ({:d})  ".format(best_auc_val, best_auc_idx))
    
    def _validate_(self, dataset):
        """
        Multiclass evaluation routine.
        Returns: avg_loss, acc(%), precision, recall, f1, auc
        - Precision/Recall/F1/AUC use macro averaging across classes.
        - AUC uses one-vs-rest macro averaging with probability matrix.
        """
        self.server_model.eval()
        testloader = data.DataLoader(
            Dataset(dataset[0], dataset[1], train=False, dataset_name=self.args.dataset, args=self.args),
            batch_size=32, shuffle=False
        )

        total_loss = 0.0
        total_correct = 0
        total_seen = 0

        # Lazy-init metrics after the first forward pass to know num_classes
        precision_metric = None
        recall_metric = None
        f1_metric = None
        auroc_metric = None

        with torch.no_grad():
            for i, (inputs, labels) in enumerate(testloader):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device).long().squeeze(dim=-1)  # [N]

                logits = self.server_model(inputs)                      # [N, C]
                loss = F.cross_entropy(logits, labels, reduction='mean')
                total_loss += loss.item()

                # Accuracy accumulation
                preds = logits.argmax(dim=1)                            # [N]
                total_correct += (preds == labels).long().sum().item()
                total_seen += labels.numel()

                # Initialize metrics on first batch (multiclass only)
                if precision_metric is None:
                    num_classes = logits.shape[1]
                    precision_metric = Precision(task="multiclass", num_classes=num_classes, average="macro").to(self.device)
                    recall_metric    = Recall(   task="multiclass", num_classes=num_classes, average="macro").to(self.device)
                    f1_metric        = F1Score(  task="multiclass", num_classes=num_classes, average="macro").to(self.device)
                    auroc_metric     = AUROC(    task="multiclass", num_classes=num_classes, average="macro").to(self.device)

                # Probabilities for AUROC (multiclass expects [N, C] probs)
                probs = torch.softmax(logits, dim=1)                    # [N, C]

                # Update metrics
                precision_metric.update(preds, labels)
                recall_metric.update(preds, labels)
                f1_metric.update(preds, labels)
                auroc_metric.update(probs, labels)

        # Mean loss and accuracy over the epoch
        avg_loss = total_loss / (i + 1)
        acc = 100.0 * (total_correct / max(total_seen, 1))

        # Compute torchmetrics
        try:
            precision = precision_metric.compute().item()
            recall    = recall_metric.compute().item()
            f1        = f1_metric.compute().item()
            auc       = auroc_metric.compute().item()
        except Exception:
            precision, recall, f1, auc = 0.0, 0.0, 0.0, 0.0

        # Optional L2 regularization term added to loss (kept from original behavior)
        if self.args.weight_decay != 0.:
            with torch.no_grad():
                l2_term = (self.args.weight_decay / 2.0) * torch.sum(self.server_model_params_list * self.server_model_params_list)
            avg_loss += l2_term.item()

        return avg_loss, acc, precision, recall, f1, auc
    
    
    def _save_results_(self):
        # Save per-round metrics to .npy for downstream analysis
        if not self.args.non_iid:
            root = f'{self.args.out_file}/IID'
        else:
            root = f'{self.args.out_file}/{self.args.split_rule}_{self.args.split_coef}'
        if not os.path.exists(root):
            os.makedirs(root)
        
        out_path = root + f'/{self.args.method}.npy'
        np.save(out_path, self.test_perf)
                        
    def process_for_communication(self):
        # Placeholder for methods that need to pre-process server-to-client data
        pass
        
    def global_update(self, selected_clients, Averaged_update, Averaged_model):
        # Placeholder for the server aggregation rule (e.g., FedAvg)
        pass
    
    def postprocess(self, client, received_vecs):
        # Placeholder for methods that need to post-process client-to-server data
        pass
        
    def train(self):
        print("##=============================================##")
        print("##           Training Process Starts           ##")
        print("##=============================================##")
        
        Averaged_update = torch.zeros(self.server_model_params_list.shape)
        
        for t in range(self.args.comm_rounds):
            start = time.time()
            # Select active clients
            selected_clients = self._activate_clients_(t)
            print('============= Communication Round', t + 1, '=============', flush=True)
            print('Selected Clients: %s' % (', '.join(['%2d' % item for item in selected_clients])))
            
            for client in selected_clients:
                dataset = (self.datasets.client_x[client], self.datasets.client_y[client])
                self.process_for_communication(client, Averaged_update)
                _edge_device = self.Client(
                    device=self.device,
                    model_func=self.model_func,
                    received_vecs=self.comm_vecs,
                    dataset=dataset,
                    lr=self.lr,
                    args=self.args
                )
                self.received_vecs = _edge_device.train()
                self.clients_updated_params_list[client] = self.received_vecs['local_update_list']
                self.clients_params_list[client] = self.received_vecs['local_model_param_list']
                self.postprocess(client, self.received_vecs)
                
                # Release reference
                del _edge_device
            
            # Aggregate updates and models from active clients
            Averaged_update = torch.mean(self.clients_updated_params_list[selected_clients], dim=0)
            Averaged_model  = torch.mean(self.clients_params_list[selected_clients], dim=0)
            
            self.server_model_params_list = self.global_update(selected_clients, Averaged_update, Averaged_model)
            set_client_from_params(self.device, self.server_model, self.server_model_params_list)
            
            # Evaluate and update LR
            self._test_(t, selected_clients)
            self._lr_scheduler_()
            
            # Round time
            end = time.time()
            self.time[t] = end - start
            print("            ----    Time: {:.2f}s".format(self.time[t]))
            
        # Persist and summarize
        self._save_results_()
        self._summary_()
