from base.base_trainer import BaseTrainer
from base.base_dataset import BaseADDataset
from base.base_net import BaseNet
from torch.utils.data.dataloader import DataLoader
from sklearn.metrics import roc_auc_score
from utils.visualization.visualizer import Visualizer

import logging
import time
import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from utils.performance import get_performance



class DeepSADTrainer(BaseTrainer):

    def __init__(self, c, eta: float, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 150,
                 lr_milestones: tuple = (), batch_size: int = 128, weight_decay: float = 1e-6, device: str = 'cuda',
                 n_jobs_dataloader: int = 0, display_freq:int=None):
        super().__init__(optimizer_name, lr, n_epochs, lr_milestones, batch_size, weight_decay, device,
                         n_jobs_dataloader)

        # Deep SAD parameters
        self.c = torch.tensor(c, device=self.device) if c is not None else None
        self.eta = eta
        self.visualizer = Visualizer("deep_sad")
        self.display_freq = display_freq
        

        # Optimization parameters
        self.eps = 1e-6
        

        # Results
        self.train_time = None
        self.test_auc = None
        self.test_time = None
        self.test_scores = None

    def train(self, dataset: BaseADDataset, net: BaseNet):
        logger = logging.getLogger()

        # Get train data loader
        train_loader, _ = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        # Set device for network
        net = net.to(self.device)

        # Set optimizer (Adam optimizer for now)
        optimizer = optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # Set learning rate scheduler
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_milestones, gamma=0.1)

        # Initialize hypersphere center c (if c not loaded)
        if self.c is None:
            logger.info('Initializing center c...')
            self.c = self.init_center_c(train_loader, net)
            logger.info('Center c initialized.')

        # Training
        logger.info('Starting training...')
        start_time = time.time()
        
        for epoch in range(self.n_epochs):
            if epoch in self.lr_milestones:
                logger.info('  LR scheduler: new learning rate is %g' % float(scheduler.get_lr()[0]))

            epoch_loss = 0.0
            n_batches = 0
            epoch_start_time = time.time()
            self.train_one_epoch(net, logger, train_loader, optimizer, scheduler, start_time, epoch, n_batches, epoch_start_time, epoch_loss)
            self.test(dataset, net, epoch)
        

        return net

    def train_one_epoch(self, net, logger, train_loader, optimizer, scheduler, start_time, epoch, n_batches, epoch_start_time, epoch_loss):
        net.train()
        with tqdm(train_loader, unit="batch") as tepoch:
            for data in tepoch:
                tepoch.set_description(f"Epoch {epoch}")
                inputs, _, semi_targets, _ = data
                inputs, semi_targets = inputs.to(self.device), semi_targets.to(self.device)

                    # Zero the network parameter gradients
                optimizer.zero_grad()

                    # Update network parameters via backpropagation: forward + backward + optimize
                outputs = net(inputs)
                dist = torch.sum((outputs - self.c) ** 2, dim=1)
                losses = torch.where(semi_targets == 0, dist, self.eta * ((dist + self.eps) ** semi_targets.float()))
                loss = torch.mean(losses)
                loss.backward()
                optimizer.step()
                scheduler.step()
                tepoch.set_postfix(loss=loss.item())
                epoch_loss += loss.item()
                self.visualizer.plot_current_errors(total_steps=(1+epoch)*n_batches, errors={"Loss": loss.item()})
                n_batches += 1

            # log epoch statistics
        epoch_train_time = time.time() - epoch_start_time
        logger.info(f'| Epoch: {epoch + 1:03}/{self.n_epochs:03} | Train Time: {epoch_train_time:.3f}s '
                        f'| Train Loss: {epoch_loss / n_batches:.6f} |')

        self.train_time = time.time() - start_time
        logger.info('Training Time: {:.3f}s'.format(self.train_time))
        logger.info('Finished training epoch {}.'.format(str(epoch)))

    def test(self, dataset: BaseADDataset, net: BaseNet, epoch=None):
        logger = logging.getLogger()

        # Get test data loader
        _, test_loader = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)
        # Set device for network
        net = net.to(self.device)

        # Testing
        logger.info('Starting testing...')
        epoch_loss = 0.0
        n_batches = 0
        start_time = time.time()
        idx_label_score = []
        net.eval()
        with torch.no_grad():
            with tqdm(test_loader, unit="batch") as tepoch:
                for data in tepoch:
                    tepoch.set_description("Testing")
                    inputs, labels, semi_targets, idx = data
                    print("labels:", str(labels))
                    print("semi_targets:", str(semi_targets))
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    semi_targets = semi_targets.to(self.device)
                    idx = idx.to(self.device)

                    outputs = net(inputs)
                    dist = torch.sum((outputs - self.c) ** 2, dim=1)
                    print("dist:", str(dist))
                    print("semi_targets.float():", str(semi_targets.float()))
                    print("self.eps:", str(self.eps))
                    print("self.eta:", str(self.eta))
                    losses = torch.where(semi_targets == 0, dist, self.eta * ((dist + self.eps) ** semi_targets.float()))
                    print("losses:", str(losses))
                    loss = torch.mean(losses)
                    scores = dist

                    # Save triples of (idx, label, score) in a list
                    idx_label_score += list(zip(idx.cpu().data.numpy().tolist(),
                                                labels.cpu().data.numpy().tolist(),
                                                scores.cpu().data.numpy().tolist()))
                    print("idx_label_score:", str(idx_label_score))

                    epoch_loss += loss.item()
                    n_batches += 1

        self.test_time = time.time() - start_time
        self.test_scores = idx_label_score

        # Compute AUC
        _, labels, scores = zip(*idx_label_score)
        labels = np.array(labels)
        scores = np.array(scores)
        self.test_auc = roc_auc_score(labels, scores)
        self.visualizer.plot_performance(epoch=epoch, performance=get_performance(y_trues=labels, y_preds=scores))

        # Log results
        logger.info('Test Loss: {:.6f}'.format(epoch_loss / n_batches))
        logger.info('Test AUC: {:.2f}%'.format(100. * self.test_auc))
        logger.info('Test Time: {:.3f}s'.format(self.test_time))
        logger.info('Finished testing epoch {}.'.format(epoch))

    def init_center_c(self, train_loader: DataLoader, net: BaseNet, eps=0.1):
        """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
        n_samples = 0
        c = torch.zeros(net.rep_dim, device=self.device)

        net.eval()
        with torch.no_grad():
            for data in train_loader:
                # get the inputs of the batch
                inputs, _, _, _ = data
                inputs = inputs.to(self.device)
                outputs = net(inputs)
                n_samples += outputs.shape[0]
                c += torch.sum(outputs, dim=0)

        c /= n_samples

        # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps

        return c
