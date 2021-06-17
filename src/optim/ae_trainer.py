from collections import OrderedDict
from base.base_trainer import BaseTrainer
from base.base_dataset import BaseADDataset
from base.base_net import BaseNet
from sklearn.metrics import roc_auc_score
from utils.visualization.visualizer import Visualizer
from utils.performance import get_performance

import logging
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm


class AETrainer(BaseTrainer):

    def __init__(self, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 150, lr_milestones: tuple = (),
                 batch_size: int = 128, weight_decay: float = 1e-6, device: str = 'cuda', n_jobs_dataloader: int = 0, display_freq:int=None):
        super().__init__(optimizer_name, lr, n_epochs, lr_milestones, batch_size, weight_decay, device,
                         n_jobs_dataloader)

        # Results
        self.train_time = None
        self.test_auc = None
        self.test_time = None
        self.visualizer = Visualizer("deep_sad_ae")
        self.display_freq = display_freq

    def train(self, dataset: BaseADDataset, ae_net: BaseNet):
        logger = logging.getLogger()

        # Get train data loader
        train_loader, _ = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        # Set loss
        criterion = nn.MSELoss(reduction='none')

        # Set device
        ae_net = ae_net.to(self.device)
        criterion = criterion.to(self.device)

        # Set optimizer (Adam optimizer for now)
        optimizer = optim.Adam(ae_net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # Set learning rate scheduler
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_milestones, gamma=0.1)

        # Training
        logger.info('Starting pretraining...')
        start_time = time.time()
        
        for epoch in range(self.n_epochs):

            if epoch in self.lr_milestones:
                logger.info('  LR scheduler: new learning rate is %g' % float(scheduler.get_lr()[0]))

            epoch_loss = 0.0
            n_batches = 0
            epoch_start_time = time.time()
            self.train_one_epoch(ae_net, logger, train_loader, criterion, optimizer, scheduler, start_time, epoch, n_batches, epoch_start_time, epoch_loss)
            self.test(dataset, ae_net, epoch)
        
        return ae_net

    def train_one_epoch(self, ae_net, logger, train_loader, criterion, optimizer, scheduler, start_time, epoch, n_batches, epoch_start_time, epoch_loss):
        ae_net.train()
        with tqdm(train_loader, unit="batch") as tepoch:
            for data in tepoch:
                tepoch.set_description(f"Epoch {epoch}")
                inputs, _, _, _ = data
                inputs = inputs.to(self.device)

                    # Zero the network parameter gradients
                optimizer.zero_grad()

                    # Update network parameters via backpropagation: forward + backward + optimize
                rec = ae_net(inputs)
                if n_batches % self.display_freq:
                    self.visualizer.plot_current_images(inputs, train_or_test="train_ae_inputs", global_step=(1+epoch)*n_batches, denormalize=True)
                    self.visualizer.plot_current_images(rec, train_or_test="train_ae_recs", global_step=(1+epoch)*n_batches, denormalize=True)
                        
                rec_loss = criterion(rec, inputs)
                    
                loss = torch.mean(rec_loss)
                    
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
        logger.info('Pretraining Time: {:.3f}s'.format(self.train_time))
        logger.info('Finished pretraining epoch {}.'.format(str(epoch)))

        

    def test(self, dataset: BaseADDataset, ae_net: BaseNet, epoch=None):
        if not epoch:
            epoch = self.n_epochs+1
        logger = logging.getLogger()

        # Get test data loader
        _, test_loader = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        # Set loss
        criterion = nn.MSELoss(reduction='none')

        # Set device for network
        ae_net = ae_net.to(self.device)
        criterion = criterion.to(self.device)

        # Testing
        logger.info('Testing autoencoder...')
        epoch_loss = 0.0
        n_batches = 0
        start_time = time.time()
        idx_label_score = []
        ae_net.eval()
        with torch.no_grad():
            with tqdm(test_loader, unit="batch") as tepoch:
                for data in tepoch:
                    tepoch.set_description("Testing")
                    inputs, labels, _, idx = data
                    inputs, labels, idx = inputs.to(self.device), labels.to(self.device), idx.to(self.device)

                    rec = ae_net(inputs)
                    if n_batches % self.display_freq:
                        self.visualizer.plot_current_images(inputs, train_or_test="test_ae_inputs", global_step=(1+epoch)*n_batches, denormalize=True)
                        self.visualizer.plot_current_images(rec, train_or_test="test_ae_recs", global_step=(1+epoch)*n_batches, denormalize=True)
                    rec_loss = criterion(rec, inputs)
                    scores = torch.mean(rec_loss, dim=tuple(range(1, rec.dim())))

                    # Save triple of (idx, label, score) in a list
                    idx_label_score += list(zip(idx.cpu().data.numpy().tolist(),
                                                labels.cpu().data.numpy().tolist(),
                                                scores.cpu().data.numpy().tolist()))

                    loss = torch.mean(rec_loss)
                    epoch_loss += loss.item()
                    n_batches += 1

        self.test_time = time.time() - start_time

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
        logger.info('Finished testing autoencoder.')
