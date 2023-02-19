import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.optim.lr_scheduler as lr_scheduler

from network import Network


class Model:

  def __init__(self):
    self.network = Network()
    self.num_epochs = 10
    self.batch_size = 16
    self.grad_clipping = 10.0
    self.optimizer_type = 'adamax'
    self.lr = 0.001
    self.num_output_units = 10

    self.init_optimizer()

  def init_optimizer(self):
    parameters = [p for p in self.network.parameters() if p.requires_grad]
    if self.optimizer_type == 'sgd':
        self.optimizer = optim.SGD(parameters, self.lr,
                                    momentum=0.4,
                                    weight_decay=0)
    elif self.optimizer_type == 'adamax':
        self.optimizer = optim.Adamax(parameters,
                                    lr=self.lr,
                                    weight_decay=0)
    else:
        raise RuntimeError('Unsupported optimizer: %s' %
                            self.optimizer_type)
    self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, milestones=[10, 15], gamma=0.5)

  
  def normalize(self, X):
    # apply standard normalization on x. mean = 0, std = 1
    X = (X - np.mean(X)) / np.std(X)

    return X

  def reshape_format(self, input):

    X = np.array([x[0] for x in input])
    y = np.array([x[1] for x in input])

    print('in reshape_format', X.shape, y.shape)

    return X, y

  def reshape_image(self, input_X, label):

      X_full = input_X.reshape(input_X.shape[0], 1, input_X.shape[1], input_X.shape[2])
      y_full = np.eye(self.num_output_units)[label.astype(int)]

      return X_full, y_full


  def report_num_trainable_parameters(self):
    num_parameters = 0
    for p in self.network.parameters():
        if p.requires_grad:
            sz = list(p.size())

            num_parameters += np.prod(sz)
    print('Number of parameters: ', num_parameters)


  def evaluate(self, input):
    self.network.eval()

    X, y = self.reshape_format(input)
    X = self.normalize(X)
    cnt = 0

    for X_batch, y_batch in tqdm(get_mini_batch(input_X=X, label=y, batch_size=self.batch_size), desc="Evaluating"):
        
        # reshape to appropriate format
        X, y_true = self.reshape_image(X_batch, y_batch)

        # convert to float32 and convert to torch tensor
        X = X.astype(np.float32)
        y_true = y_true.astype(np.float32)
        X = torch.from_numpy(X)
        y_true = torch.from_numpy(y_true)

        pred_proba = self.network(X)

        y_pred = torch.argmax(pred_proba, dim=1)
        y_true = torch.argmax(y_true, dim=1)   

        cnt += torch.sum(y_pred == y_true).item()
    
    print(f'accuracy: {cnt/input.shape[0]}')


  def train(self, input_X, label):
    self.updates = 0
    iter_cnt, num_iter = 0, (len(input_X) + self.batch_size - 1) // self.batch_size

    for X_batch, y_batch in tqdm(get_mini_batch(input_X=input_X, label=label, batch_size=self.batch_size), desc="Training"):
        
        # reshape to appropriate format
        X, y_true = self.reshape_image(X_batch, y_batch)
        
        # convert to float32 and convert to torch tensor
        X = X.astype(np.float32)
        y_true = y_true.astype(np.float32)
        X = torch.from_numpy(X)
        y_true = torch.from_numpy(y_true)

        pred_proba = self.network(X)

        loss = F.cross_entropy(pred_proba, y_true)
        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm(self.network.parameters(), self.grad_clipping)

        # Update parameters
        self.optimizer.step()

        self.updates += 1
        iter_cnt += 1

        if self.updates % 1000 == 0:
            print('Iter: %d/%d, Loss: %f' % (iter_cnt, num_iter, loss.item()))
    
    self.scheduler.step()
    print('LR:', self.scheduler.get_last_lr()[0])            


  def fit(self, input, validation):

      X, y = self.reshape_format(input)
      X = self.normalize(X)

      self.network.train()

      for epoch in range(self.num_epochs):
          self.train(input_X=X, label=y)
          print("Epoch: ", epoch)

          self.evaluate(validation)