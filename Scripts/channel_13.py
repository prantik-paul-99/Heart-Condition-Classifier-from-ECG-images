# %% [code]
!pip install gdown

# %% [code] {"execution":{"iopub.status.busy":"2023-02-20T16:14:37.599726Z","iopub.execute_input":"2023-02-20T16:14:37.600201Z","iopub.status.idle":"2023-02-20T16:14:48.197194Z","shell.execute_reply.started":"2023-02-20T16:14:37.600110Z","shell.execute_reply":"2023-02-20T16:14:48.195929Z"}}
# !gdown --folder 'https://drive.google.com/drive/folders/1r-VVeY47jW8O78oebrzzn_b8KhCX4HVC?usp=share_link'
!gdown --folder 'https://drive.google.com/drive/folders/1bmlGhMrsL-911W9sOivk_2URdW8zp2hm?usp=sharing'

# %% [code] {"execution":{"iopub.status.busy":"2023-02-20T16:15:07.604697Z","iopub.execute_input":"2023-02-20T16:15:07.605741Z","iopub.status.idle":"2023-02-20T16:15:08.594238Z","shell.execute_reply.started":"2023-02-20T16:15:07.605697Z","shell.execute_reply":"2023-02-20T16:15:08.593015Z"}}
%cd /kaggle/working/npy_files
%ls

# %% [code] {"execution":{"iopub.status.busy":"2023-02-20T16:15:08.597103Z","iopub.execute_input":"2023-02-20T16:15:08.597627Z","iopub.status.idle":"2023-02-20T16:15:08.603660Z","shell.execute_reply.started":"2023-02-20T16:15:08.597562Z","shell.execute_reply":"2023-02-20T16:15:08.602597Z"}}
import numpy as np
import pandas as pd
from tqdm import tqdm

# %% [code] {"execution":{"iopub.status.busy":"2023-02-20T16:15:08.605112Z","iopub.execute_input":"2023-02-20T16:15:08.605741Z","iopub.status.idle":"2023-02-20T16:15:08.615180Z","shell.execute_reply.started":"2023-02-20T16:15:08.605703Z","shell.execute_reply":"2023-02-20T16:15:08.614138Z"}}
def get_mini_batch(input_X, label, batch_size):
    for i in range(0, len(input_X), batch_size):
        yield input_X[i: i + batch_size], label[i: i + batch_size]

def get_mini(input_X, batch_size):
    for i in range(0, len(input_X), batch_size):
        yield input_X[i: i + batch_size]

# %% [code] {"execution":{"iopub.status.busy":"2023-02-20T16:15:08.618976Z","iopub.execute_input":"2023-02-20T16:15:08.619976Z","iopub.status.idle":"2023-02-20T16:15:08.629550Z","shell.execute_reply.started":"2023-02-20T16:15:08.619939Z","shell.execute_reply":"2023-02-20T16:15:08.628537Z"}}
def load_images_from_npy():
    covid_images = np.load('./covid.npy', allow_pickle=True)
    hb_images = np.load('./hb.npy', allow_pickle=True)
    mi_images = np.load('./mi.npy', allow_pickle=True)
    normal_images = np.load('./normal.npy', allow_pickle=True)
    pmi_images = np.load('./pmi.npy', allow_pickle=True)


    # covid - 0
    # hb - 1
    # mi - 2
    # normal - 3
    # pmi - 4

    # fill labels 0 for covid, 1 for hb, 2 for mi, 3 for normal, 4 for pmi
    labels = [0] * (len(covid_images) // 13)
    labels.extend([1] * (len(hb_images) // 13))
    labels.extend([2] * (len(mi_images) // 13))
    labels.extend([3] * (len(normal_images) // 13))
    labels.extend([4] * (len(pmi_images) // 13))

    images = np.concatenate((covid_images, hb_images, mi_images, normal_images, pmi_images), axis=0)

    images = np.array(images)
    labels = np.array(labels)

    return images, labels



def combine_leads_of_image(data):
  
    # store each 13 lead images in a list of 13*28*28 arrays
    images = []

    for X in get_mini(data, 13):
        images.append(X)

    images = np.array(images)
    return images

# %% [code] {"execution":{"iopub.status.busy":"2023-02-20T16:15:08.631056Z","iopub.execute_input":"2023-02-20T16:15:08.631614Z","iopub.status.idle":"2023-02-20T16:15:08.696397Z","shell.execute_reply.started":"2023-02-20T16:15:08.631576Z","shell.execute_reply":"2023-02-20T16:15:08.695342Z"}}
images, labels = load_images_from_npy()
print(images.shape, labels.shape, images[0].shape)

images = combine_leads_of_image(images)
print(images.shape)

# %% [code] {"execution":{"iopub.status.busy":"2023-02-20T16:15:31.039437Z","iopub.execute_input":"2023-02-20T16:15:31.040489Z","iopub.status.idle":"2023-02-20T16:15:31.049419Z","shell.execute_reply.started":"2023-02-20T16:15:31.040443Z","shell.execute_reply":"2023-02-20T16:15:31.048267Z"}}
def train_val_split(images, labels, split_factor=0.8, shuffle=True):
    # if shuffle:
    random_indices = np.random.choice(len(images), len(images), replace=False)
    training_data_full = images[random_indices]
    training_label_full = labels[random_indices]

#     training_data_full = training_data_full[:1500]
#     training_label_full = training_label_full[:1500]

    split_index_train = int(split_factor * len(training_data_full))
    split_index_test = int(0.9 * len(training_data_full))

    training_data = training_data_full[:split_index_train]
    training_label = training_label_full[:split_index_train]

    validation_data = training_data_full[split_index_train : split_index_test]
    validation_label = training_label_full[split_index_train : split_index_test]

    test_data = training_data_full[split_index_test :]
    test_label = training_label_full[split_index_test :]

    return training_data, training_label, validation_data, validation_label, test_data, test_label

# %% [code] {"execution":{"iopub.status.busy":"2023-02-20T16:15:08.708154Z","iopub.execute_input":"2023-02-20T16:15:08.708810Z","iopub.status.idle":"2023-02-20T16:15:11.832461Z","shell.execute_reply.started":"2023-02-20T16:15:08.708770Z","shell.execute_reply":"2023-02-20T16:15:11.831278Z"}}
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.optim.lr_scheduler as lr_scheduler

# %% [code] {"execution":{"iopub.status.busy":"2023-02-20T16:15:11.834111Z","iopub.execute_input":"2023-02-20T16:15:11.834826Z","iopub.status.idle":"2023-02-20T16:15:11.844980Z","shell.execute_reply.started":"2023-02-20T16:15:11.834783Z","shell.execute_reply":"2023-02-20T16:15:11.843631Z"}}
class HybridNetwork(nn.Module):
    def __init__(self):
        super(HybridNetwork, self).__init__()

        # build LeNet-5 using nn.Sequential
        self.layers = nn.Sequential(
            nn.LazyConv2d(out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.LazyConv2d(out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # nn.LazyConv3d(out_channels=256, kernel_size=3, padding=1),
            # nn.ReLU(),
            # nn.MaxPool3d(kernel_size=2, stride=2),

            # nn.LazyConv3d(out_channels=512, kernel_size=3, padding=1),
            # nn.ReLU(),
            # nn.MaxPool3d(kernel_size=2, stride=2),

            # nn.LazyConv3d(out_channels=512, kernel_size=3, padding=1),
            # nn.ReLU(),
            # nn.MaxPool3d(kernel_size=2, stride=2),

            # nn.LSTM(input_size=(8, 512), batch_first=True, dropout=0.25, bidirectional=True, hidden_size=10)

            nn.Flatten(start_dim=1),
            nn.LazyLinear(64),
            nn.ReLU(),
            nn.LazyLinear(5),
            nn.LogSoftmax(dim=1)
        )


    def forward(self, input_X):

        for layer in self.layers:
            input_X = layer(input_X)
            # print(input_X.shape)

        return input_X

# %% [code] {"execution":{"iopub.status.busy":"2023-02-20T16:19:38.170255Z","iopub.execute_input":"2023-02-20T16:19:38.170705Z","iopub.status.idle":"2023-02-20T16:19:38.198316Z","shell.execute_reply.started":"2023-02-20T16:19:38.170666Z","shell.execute_reply":"2023-02-20T16:19:38.196941Z"}}
class Model:
    def __init__(self):
        self.network = HybridNetwork()
        self.num_epochs = 10
        self.batch_size = 16
        self.grad_clipping = 10.0
        self.optimizer_type = 'adamax'
        self.lr = 0.001
        self.num_output_units = 5

        self.use_cuda = torch.cuda.is_available()
        print('Use cuda:', self.use_cuda)

        if self.use_cuda:
            torch.cuda.set_device(0)
            self.network.cuda()    

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


    def reshape_image(self, input_X, label):

        X_full = input_X.reshape(input_X.shape[0], input_X.shape[1], input_X.shape[2], input_X.shape[3])
        y_full = np.eye(self.num_output_units)[label.astype(int)]

        return X_full, y_full

    def report_num_trainable_parameters(self):
        num_parameters = 0
        for p in self.network.parameters():
            if p.requires_grad:
                sz = list(p.size())

                num_parameters += np.prod(sz)
        print('Number of parameters: ', num_parameters)


    def predict(self, test_data, test_label):
        self.network.eval()

        X = test_data
        y = test_label

        X = self.normalize(X)
        cnt = 0

        for X_batch, y_batch in tqdm(get_mini_batch(input_X=X, label=y, batch_size=self.batch_size), desc="Testing"):

            # reshape to appropriate format
            X, y_true = self.reshape_image(X_batch, y_batch)

            # convert to float32 and convert to torch tensor
            X = X.astype(np.float32)
            y_true = y_true.astype(np.float32)
            X = torch.from_numpy(X)
            y_true = torch.from_numpy(y_true)

            X = X.to("cuda:0")
            y_true = y_true.to("cuda:0")

            pred_proba = self.network(X)

            y_pred = torch.argmax(pred_proba, dim=1)
            y_true = torch.argmax(y_true, dim=1)

            cnt += torch.sum(y_pred == y_true).item()

        print(f'accuracy: {cnt/test_data.shape[0]}')


    def evaluate(self, validation_data, validation_label):
        self.network.eval()

        X = validation_data
        y = validation_label

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

            # X = X.cuda(non_blocking=True)
            X = X.to("cuda:0")
            y_true = y_true.to("cuda:0")

            pred_proba = self.network(X)

            y_pred = torch.argmax(pred_proba, dim=1)
            y_true = torch.argmax(y_true, dim=1)

            cnt += torch.sum(y_pred == y_true).item()

        print(f'accuracy: {cnt/validation_data.shape[0]}')


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

            X = X.to("cuda:0")
            y_true = y_true.to("cuda:0")

            pred_proba = self.network(X)

            loss = F.cross_entropy(pred_proba, y_true)
            self.optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.grad_clipping)

            # Update parameters
            self.optimizer.step()

            self.updates += 1
            iter_cnt += 1

#             if self.updates % 1000 == 0:
#                 print('Iter: %d/%d, Loss: %f' % (iter_cnt, num_iter, loss.item()))
        
        print('Iter: %d/%d, Loss: %f' % (iter_cnt, num_iter, loss.item()))
        self.scheduler.step()
        print('LR:', self.scheduler.get_last_lr()[0])            


    def fit(self, training_data, training_label, validation_data, validation_label):

        X = training_data
        y = training_label

        X = self.normalize(X)

        self.network.train()

        for epoch in range(self.num_epochs):
            self.train(input_X=X, label=y)
            print("Epoch: ", epoch)

            self.evaluate(validation_data, validation_label)

# %% [code] {"execution":{"iopub.status.busy":"2023-02-20T16:19:43.726696Z","iopub.execute_input":"2023-02-20T16:19:43.727098Z","iopub.status.idle":"2023-02-20T16:19:50.420377Z","shell.execute_reply.started":"2023-02-20T16:19:43.727063Z","shell.execute_reply":"2023-02-20T16:19:50.419268Z"}}
def main():
    training_data, training_label, validation_data, validation_label, test_data, test_label = train_val_split(images=images, labels=labels, split_factor=0.8, shuffle=True)
    print(training_data.shape, training_label.shape, validation_data.shape, validation_label.shape, test_data.shape, test_label.shape)
   
    model = Model()

    model.evaluate(validation_data, validation_label)
    model.fit(training_data, training_label, validation_data, validation_label)

    model.predict(test_data, test_label)

main()