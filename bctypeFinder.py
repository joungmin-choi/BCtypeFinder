import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import pandas as pd 
import os
import numpy as np
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class MyBaseDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data
        
    def __getitem__(self, index): 
        return self.x_data[index], self.y_data[index]
        
    def __len__(self): 
        return self.x_data.shape[0]


class DomainDataset(Dataset) :
    def __init__(self, x_data, y_data, z_data):
        self.x_data = x_data
        self.y_data = y_data
        self.z_data = z_data

    def __getitem__(self, index): 
        return self.x_data[index], self.y_data[index], self.z_data[index]
        
    def __len__(self): 
        return self.x_data.shape[0]

device = (
    #torch.device('cuda:0')
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print(f"Using {device} device")

result_dir = "./results/"
os.makedirs(result_dir, exist_ok = True)

x_filename = sys.argv[1] 
y_filename = sys.argv[2] 
target_filename = sys.argv[3] 
test_target_filename = sys.argv[4] 

raw_x = pd.read_csv(x_filename, index_col = 0)
raw_y = pd.read_csv(y_filename, index_col = 0)

raw_test_target_x = pd.read_csv(test_target_filename, index_col = 0)
raw_test_target_y = raw_test_target_x['subtype'].values
del raw_test_target_x['subtype']
raw_test_target_x = raw_test_target_x.values

raw_target_x = pd.read_csv(target_filename, index_col = 0)
raw_target_domain_y = raw_target_x['domain_idx'].tolist()

y_train = raw_y['subtype'].tolist()
num_subtype = len(set(y_train))
y_train = np.array(y_train)

del raw_target_x['domain_idx']

raw_target_x = raw_target_x.values
x_train = raw_x.values

domain_x = np.append(x_train, raw_target_x, axis = 0)

raw_source_domain_y = np.zeros(len(y_train), dtype = int) 
domain_y = np.append(raw_source_domain_y, raw_target_domain_y)

num_domain = len(set(domain_y))

x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train)

domain_x = torch.from_numpy(domain_x)
domain_y = torch.from_numpy(domain_y)

target_x = torch.from_numpy(raw_target_x)
target_init_y = torch.randint(low=0, high=num_subtype, size = (len(target_x),))

domain_z = torch.cat((y_train, target_init_y), 0)

raw_test_target_x = torch.from_numpy(raw_test_target_x)
raw_test_target_y = torch.from_numpy(raw_test_target_y)

num_feature = len(x_train[0])
num_train = len(x_train)
num_test = len(raw_target_x)

train_dataset = MyBaseDataset(x_train, y_train)
domain_dataset = DomainDataset(domain_x, domain_y, domain_z)
target_dataset = MyBaseDataset(target_x, target_init_y)
test_target_dataset = MyBaseDataset(raw_test_target_x, raw_test_target_y)

batch_size = 128
target_batch_size = 128
test_target_batch_size = 64

train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
domain_dataloader = DataLoader(domain_dataset, batch_size = batch_size, shuffle = True)
target_dataloader = DataLoader(target_dataset, batch_size = target_batch_size, shuffle = True)
test_target_dataloader = DataLoader(test_target_dataset, batch_size = test_target_batch_size)


n_fe_embed1 = 1024
n_fe_embed2 = 512
n_c_h1 = 256
n_c_h2 = 64
n_d_h1 = 256
n_d_h2 = 64


class FeatureExtractor(nn.Module) :
    def __init__(self) :
        super().__init__()
        self.feature_layer = nn.Sequential(
            nn.Linear(num_feature, n_fe_embed1),
            nn.LeakyReLU(),
            nn.Linear(n_fe_embed1, n_fe_embed2),
            nn.LeakyReLU()
            )
    def forward(self, x) :
        embedding = self.feature_layer(x)
        return embedding


class DomainDiscriminator(nn.Module) :
    def __init__(self) :
        super().__init__()
        self.disc_layer = nn.Sequential(
            nn.Linear(n_fe_embed2, n_d_h1),
            nn.LeakyReLU(),
            nn.Linear(n_d_h1, n_d_h2),
            nn.LeakyReLU(),
            nn.Linear(n_d_h2, num_domain)
            )
    def forward(self, x) :
        domain_logits = self.disc_layer(x)
        return domain_logits


class SubtypeClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(n_fe_embed2, n_c_h1),
            nn.LeakyReLU(),
            nn.Linear(n_c_h1, n_c_h2),
            nn.LeakyReLU(),
            nn.Linear(n_c_h2, num_subtype)
        )
    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits


feature_extract_model = FeatureExtractor().to(device)
domain_disc_model = DomainDiscriminator().to(device)
subtype_pred_model = SubtypeClassifier().to(device)

c_loss = nn.CrossEntropyLoss() # Already have softmax
domain_loss = nn.CrossEntropyLoss() # Already have softmax


fe_optimizer = torch.optim.Adam(feature_extract_model.parameters(), lr=1e-4)
c_optimizer = torch.optim.Adam(subtype_pred_model.parameters(), lr=1e-5)
d_optimizer = torch.optim.Adam(domain_disc_model.parameters(), lr=1e-6)


def pretrain_classifier(epoch, dataloader, fe_model, c_model, c_loss, fe_optimizer, c_optimizer):
    size = len(dataloader.dataset)
    correct = 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        X = X.float()
        extracted_feature = fe_model(X)
        pred = c_model(extracted_feature)
        loss = c_loss(pred, y)
        fe_optimizer.zero_grad()
        c_optimizer.zero_grad()
        loss.backward()
        fe_optimizer.step()
        c_optimizer.step()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    loss = loss.item()
    correct /= size
    if epoch % 10 == 0 :
        print(f"[PT Epoch {epoch+1}] \tTraining loss: {loss:>5f}, Training Accuracy: {(100*correct):>0.2f}%")


def class_alignment_train(epoch, domain_dataloader, fe_model, fe_optimizer, c_optimizer) :
    for batch, (X, y_domain, z_subtype) in enumerate(domain_dataloader):
        X, y_domain, z_subtype = X.to(device), y_domain.to(device), z_subtype.to(device)
        X = X.float()
        batch_subtype_list = z_subtype.unique()
        X_embed = fe_model(X)
        #
        align_loss = torch.zeros((1) ,dtype = torch.float64)
        align_loss = align_loss.to(device)
        #
        for subtype in batch_subtype_list :
            sample_idx_list = (z_subtype == subtype).nonzero(as_tuple = True)[0]
            if len(sample_idx_list) < 1 :
                continue
            tmp_x = X_embed[sample_idx_list]
            tmp_y = y_domain[sample_idx_list]
            tmp_z = z_subtype[sample_idx_list]
            batch_domain_list = tmp_y.unique()
            domain_centroid_stack = []
            for domain in batch_domain_list :
                domain_idx_list = (tmp_y == domain).nonzero(as_tuple = True)[0]
                if len(domain_idx_list) != 1 :
                    tmp_x_domain = tmp_x[domain_idx_list]
                    tmp_centroid = torch.div(torch.sum(tmp_x_domain, dim = 0), len(domain_idx_list))
                    domain_centroid_stack.append(tmp_centroid)
            if len(domain_centroid_stack) == 0 :
                continue
            else :
                domain_centroid_stack = torch.stack(domain_centroid_stack)
            subtype_centroid = torch.mean(domain_centroid_stack, dim = 0)
            subtype_centroid_stack = []
            for i in range(len(domain_centroid_stack)) :
                subtype_centroid_stack.append(subtype_centroid)
            subtype_centroid_stack = torch.stack(subtype_centroid_stack)
            pdist_stack = nn.L1Loss()(subtype_centroid_stack, domain_centroid_stack)
            align_loss +=  torch.mean(pdist_stack, dim = 0)
        if align_loss == 0.0 :
            continue
        align_loss = align_loss / len(batch_subtype_list)
        fe_optimizer.zero_grad()
        c_optimizer.zero_grad()
        align_loss.backward()
        fe_optimizer.step() 
        c_optimizer.step()
    align_loss = align_loss.item()
    if epoch % 10 == 0 :
        print(f"[CA Epoch {epoch+1}] align loss: {align_loss:>5f}\n")


def ssl_train_classifier(epoch, source_dataloader, target_dataloader, fe_model, c_model, c_loss, fe_optimizer, c_optimizer) :
    source_size = len(source_dataloader.dataset)
    target_size = len(target_dataloader.dataset)
    #
    # 1. Obtain the pseudo-label for target dataset
    #
    target_pseudo_label = torch.empty((0), dtype = torch.int64)
    target_pseudo_label = target_pseudo_label.to(device)
    #
    for batch, (target_X, target_y) in enumerate(target_dataloader):
        target_X, target_y = target_X.to(device), target_y.to(device)
        target_X = target_X.float()
        extracted_feature = fe_model(target_X)
        batch_target_pred = c_model(extracted_feature)
        batch_pseudo_label = batch_target_pred.argmax(1)
        target_pseudo_label = torch.cat((target_pseudo_label, batch_pseudo_label), 0)
        if batch == 0 :
            target_loss = c_loss(batch_target_pred, target_y)
        else :
            target_loss = target_loss + c_loss(batch_target_pred, target_y)
    target_loss = target_loss / (batch + 1)
    #
    # Define alpha value
    alpha_f = 0.01
    t1 = 100
    t2 = 200
    if epoch < t1 :
        alpha = 0
    elif epoch < t2 :
        alpha = (epoch - t1) / (t2 - t1) * alpha_f
    else :
        alpha = alpha_f
    #
    # 2. Calculate the loss for the source dataset
    #
    correct = 0
    for batch, (source_X, source_y) in enumerate(source_dataloader):
        source_X, source_y = source_X.to(device), source_y.to(device)
        source_X = source_X.float()
        source_extracted_feature = fe_model(source_X)
        source_pred = c_model(source_extracted_feature)
        source_loss = c_loss(source_pred, source_y)
        ssl_loss = source_loss + alpha * target_loss
        # Backpropogation
        target_loss.detach_()
        fe_optimizer.zero_grad()
        c_optimizer.zero_grad()
        ssl_loss.backward() #retain_graph=True
        fe_optimizer.step()
        c_optimizer.step()
        correct += (source_pred.argmax(1) == source_y).type(torch.float).sum().item()
    ssl_loss = ssl_loss.item()
    source_loss = source_loss.item()
    target_loss = target_loss.item()
    correct /= source_size
    if epoch % 10 == 0 :
        print(f"[SSL Epoch {epoch+1}] alpha : {alpha:>3f}, SSL loss: {ssl_loss:>5f}, source loss: {source_loss:>5f}, target loss: {target_loss:>4f}, source ACC: {(100*correct):>0.2f}%\n")
    return target_pseudo_label


def adversarial_train_disc(epoch, dataloader, fe_model, d_model, domain_loss, fe_optimizer, d_optimizer) :
    size = len(dataloader.dataset)
    for batch, (X, y, z_subtype) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        X = X.float()
        extracted_feature = fe_model(X)
        pred = d_model(extracted_feature)
        d_loss = domain_loss(pred, y)
        # Backpropagation
        fe_optimizer.zero_grad()
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()
    d_loss = d_loss.item()
    if t % 10 == 0 :
        print(f"[AT Epoch {epoch+1}] Disc loss: {d_loss:>5f}", end = ", ")


def adversarial_train_fe(epoch, dataloader, fe_model, d_model, domain_loss, fe_optimizer, d_optimizer) :
    size = len(dataloader.dataset)
    for batch, (X, y, z_subtype) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        X = X.float()
        extracted_feature = fe_model(X)
        pred = d_model(extracted_feature)
        fake_y = torch.randint(low=0, high=num_domain, size = (len(y),))
        fake_y = fake_y.to(device)
        g_loss = domain_loss(pred, fake_y)
        # Backpropagation
        fe_optimizer.zero_grad()
        d_optimizer.zero_grad()
        g_loss.backward()
        fe_optimizer.step()
    g_loss = g_loss.item()
    if epoch % 10 == 0:
        print(f"Gen loss: {g_loss:>5f}")


def test_classifier(dataloader, fe_model, c_model, c_loss):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    fe_model.eval()
    c_model.eval()
    test_loss, test_acc = 0, 0
    pred_subtype_list = []
    label_subtype_list = []
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            X = X.float()
            extracted_feature = fe_model(X)
            pred = c_model(extracted_feature)
            test_loss += c_loss(pred, y).item()
            test_acc += (pred.argmax(1) == y).type(torch.float).sum().item()
            pred_subtype_list.append(pred.argmax(1))
            label_subtype_list.append(y)
    pred_subtype_list = torch.cat(pred_subtype_list, 0)
    label_subtype_list = torch.cat(label_subtype_list, 0)
    test_loss /= num_batches
    test_acc /= size
    print(f"\t\tTesting Accuracy: {(100*test_acc):>0.3f}%, Avg loss: {test_loss:>5f} \n")
    return pred_subtype_list, label_subtype_list, test_acc


def get_embed(dataloader, fe_model, c_model) :
    fe_model.eval()
    c_model.eval()
    X_embed_list = []
    y_list = []
    with torch.no_grad() :
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            X = X.float()
            X_embed = fe_model(X)
            X_embed_list.append(X_embed)
            y_list.append(y)
    X_embed_list = torch.cat(X_embed_list, 0)
    y_list = torch.cat(y_list, 0)
    return X_embed_list, y_list


def get_embed_domain(domain_dataloader, fe_model, c_model) :
    fe_model.eval()
    c_model.eval()
    X_embed_list = []
    y_list = []
    with torch.no_grad() :
        for batch, (X, y, z) in enumerate(domain_dataloader):
            X, y, z = X.to(device), y.to(device), z.to(device)
            X = X.float()
            X_embed = fe_model(X)
            X_embed_list.append(X_embed)
            y_list.append(y)
    X_embed_list = torch.cat(X_embed_list, 0)
    y_list = torch.cat(y_list, 0)
    return X_embed_list, y_list


pt_epochs = 500
ad_train_epochs = 500
ssl_train_epochs = 500
ft_epochs = 800
test_target_acc_ft = 0.0


# 1. Pre-training
for t in range(pt_epochs):
    pretrain_classifier(t, train_dataloader, feature_extract_model, subtype_pred_model, c_loss, fe_optimizer, c_optimizer)

# 2. Adversarial training
for t in range(ad_train_epochs):
    adversarial_train_disc(t, domain_dataloader, feature_extract_model, domain_disc_model, domain_loss, fe_optimizer, d_optimizer)
    adversarial_train_fe(t, domain_dataloader, feature_extract_model, domain_disc_model, domain_loss, fe_optimizer, d_optimizer)

# 3.  Fine-tuning
for t in range(ssl_train_epochs) :
    target_pseudo_label = ssl_train_classifier(t, train_dataloader, target_dataloader, feature_extract_model, subtype_pred_model, c_loss, fe_optimizer, c_optimizer)
    target_dataset = MyBaseDataset(target_x, target_pseudo_label)
    target_dataloader = DataLoader(target_dataset, batch_size = target_batch_size)

for t in range(ft_epochs) :
    # SSL
    target_pseudo_label = ssl_train_classifier(t, train_dataloader, target_dataloader, feature_extract_model, subtype_pred_model, c_loss, fe_optimizer, c_optimizer)
    target_dataset = MyBaseDataset(target_x, target_pseudo_label)
    target_dataloader = DataLoader(target_dataset, batch_size = target_batch_size)

    # CA
    target_pseudo_label = target_pseudo_label.to("cpu")
    domain_z = torch.cat((y_train, target_pseudo_label), 0)
    domain_dataset = DomainDataset(domain_x, domain_y, domain_z)
    domain_dataloader = DataLoader(domain_dataset, batch_size = batch_size, shuffle = True)
    class_alignment_train(t, domain_dataloader, feature_extract_model, fe_optimizer, c_optimizer)
    

    # Test
    if t % 10 == 0 :
        test_target_pred, test_target_label, test_acc = test_classifier(test_target_dataloader, feature_extract_model, subtype_pred_model, c_loss)
        if test_acc > test_target_acc_ft :
            test_target_acc_ft = test_acc
            test_target_pred_ft = test_target_pred
            test_target_label_ft = test_target_label


test_target_pred_ft = test_target_pred_ft.detach().cpu().numpy()
test_target_label_ft = test_target_label_ft.detach().cpu().numpy()
np.savetxt(os.path.join(result_dir, "ft_test_target_pred.csv"), test_target_pred_ft, fmt="%.0f", delimiter=",")
np.savetxt(os.path.join(result_dir, "ft_test_target_label.csv"), test_target_label_ft, fmt="%.0f", delimiter=",")

