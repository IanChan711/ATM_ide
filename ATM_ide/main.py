import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
import pandas as pd
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transform
from torchvision.utils import save_image
import os
from torch.utils.data import Dataset
from tqdm import tqdm

from utils import creatdir, get_filepaths, SPP_make, SPP_rec, read_wav
from Model import ATM_ide

os.environ["CUDA_VISIBLE_DEVICES"]="3" #Your GPU number, default = 0

random.seed(999)

batch_size= 1
Epoch = 20
############################################################

def data_generator(noisy_list, clean_path, index, label, shuffle="True"):

    
    # Load your own data here
    noisy = read_wav(noisy_list[index])
    S = noisy_list[index].split('/')[-1]
    a = noisy_list[index].split('/')[-1] + '_vad'
    clean = read_wav(clean_path+S)
    frames_label = label[label['cleanID'].isin([a])]
    frames_label = frames_label.iloc[0,1:]
    frames_label = np.array(frames_label, dtype='float32') 
    
    n_sp, n_p = SPP_make(noisy, Noisy=True)
    c_sp, c_p = SPP_make(clean, Noisy=False)

    return n_sp, c_sp, frames_label

def valid_generator(noisy_list, clean_path, index, label, shuffle="True"):
    
    # Load your own data here
    noisy = read_wav(noisy_list[index])
    S = noisy_list[index].split('/')[-1]
    a = noisy_list[index].split('/')[-1] + '_vad'
    clean = read_wav(clean_path+S)
    frames_label = label[label['cleanID'].isin([a])]
    frames_label = frames_label.iloc[0,1:]
    frames_label = np.array(frames_label, dtype='float32') 
    
    n_sp, n_p = SPP_make(noisy, Noisy=True)
    c_sp, c_p = SPP_make(clean, Noisy=False)

    return n_sp, c_sp, frames_label


##############################################################

# Load your own data
clean_path = ""
speaker_label = pd.read_csv('')

noisy_list = np.load('')
noisy_list = noisy_list.tolist()

idx = int(len(noisy_list)*0.95)
Train_list = noisy_list[0:idx]
Num_traindata = len(Train_list)
Valid_list = noisy_list[idx:]

steps_per_epoch = (Num_traindata)//batch_size
Num_testdata=len(Valid_list)

save_dir = ""
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

######################### Training Stage ########################           
start_time = time.time()

print('model building...')
def initialize_weights(model):
    if type(model) in [nn.Linear]:
        nn.init.xavier_uniform_(model.weight.data)

enh_model = ATM_ide().cuda()
enh_model.apply(initialize_weights)

log_std_a = torch.zeros((1,), requires_grad=True, device = "cuda")
log_std_b = torch.zeros((1,), requires_grad=True, device = "cuda")

criterion_enh = nn.MSELoss()
criterion_speaker = nn.CrossEntropyLoss()

optimizer_enh = torch.optim.Adam(([p for p in enh_model.parameters()] + [log_std_a] + [log_std_b]), lr=0.0001)

print('training...')

loss_enh = []
loss_speaker = []

loss_enh_val_avg = []
loss_speaker_val_avg = []
acc = []
Acc_epoch = []

for epoch in range(Epoch):

    random.shuffle(Train_list)
    loss_enh_avg = []
    loss_speaker_avg = []

    for iter in tqdm(range(len(Train_list))):

        n_sp , c_sp , frames_label = data_generator(Train_list, clean_path, iter, speaker_label)

        enh_model.train()
        optimizer_enh.zero_grad()

        enh_out, speaker_out = enh_model(torch.from_numpy(n_sp.astype('float32')).cuda())
        
        loss_s = criterion_speaker(speaker_out,torch.from_numpy(frames_label).type(torch.LongTensor).cuda())

        loss_speaker.append(loss_s.cpu().item())
        loss_speaker_avg.append(loss_s.cpu().item())

        loss_e = criterion_enh(enh_out,torch.squeeze(torch.from_numpy(c_sp.astype('float32'))).cuda())
        
        loss_enh.append(loss_e.cpu().item())
        loss_enh_avg.append(loss_e.cpu().item())

        var_1 = torch.exp(log_std_a)**2
        var_2 = torch.exp(log_std_b)**2
        loss = (1/(2*var_1)) * loss_e + (1/var_2) * loss_s + log_std_a + log_std_b

        loss.backward()
        optimizer_enh.step()

    loss_enh_avg = sum(loss_enh_avg)/len(loss_enh_avg)
    loss_speaker_avg = sum(loss_speaker_avg)/len(loss_speaker_avg)
    print("Epoch(%d/%d): enh_loss %f spk_loss %f" % (epoch+1, Epoch, loss_enh_avg,loss_speaker_avg))

            

    with torch.no_grad():
        loss_enh_val = []
        loss_speaker_val = []
        enh_model.eval()
        for iter in range(len(Valid_list)):
            n_sp , c_sp , frames_label = valid_generator(Valid_list, clean_path, iter, speaker_label)

            enh_out, speaker_out = enh_model(torch.from_numpy(n_sp.astype('float32')).cuda())

            loss_s = criterion_speaker(speaker_out,torch.from_numpy(frames_label).type(torch.LongTensor).cuda())
            loss_speaker_val.append(loss_s.cpu().item())

            loss_e = criterion_enh(enh_out,torch.squeeze(torch.from_numpy(c_sp)).cuda())
            loss_enh_val.append(loss_e.cpu().item())

            running_acc_initial = 0.0
            prediction = torch.argmax(speaker_out, dim=1)
            running_acc_initial += torch.sum(prediction == torch.from_numpy(frames_label).cuda())
            acc.append((running_acc_initial/len(frames_label)))

        loss_enh_val_avg.append(sum(loss_enh_val)/len(loss_enh_val))
        loss_speaker_val_avg.append(sum(loss_speaker_val)/len(loss_speaker_val))
        Acc_epoch.append((sum(acc)/len(acc)).item())
        print("Epoch(%d/%d): val_loss %f" % (epoch+1, Epoch, sum(loss_enh_val)/len(loss_enh_val)))
        print("Acc : %f" %((sum(acc)/len(acc))))

    Path = 'LSTM_EPOCH_' + str(epoch) + '.pth'
    torch.save(enh_model.state_dict(),'./1104Enhanced_result_LSTM2_mask/ENHANCE_'+Path)

np.save(os.path.join(save_dir, 'loss_enh.npy'), loss_enh)
np.save(os.path.join(save_dir, 'loss_speaker.npy'), loss_speaker)
np.save(os.path.join(save_dir, 'acc_val.npy'), Acc_epoch)

for i in range(len(loss_speaker_val_avg)):
    print("Epoch:%d enh_val_avg:%f speaker_val_avg:%f \n"%(i,loss_enh_val_avg[i],loss_speaker_val_avg[i]))

        

        





        
        


