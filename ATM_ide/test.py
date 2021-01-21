import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

import os, pdb
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import scipy.io
import soundfile as sf
import time  
import numpy as np
import numpy.matlib
import random
import soundfile as sf
from tqdm import tqdm

from model import ATM_ide
from scipy.io import wavfile

from utils import creatdir, get_filepaths, SPP_make, SPP_rec, read_wav

batch_size = 100

######################### Testing data #########################

Test_Noisy_paths = get_filepaths("", '.wav')# your testing noisy data path here
Test_Clean_paths = get_filepaths("", '.wav')# your testing clean data path here
                
Num_testdata=len(Test_Noisy_paths)   

device = torch.device('cuda')
model = ATM_ide()
model.load_state_dict(torch.load(''))# model path here
model.to(device)
model.eval()

save_dir = ""
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

SNRs = ['SNR_-5', 'SNR_0', 'SNR_5']# your noisy SNR here
Noises = ['engine', 'PINKNOISE_16k', 'new_street', 'white']# your noise type here
for N in Noises:
    for SNR in SNRs:
        if not os.path.exists(os.path.join(save_dir, N, SNR)):
            os.makedirs(os.path.join(save_dir, N, SNR))

#dense3_out = torch.zeros((999,32),dtype=torch.float32)
print ('testing...')
start_time = time.time()
for path in tqdm(Test_Noisy_paths):
    
    #### Process your path name ####  
    # Our noisy path expample : /test/SNR_5/white/file_01.wav 
    S=path.split('/')
    dB=S[-3]
    wave_name=S[-1]
    noise_name=S[-2]
    ################################

    noisy = read_wav(path)
    n_sp, n_p = SPP_make(noisy, Noisy=True)
    
    enhanced_LP, _, _ = model(torch.from_numpy(n_sp.astype('float32')).to(device))

    enhanced_LP = enhanced_LP.cpu().detach().numpy().squeeze()
    enhanced_wav = SPP_rec(np.exp(enhanced_LP.T) - 1,n_p)
    enhanced_wav=enhanced_wav/np.max(abs(enhanced_wav))
    
    sf.write(os.path.join(save_dir,noise_name,dB,wave_name), enhanced_wav, 16000)


end_time = time.time()
print ('The testing for this file ran for %.2fm' % ((end_time - start_time) / 60.))
