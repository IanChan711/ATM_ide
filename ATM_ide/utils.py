import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt, librosa, numpy as np, sys, os, scipy
from scipy.io import wavfile

def creatdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_filepaths(directory, data_format):

    file_paths = []  # List which will store all of the full filepaths.

    # Walk the tree.
    for root, directories, files in os.walk(directory):
        for filename in files:
            # Join the two strings in order to form the full filepath.
            filepath = os.path.join(root, filename)
            #pdb.set_trace()
            if filepath.split('/')[-1][-4:] == data_format: #and filepath.split('/')[-2] != '-10db':
                file_paths.append(filepath)  # Add it to the list.
    return file_paths  # Self-explanatory.

def read_wav(path):

    rate, sig = wavfile.read(path)
    sig = sig/np.max(abs(sig))
    #pdb.set_trace()
    sig=sig.astype('float')
    if len(sig.shape)==2:
        sig=(sig[:,0]+sig[:,1])/2
    sig=sig/np.max(abs(sig))
    length=(len(sig)-512)//256
    sig=sig[0:512+length*256]
    #pb.set_trace()
    return sig

def SPP_make(y, Noisy=False):

    F = librosa.stft(y,n_fft=512,hop_length=256,win_length=512,window=scipy.signal.hamming,center=False)
    phase=np.angle(F)
    #Lp = np.log10(np.abs(F)**2+1e-12)
    Lp = np.log1p(np.abs(F)+1e-12)
    
    if Noisy==True:
        meanR = np.mean(Lp, axis=1).reshape((257,1))
        stdR = np.std(Lp, axis=1).reshape((257,1))+1e-12
        NLp = (Lp-meanR)/stdR
        #print(np.mean(NLp))
    else:
        NLp=Lp

    NLp=np.reshape(NLp.T, (1, NLp.shape[1], 257))
    return NLp, phase

def SPP_rec(mag,phase):
    Rec = np.multiply(mag , np.expm1(1j*phase))
    result = librosa.istft(Rec,
                           hop_length=256,
                           win_length=512,
                           window=scipy.signal.hamming,
                           center=False)
    return result

if __name__=="__main__":
    path=sys.argv[1]
    
    print("done")


