import numpy as np
import h5py
import glob
import librosa
import time

from scipy.io import wavfile

fs = 22050
fs = 44100
scale = np.power(2,16)

def pic(x,x1):
    k = x.shape(0) // scale
    a = x[:k*scale]
    b = x1[:k*scale]

def savefile(a,b,num):
    filename = '/data1/littletree/sepvoice_w/%d.h5' % num
    f = h5py.File(filename,'w')
    f['in']  = a
    f['out'] = b
    f.close()

def tobit(a):
    base = np.zeros((17,4096),dtype=np.uint8)
    base[0] = (a > 0) * 1
    s = a
    for x in range(16): 
        base[x+1] = s % 2
        s = s // 2
     
    return base

    
    
def main():
    res = glob.glob("/data1/littletree/DSD100/Mixtures/*/*/*.wav")
    num = 0
    for f in res:
        print(f)
        now = time.time()
        f1   = f.replace('Mixtures','Sources').replace('mixture','vocals') 
        m    = wavfile.read(f)
        m1   = wavfile.read(f1)
        print('wav:',time.time() - now)
        now = time.time()
        x,_  = librosa.load(f, sr=fs, mono=False)
        x1,_ = librosa.load(f1, sr=fs, mono=False)
        print(x.shape)
        print(m[1].shape,m[0])
        print(np.sum(np.abs(x.T - m[1]/32768.0)))
        print('librosa:',time.time() - now)
        k = x.shape[1] // scale

        a = x[:,:k*scale]
        b = x1[:,:k*scale]
        #savefile(a,b,num)
        num += 1


if __name__ == "__main__":
    main()


