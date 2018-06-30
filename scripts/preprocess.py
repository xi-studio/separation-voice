import numpy as np
import h5py
import glob
import librosa

from scipy.io import wavfile

fs = 22050
scale = np.power(2,16)

def pic(x,x1):
    k = x.shape(0) // scale
    a = x[:k*scale]
    b = x1[:k*scale]

def savefile(a,b,num):
    filename = '/data1/littletree/sepvoice/%d.h5' % num
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
        f1   = f.replace('Mixtures','Sources').replace('mixture','vocals') 
        x,_  = librosa.load(f, sr=fs, mono=False)
        x1,_ = librosa.load(f1, sr=fs, mono=False)
        k = x.shape[1] // scale

        a = x[:,:k*scale]
        b = x1[:,:k*scale]
        a = a.reshape((2,k,scale))
        b = b.reshape((2,k,scale))
        for i in range(k):
            if np.sum(a[:,i]) !=0:
                savefile(a[:,i],b[:,i],num)
                num += 1


if __name__ == "__main__":
    main()


