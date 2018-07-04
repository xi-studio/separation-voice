import numpy as np
import h5py
import glob
import librosa

from scipy.io import wavfile

fs = 22050
scale = np.power(2,16)


def savefile(a,num):
    filename = '../data/voice/%d.h5' % num
    f = h5py.File(filename,'w')
    f['in']  = a
    f.close()
    
def main():
    res = glob.glob("../data/*.wav")
    num = 0
    for f in res:
        print(f)
        _, x = wavfile.read(f)
        k = x.shape[0] // scale
        print(k)

        print(x.shape)
        a = x[:k*scale,:]
        print(a.shape)
        a = a.reshape((k,scale,2))
        a = a.transpose(2,0,1)
        for i in range(k):
            if np.sum(a[:,i]) !=0:
                savefile(a[:,i],num)
                num += 1


if __name__ == "__main__":
    main()


