import numpy as np
import h5py
import glob

from scipy.io import wavfile


scale = np.power(2,12)

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
     
    #print(np.sum(base))
    return base

    
    
def main():
    res = glob.glob("/data1/littletree/DSD100/Mixtures/*/*/*.wav")
    num = 0
    for f in res:
        print(f)
        f1 = f.replace('Mixtures','Sources').replace('mixture','vocals') 
        fs,x = wavfile.read(f)
        fs1,x1 = wavfile.read(f1)
        k = x.shape[0] // scale
        a = x[:k*scale]
        b = x1[:k*scale]
        a = a.reshape((k,scale,2))
        b = b.reshape((k,scale,2))
        for i in range(k):
            if np.sum(a[i]) !=0:
                savefile(tobit(a[i,:,0]),tobit(b[i,:,0]),num)
                num += 1
                savefile(tobit(a[i,:,1]),tobit(b[i,:,1]),num)
                num += 1


if __name__ == "__main__":
    main()


