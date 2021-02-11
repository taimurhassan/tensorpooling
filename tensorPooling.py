import cv2
import numpy as np
import os
from scipy import signal
from scipy import misc
from sklearn.preprocessing import MinMaxScaler
from anisotropicDiffusionFilter import anisodiff

class tensor_dict(dict): 
  
    # __init__ function 
    def __init__(self): 
        self = dict() 
          
    # Function to add key:value 
    def add(self, key, value): 
        if key in self:
            self[key].append(value)
        else:
            self[key] = list()
            self[key].append(value) 
        #print(self[key].shape)
        
def scale_feature(scaler,feature):
    scaler.fit(feature)
    return scaler.transform(feature)


def remove_center_line(img,threshold):
    rows,cols,_ = img.shape
    crow,ccol = rows//2 , cols//2
    mw = 10
    center_reg = img[:,ccol-mw:ccol+mw,:]

    idx = np.where(np.all(center_reg>threshold,axis=2))

    sorted_cr = np.sort(center_reg[idx[0]],axis=1)
    if(center_reg.shape[1]%2==0):
        median_vals = np.mean(sorted_cr[:,mw:mw+2],axis=1)
    else:
        median_vals = sorted_cr[:,center_reg.shape[1]//2]

    center_reg[idx] = median_vals
    img[:,ccol-mw:ccol+mw,:] = center_reg
    return img

def set_white_roc(img,roc,threshold):
    (c1,r1),(c2,r2) = roc
    reg = img[r1:r2,c1:c2,:]
    idx = np.where(np.all(reg>threshold,axis=2))
    reg[idx[0],idx[1],:] = 0
    img[r1:r2,c1:c2,:] = reg
    return img

def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result
    
def structureTensor(I,N):
    si = 1
    so = 1
    tensors = tensor_dict()
    gradients = []
    
    a = np.zeros((5,5))
    b = np.zeros((5,5))
    
    #print(a)
    I = I.astype(np.float64)
    m,n = I.shape

    x = np.arange(-2*si,2*si+1,1)
    g  = np.exp(-0.5*(x/si)**2);
    g  = g/sum(g);
    gd = -x*g/(si**2);
    tg = np.transpose(g)
    tgd = np.transpose(gd)

    a[3][:] = gd;
    b[:][3] = g;
    
    an = a
    index = 0
    
    for i in range(0,N):
        angle = (2*np.pi*i)/N
        a = rotate_image(an,angle)
        Ig = signal.convolve2d(signal.convolve2d(I, a, boundary='symm', mode='same'),np.transpose(b), boundary='symm', mode='same') #np.apply_along_axis(np.convolve, 0, np.apply_along_axis(np.convolve, 1, I, a, 'same'), np.transpose(b), 'same')
        gradients.append(Ig)
    

    nGradients = len(gradients)
    #print(nGradients)
    for i in range(0,nGradients):
        I1 = gradients[i]
        for j in range(0,nGradients):
            I2 = gradients[j]
            Ixy = I1 * I2 #np.dot(I1, I2)
            x  = np.arange(-2*so,2*so+1,1)
            g  = np.exp(-0.5*(x/so)**2);
            Sxy = anisodiff(Ixy)
            #print(Sxy.shape)
            tensors.add(i, Sxy)
            #print(tensors)
        
    return tensors
    
def getCoherentOne(tensors):
    cTensor = np.array([]);
    cTensor2 = np.array([]);
    
    r = len(tensors)
    m = -100000000000000000
    for i in range(0,r):
        c = tensors[i]
        #print(len(c))
        for j in range(0,len(c)):
            if c[j] is None:
                continue
            t = np.array(c[j])
            #print(t.shape)
            u, s, v = np.linalg.svd(t, full_matrices=False)
            #print(s.shape)
            if m < np.amax(np.amax(s)):
                cTensor2 = cTensor
                cTensor = t
                m = np.amax(np.amax(s))
                #print(m)
    
    #print(len(cTensor))
    return cTensor, cTensor2
    
def structureTensor2(I,si,so):
    I = I.astype(np.float64)
    m,n = I.shape

    x = np.arange(-2*si,2*si+1,1)
    g  = np.exp(-0.5*(x/si)**2);
    g  = g/sum(g);
    gd = -x*g/(si**2);
    tg = np.transpose(g)
    tgd = np.transpose(gd)

    tempy = np.apply_along_axis(np.convolve, 1, I, gd, 'same')

    tempx = np.apply_along_axis(np.convolve, 0, I, tgd, 'same')


    Iy = np.apply_along_axis(np.convolve, 0, tempy, tg, 'same')
    Ix = np.apply_along_axis(np.convolve, 1, tempx, g, 'same')


    Iyy = Iy**2;
    Ixx = Ix**2;

    x  = np.arange(-2*so,2*so+1,1)
    g  = np.exp(-0.5*(x/so)**2);
    tg = np.transpose(g)

    tempsyy = np.apply_along_axis(np.convolve, 1, Iyy, g, 'same')
    Syy = np.apply_along_axis(np.convolve, 0, tempsyy, tg, 'same')

    tempsxx = np.apply_along_axis(np.convolve, 1, Ixx, g, 'same')
    Sxx = np.apply_along_axis(np.convolve, 0, tempsxx, tg, 'same')


    return Sxx,Syy

    # Iy = signal.convolve2d(,g,'same',axis=1);

def heatmap(I, colormap=cv2.COLORMAP_JET):
    
    im = cv2.applyColorMap((I).astype(np.uint8), colormap)

    return (im).astype(np.uint8) 

folder = 'testingDataset/original/' # update these paths as per your dataset
folder2 = 'testingDataset/test_images/'

for filename in os.listdir(folder):
    img = cv2.imread(os.path.join(folder,filename))

    row,col,_ = img.shape
    
    if len(img.shape) is 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    N = 4
    n = 3
    i1 = img
    p = [];
    s = [];
    
    i2 = np.zeros(i1.shape)
    
    for j in range(0,n):
        i1 = cv2.pyrDown(i1)
        i3 = i1
        
        tensors = structureTensor(i1,N)
        c1, c2 = getCoherentOne(tensors)
        
        s1 = c1
        
        if c2.size != 0:
            s1 = c1 + c2
        
        s.append(s1)
        #print(s1.shape)
        #cv2.imshow('',s1)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        if j > 1:
            s4 = cv2.resize(s[j-1], (i1.shape), interpolation = cv2.INTER_LINEAR)
            i1 = np.transpose(i1.astype(np.float64)) * s4 #np.dot(i1.astype(np.float64), s4)
        else:
            i1 = cv2.resize(i1, (s1.shape), interpolation = cv2.INTER_LINEAR)
            i1 = np.transpose(i1.astype(np.float64)) * s1 #np.dot(i1.astype(np.float64), s1)
            
        p.append(i1)
        i1 = cv2.resize(i1, (i2.shape), interpolation = cv2.INTER_LINEAR)
        i2 = i2 + np.transpose(i1.astype(np.float64))
        i1 = i3;
        
        cv2.imwrite(os.path.join(folder2,filename),i2,[cv2.IMWRITE_PNG_COMPRESSION,0])
        

print('Done!')
# cv2.waitKey(0)
cv2.destroyAllWindows()
