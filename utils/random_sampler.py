import numpy as np
import cv2

def sample_random_number_mnist(x_train, y_train,value):
    '''
    #returns a random MNIST sample of a certain value [0...9]
    '''
    temp=np.where(y_train==value)
    numbersamplelen=len(temp[0])
    randominst=np.random.randint(numbersamplelen)
    return x_train[temp[randominst]]

def find_nearest_white(img, target):
    '''
    #finds the nearest nonzero value of a 2d image (needed for trajectory matching)
    '''
    nonzero = cv2.findNonZero(img)
    distances = np.sqrt((nonzero[:,:,0] - target[0]) ** 2 + (nonzero[:,:,1] - target[1]) ** 2)
    nearest_index = np.argmin(distances)
    return nonzero[nearest_index]

def generate_from_mnist_images(shape,shapelabel, brush,brushlabel, traces):
    from skimage.morphology import skeletonize, thin
    shapetrace=cv2.resize(shape,(224,224))
    shapetrace[np.where(shapetrace!=0)]=1
    shapetrace = skeletonize(shapetrace)
    shapetrace=shapetrace*1
    shapetrace=shapetrace.astype('uint8')
    newtrace=np.zeros([2,150])
    for j in range(150):
        newcoord=find_nearest_white(shapetrace,[int(traces[shapelabel,1,j]),int(traces[shapelabel,0,j])])
        newtrace[:,j]=[newcoord[0][1],newcoord[0][0]]
    newtrace=newtrace.astype('uint8')
    sample=np.zeros([150,224,224])

    for k in range(150):
        coords=[max(0,newtrace[0,k]-14),min(224,newtrace[0,k]+14),max(0,newtrace[1,k]-14),min(224,newtrace[1,k]+14)]
        sample[k,coords[0]:coords[1],coords[2]:coords[3]]=brush[0:coords[1]-coords[0],0:coords[3]-coords[2]]
    return sample
