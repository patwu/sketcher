from PIL import Image as Img
import numpy as np

from cnn import CNN as CNN

def rot90(img,n):
    rot_img=np.zeros((28,28))
    for t in range(n):
        for i in range(28):
            for j in range(28):
                rot_img[27-j][i]=img[i][j]
        img=rot_img
    return rot_img

def flip(img):
    flip_img=np.zeros((28,28))
    for i in range(28):
        for j in range(28):
            flip_img[i][27-j]=img[i][j]
    return flip_img

def crop(img,x,y):
    crop_img=np.zeros((28,28))
    for i in range(28):
        for j in range(28):
            if 0<i+x<28 and 0<j+y<28:
                crop_img[i,j]=img[i+x,j+y]
    return crop_img

def save(img,name='tmp.jpg'):
    dd=np.asarray(img*255,dtype='uint8')
    pic=Img.fromarray(dd)
    pic.save(name)

n_class=30
max_step=5000
batch_size=32
n_sample_in_train=1000
n_sample_in_test=50
n_channel=16

f = open("classes.txt","r")
classes = f.readlines()
f.close()
classes = [c.replace('\n','').replace(' ','_') for c in classes]
classes = classes[:n_class]
train_data=[]
test_data=[]
for i,c in enumerate(classes):
    filename='data/%s.npy'%c
    file_=open(filename,'rb')
    data_bundle=np.load(file_)
    for d in data_bundle[-n_sample_in_test:]:
        test_data.append((d/255.,i))
    for d in data_bundle[:n_sample_in_train]:
        train_data.append((d/255.,i))

model=CNN(28,28,n_channel=n_channel,n_output=n_class)
all_idx=range(len(train_data))
for n_step in range(max_step):
    batch_x=[]
    batch_y=[]
    idx=np.random.choice(all_idx,batch_size)
    for i in idx:
        #add your data augmentation to train_data[i][0] here
        img=train_data[i][0]
        if True or np.random.randint(2)==0:
            img=flip(img)
        img=crop(img,np.random.randint(10)-5,np.random.randint(10)-5)
        batch_x.append(img)
        batch_y.append(train_data[i][1])
    
    _,loss=model.train(batch_x,batch_y)   
    if n_step%100==0:
        acc=0.
        for i in range(len(test_data)/batch_size):
            batch=test_data[i*batch_size:(i+1)*batch_size]            
            batch_x=[x for x,_ in batch]
            batch_y=[y for _,y in batch]
            pred,prob=model.predict(batch_x)
            for p,g in zip(pred,batch_y):
                if p==g:
                    acc+=1
        print n_step,loss,acc/len(test_data)

model.save('tfmodel')

