from PIL import Image
import numpy as np

from cnn import CNN as CNN

f = open("classes.txt","r")
# And for reading use
classes = f.readlines()
f.close()
classes = [c.replace('\n','').replace(' ','_') for c in classes]
classes=classes[:2]
data=[]
for i,c in enumerate(classes):
    filename='data/%s.npy'%c
    file_=open(filename,'rb')
    data_bundle=np.load(file_)
    for d in data_bundle:
        data.append((d/255.,i))

model=CNN(28,28,n_output=2)
all_idx=range(len(data))
for n_step in range(100):
    batch_x=[]
    batch_y=[]
    idx=np.random.choice(all_idx,32)
    for i in idx:
        batch_x.append(data[i][0])
        batch_y.append(data[i][1])
    
    pred,loss=model.train(batch_x,batch_y)   
    if n_step%100==0:
        acc=0.
        for i in range(32):
            if pred[i]==batch_y[i]:
                acc+=1
        print n_step,loss,acc/32
model.save('tfmodel')

