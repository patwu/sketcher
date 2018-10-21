from PIL import Image
import numpy as np

from cnn import CNN as CNN

f = open("classes.txt","r")
classes = f.readlines()
f.close()
classes = [c.replace('\n','').replace(' ','_') for c in classes]
train_data=[]
test_data=[]
for i,c in enumerate(classes):
    filename='data/%s.npy'%c
    file_=open(filename,'rb')
    data_bundle=np.load(file_)
    for d in data_bundle[-20:]:
        test_data.append((d/255.,i))
    for d in data_bundle[:100]:
        train_data.append((d/255.,i))

model=CNN(28,28,n_output=80)
all_idx=range(len(train_data))
for n_step in range(5000):
    batch_x=[]
    batch_y=[]
    idx=np.random.choice(all_idx,32)
    for i in idx:
        batch_x.append(train_data[i][0])
        batch_y.append(train_data[i][1])
    
    _,loss=model.train(batch_x,batch_y)   
    if n_step%100==0:
        acc=0.
        for i in range(len(test_data)/32):
            batch=test_data[i*32:(i+1)*32]            
            batch_x=[x for x,_ in batch]
            batch_y=[y for _,y in batch]
            pred,prob=model.predict(batch_x)
            for p,g in zip(pred,batch_y):
                if p==g:
                    acc+=1
        print n_step,loss,acc/len(test_data)

model.save('tfmodel')

