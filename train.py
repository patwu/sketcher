from PIL import Image
import numpy as np

from cnn import CNN as CNN

n_class=30
max_step=5000
batch_size=32
n_sample_in_train=1000
n_sample_in_test=50

f = open("classes.txt","r")
classes = f.readlines()
f.close()
classes = [c.replace('\n','').replace(' ','_') for c in classes]
classes = classes[:30]
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

model=CNN(28,28,n_output=n_class)
all_idx=range(len(train_data))
for n_step in range(max_step):
    batch_x=[]
    batch_y=[]
    idx=np.random.choice(all_idx,batch_size)
    for i in idx:
        batch_x.append(train_data[i][0])
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

