from PIL import Image as Img
import numpy as np

f = open("classes.txt","r")
classes = f.readlines()
f.close()
classes = [c.replace('\n','').replace(' ','_') for c in classes]

print classes

file_=open('data/'+classes[3]+'.npy','rb')
data=np.load(file_)
file_.close()
img=data[0]

for x in range(28):
    for y in range(28):
        print '%4d'%img[x][y], 
    print

pic=Img.fromarray(img)
pic.save('tmp.jpg')
