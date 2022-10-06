#!/usr/bin/env python
# coding: utf-8

# In[17]:


import numpy as np 
import pandas as pd 
import cv2
import os 
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import img_to_array


# In[3]:


imgpath = "C:/Users/ajjup/OneDrive/Desktop/Malaria/"


# In[4]:


print(imgpath)


# In[5]:


imgnames = os.listdir(imgpath+'Parasitized/')


# In[6]:


imgnames


# In[7]:


len(imgnames)


# In[20]:


dataset = []
labels = []


# In[22]:


len(dataset)


# In[23]:


len(labels)


# In[8]:


imgnames[0]


# In[19]:


import time 
# for img in imgnames:
#     pic = cv2.imread(imgpath +'Parasitized/'+ img)
#     print(pic.shape)
#     print(pic.dtype)
#     pic = cv2.resize(pic, (64,64))
#     pic = img_to_array(pic)
#     print()
#     print(pic.shape)
#     print(pic.dtype)
#     time.sleep(4)
#     print("******************************")
    


# In[24]:


for img in imgnames:
    pic = cv2.imread(imgpath +'Parasitized/'+ img)
    pic = cv2.resize(pic, (64,64))
    pic = img_to_array(pic)
    dataset.append(pic)
    labels.append(1)

print("Done with the Uploading Process")


# In[25]:


print(len(dataset))


# In[26]:


print(len(labels))


# ###   unloading uninfected data

# In[33]:


imgnames = os.listdir(imgpath+'Uninfected/')


# In[34]:


len(imgnames)


# In[35]:


imgnames[0]


# In[36]:


for img in imgnames:
    pic = cv2.imread(imgpath +'Uninfected/'+ img)
    pic = cv2.resize(pic, (64,64))
    pic = img_to_array(pic)
    dataset.append(pic)
    labels.append(0)

print("Done with the Uploading Process")


# In[37]:


print(len(dataset))


# In[38]:


print(len(labels))


# In[39]:


pd.DataFrame(labels)[0].unique()


# In[41]:


dataset[0].max()


# In[43]:


dataset[0].min()


# In[42]:


dataset[0]


# In[44]:


dataset = np.array(dataset,  dtype = 'float')/255.0


# In[45]:


dataset


# In[47]:


dataset[0].max()


# In[48]:


labels  = np.array(labels)


# In[55]:


plt.imshow(dataset[350])


# In[54]:


plt.imshow(dataset[3500])


# ###  Splitting of data 

# In[65]:


from sklearn.model_selection import train_test_split


# In[67]:


Xtrain, Xtest, ytrain, ytest = train_test_split(dataset, labels, test_size=0.25)


# In[69]:


Xtrain.shape, Xtest.shape, dataset.shape


# ###  Model Preparation 

# In[49]:


from tensorflow.keras.models import Sequential


# In[51]:


from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, BatchNormalization, Flatten, Dense 


# In[52]:


conmodel = Sequential()


# ####  First Convolutional Layer

# In[56]:


conmodel.add(Conv2D(32, (3,3),activation = 'relu', padding = 'same', 
                    input_shape = (64,64,3)))

conmodel.add(MaxPooling2D(pool_size =(2,2)))
conmodel.add(BatchNormalization(axis = -1))
conmodel.add(Dropout(0.20))


# ####  Second Convolutional Layer

# In[57]:


conmodel.add(Conv2D(64, (3,3),activation = 'relu', padding = 'same'))

conmodel.add(MaxPooling2D(pool_size =(2,2)))
conmodel.add(BatchNormalization(axis = -1))
conmodel.add(Dropout(0.20))


# In[58]:


conmodel.add(Flatten())


# In[60]:


conmodel.add(Dense(512, activation = 'relu', ))
conmodel.add(BatchNormalization())
conmodel.add(Dropout(0.20))


# In[61]:


conmodel.add(Dense(256, activation = 'relu', ))
conmodel.add(BatchNormalization())
conmodel.add(Dropout(0.20))


# In[62]:


conmodel.add(Dense(2, activation = 'softmax'))


# In[63]:


conmodel.summary()


# In[64]:


conmodel.compile(optimizer = 'Adam', loss =  "sparse_categorical_crossentropy",
                metrics = ['accuracy'])


# In[71]:


conmodel.fit(Xtrain, ytrain, epochs=50, verbose= 2)


# In[ ]:




