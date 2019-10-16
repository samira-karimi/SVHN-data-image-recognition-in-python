#!/usr/bin/env python
# coding: utf-8

# In[21]:


from __future__ import absolute_import, division, print_function, unicode_literals
import scipy.io as sio # For loading matlab files
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
plt.rcParams['figure.figsize'] = (32.0, 8.0)
headertr, versiontr, globstr, xtr, ytr= sio.loadmat('train_32x32.mat')
TrainSet = sio.loadmat('train_32x32.mat')
train_y = TrainSet["y"]
train_x = TrainSet["X"]
type(train_x)


# In[22]:


TestSet = sio.loadmat('test_32x32.mat')
test_x, test_y = TestSet["X"],TestSet["y"]
print("Train set X and y shapes are ", train_x.shape , train_y.shape)
print("Test set X and y shapes are ", test_x.shape , test_y.shape)


# In[23]:


# Transpose x arrays
train_x, train_y = train_x.transpose((3,0,1,2)), train_y[:,0]
test_x, test_y = test_x.transpose((3,0,1,2)), test_y[:,0]


# In[24]:


print("Train set X and y shapes are ", train_x.shape , train_y.shape)
print("Test set X and y shapes are ", test_x.shape , test_y.shape)


# In[29]:


fig=plt.figure(figsize=(8, 8))
columns = 4
rows = 5
for i in range(1, columns*rows +1):
    img = train_x[i,:,:,:]
    fig.add_subplot(rows, columns, i)
    plt.imshow(img)
    
plt.show()


# In[30]:


print(np.unique(train_y))


# In[31]:


result = np.where(train_y==10)
result


# In[32]:


train_y[train_y==10] = 0
test_y[test_y==10] = 0
train_y[52,]


# In[33]:


model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))


# In[34]:


model.summary()


# In[35]:


model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))


# In[36]:


model.summary()


# In[37]:


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
history = model.fit(train_x, train_y, epochs=10, 
                    validation_data=(test_x, test_y))


# In[38]:


test_loss, test_acc = model.evaluate(test_x,  test_y, verbose=2)


# In[39]:


print(test_acc)


# In[ ]:




