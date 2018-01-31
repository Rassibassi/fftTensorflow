
# coding: utf-8

# In[1]:


import tensorflow as tf
import scipy.io
import numpy as np


# In[9]:


print('tf',tf.__version__)
print('scipy',scipy.__version__)
print('numpy',np.__version__)


# In[2]:


def error(matlab,tf):
    error = np.mean(np.square(np.abs(matlab-np.array(tf))))
    return error
def relError(matlab,tf):
    error = np.mean(np.square(np.abs(matlab-np.array(tf)))/np.square(np.abs(matlab)))
    return error
def tf_i_fft(var,i=1):
    for _ in range(i):
        var = tf.ifft(tf.fft(var))
    return var
def np_i_fft(var,i=1):
    for _ in range(i):
        var = np.fft.ifft(np.fft.fft(var))
    return var    


# In[3]:


def runTF(signal):
    x = tf.placeholder(dtype=tf.complex64,shape=signal.shape)
    out1 = tf_i_fft(x,1)
    out20 = tf_i_fft(x,20)
    out200 = tf_i_fft(x,200)
    with tf.Session() as sess:
        o1,o20,o200 = sess.run([out1,out20,out200],feed_dict={x:signal})
        print("relError():")
        for x in [o1,o20,o200]:
            print(relError(signal, x))
        print("error():")
        for x in [o1,o20,o200]:
            print(error(signal, x))
        
def runNP(signal):
    print("relError():")
    for i in [1,20,200]:
        print(relError(signal, np_i_fft(signal,i)))
    print("error():")
    for i in [1,20,200]:
        print(relError(signal, np_i_fft(signal,i)))


# In[5]:


signal = np.random.normal(size=2**14)+1j*np.random.normal(size=2**14)
print("TF error:")
runTF(signal)
print("NP error:")
runNP(signal)


# In[6]:


signal = np.random.normal(size=2**16)+1j*np.random.normal(size=2**16)
print("TF error:")
runTF(signal)
print("NP error:")
runNP(signal)

