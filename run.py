import tensorflow as tf
import scipy.io
import numpy as np

print('tf:',tf.__version__)
print('scipy:',scipy.__version__)
print('np:',np.__version__)

# numpy fft
signal = scipy.io.loadmat('signal.mat')['signal']
signal_f = np.fft.fft(signal)

# tf fft
tf_signal = tf.placeholder(shape=signal.shape,dtype=tf.complex64)
tf_signal_f = tf.fft(tf_signal)
session = tf.InteractiveSession()
out_signal_f, out_signal = session.run([tf_signal_f,tf_signal], feed_dict={tf_signal:signal})

errorBefore = np.mean(np.power(np.abs(out_signal-signal),2))
errorAfter = np.mean(np.power(np.abs(out_signal_f-signal_f),2))
print('Error before fft:',errorBefore)
print('Error after fft:',errorAfter)