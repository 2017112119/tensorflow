#import tensorflow and pyplot
import tensorflow as tf
import matplotlib.pyplot as plt

#tf.Graph represents a collection of tf.Operations
#you can creat operations by writing out equations

#First a tf.Tensor
n_values = 32
x = tf.linspace(-3.0,3.0,n_values)

#Construct a tf.Session to execute the graph.
sess = tf.Session()
result = sess.run(x)

#Alternatively pass a session to the eval fn
x.eval(session=sess)
#x.eval() does not work, as it requires a session!

#we can setup an interactive session if we don't
#want to keep passing the session around
sess.close()
sess = tf.InteractiveSession()

#Now this will work!
x.eval()

#Now a tf.Operation

