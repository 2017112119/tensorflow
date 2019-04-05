'''
加速神经网络训练
使用梯度下降的时候，有一个问题，就是在网络非常复杂的时候，梯度下降的时候对计算的需求非常高。

常见的几种加速求解cost的最小值的方法：

Stochastic Gradient Descent (SGD)

Momentum

AdaGrad

RMSProp

Adam

在tensorflow里面提供的几种优化器

tf.train.GradientDescentOptimizer
tf.train.AdaeltaOptimizer
tf.train.AdagradDAOptimizer
tf.train.AdamOptimizer
tf.train.MomentumOptimizer
'''
