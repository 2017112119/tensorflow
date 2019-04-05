import tensorflow as tf
import numpy as np


#创建一个随机的数据集
x_data=np.random.rand(100).astype(np.float32)
y_data=x_data*0.1+0.3

#随机初始化 权重
Weights = tf.Variable(tf.random_uniform([1],-1.0,1.0))
biases=tf.Variable(tf.zeros([1]))

#估计的y值
y=Weights*x_data+biases


#估计的y和真实的y，计算cost
loss=tf.reduce_mean(tf.square(y-y_data))

#梯度下降优化
optimizer = tf.train.GradientDescentOptimizer(0.5)  #0.5学习率
train = optimizer.minimize(loss)

'''到目前为止，我们只是建立了神经网络的结构，还没有使用这个结构。
在使用这个结构之前，我们必须先初始化所有之前定义的Variable，所以这一步是很重要的
'''
#init=tf.initialize_all_variables() #此方法已被废弃
init = tf.global_variables_initializer()   #这是能用的方法  ，




#创建会话
sess=tf.Session()
sess.run(init)       #用Session来run每一次training的数据



print(x_data)
print(y_data)


for step in range(201):
    sess.run(train)
    if step%20==0:
        print(step,sess.run(Weights),sess.run(biases))


'''
输出：
0 [0.74904037] [-0.09160855]
20 [0.23801073] [0.22408134]
40 [0.12870067] [0.284212]
60 [0.10596858] [0.29671675]
80 [0.10124122] [0.29931724]
100 [0.10025813] [0.299858]
120 [0.10005369] [0.29997048]
140 [0.10001117] [0.29999387]
160 [0.10000232] [0.29999873]
180 [0.10000049] [0.29999974]
200 [0.1000001] [0.29999995]
'''
