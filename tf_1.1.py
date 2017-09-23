import tensorflow as tf##引入模块
import numpy as np
x=np.float32(np.random.rand(2,100))#二维数组，也可看作是2*100矩阵
y=np.dot([0.1,0.2],x)+0.3  #矩阵乘法+0.3*E 结果为1*100矩阵
'''这里自变量为x，因变量为y  显然:y=w*x+b
   现在告诉你 x,y ,求b以及w
'''
b=tf.Variable(tf.zeros([1]))#？一行
w=tf.Variable(tf.random_uniform([1,2],-1,1))#范围     一行两列
m_y=tf.matmul(w,x)+b #矩阵乘法m_ym_y

#最小化方差
loss=tf.reduce_mean(tf.square(y-m_y))#平方和，再平均
optimizer=tf.train.GradientDescentOptimizer(0.5)#梯度下降，步长0.5
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()


sess = tf.Session()
sess.run(init)



for step in range(0, 201):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(w), sess.run(b))



