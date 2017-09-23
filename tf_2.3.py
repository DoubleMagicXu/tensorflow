import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
x=tf.placeholder("float",[None,784])#x是占位符,None代表输入的个数未知，但是我们知道这个数据有784维
W=tf.Variable(tf.zeros([784,10]))
b=tf.Variable(tf.zeros([10]))#注意：这里的shape要和W的列看齐，而不是和行数看齐
m_y=tf.nn.softmax(tf.matmul(x,W)+b)#注意乘法的顺序,m_y是我们拟合的值
y=tf.placeholder("float",[None,10])#这是正确的值

#计算交叉熵
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y* tf.log(m_y), reduction_indices=[1]))   #内积

train_step=tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)#梯度下降法

#初始化
init=tf.initialize_all_variables()

sess=tf.Session()
sess.run(init)

#训练
for i in range(1000):#训练1000次
    xs,ys=mnist.train.next_batch(100)#随机抓取100个图片
    sess.run(train_step,feed_dict={x:xs,y:ys})
    correct_prediction=tf.equal(tf.argmax(m_y,1),tf.argmax(y,1))
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,"float"))#cast类型转化  再求平均值
    print(sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels}))

