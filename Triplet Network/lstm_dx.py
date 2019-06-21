from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
from tensorflow.contrib import rnn
from model.triplet_loss import batch_all_triplet_loss
from model.triplet_loss import batch_hard_triplet_loss
import pprint, pickle
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn import metrics

# 解析数据
data1=[]
with open('../1.pkl', 'rb') as f1:
    data1 = np.array(pickle.load(f1))
    print("data1=", data1.shape)

data2=[]
with open('../2.pkl', 'rb') as f2:
    data2 = np.array(pickle.load(f2))
    print("data2=", data2.shape)

# 预处理
data1_arr=[]
data2_arr=[]
for i in range(1000):
    data1_seq = pad_sequences(data1.reshape(300, -1), maxlen=300, dtype="float32")
    data2_seq = pad_sequences(data2.reshape(300, -1), maxlen=300, dtype="float32")
    data1_arr.append(data1_seq)
    data2_arr.append(data2_seq)
#     # print(i)
#     # print(data1_seq.shape)
#     # print(data2_seq.shape)
data1_arr = np.array(data1_arr)
data2_arr = np.array(data2_arr)
print("data1_arr=",data1_arr.shape)  #data1_arr= (1000, 300, 300)
print("data1_arr=",data2_arr.shape)  #data1_arr= (1000, 300, 300)

# 数据集拆分
print("------------------------ experiment ---------------------------")
data1_size = 50
data2_size = 50
data1_arr_train = data1_arr[0:data1_size]
data2_arr_train = data2_arr[0:data2_size]
labels_train = []
data_train = np.append(data1_arr_train, data2_arr_train, axis=0)
for i in range(len(data1_arr_train)):
    labels_train.append([0])
for i in range(len(data2_arr_train)):
    labels_train.append([1])
labels_train = np.array(labels_train)
data1_arr_test = data1_arr[100:150]
data2_arr_test = data2_arr[100:150]
data_test = np.append(data1_arr_test, data2_arr_test, axis=0)
labels_test = []
for i in range(len(data1_arr_test)):
    labels_test.append([0])
for i in range(len(data2_arr_test)):
    labels_test.append([1])
labels_test = np.array(labels_test)

print("labels_train=", labels_train.shape)#labels_train= (100, 1)
print("data_train=", data_train.shape) #data_train= (100, 300, 300)

# 训练
batch_size = 100        # batch的大小
time_step = 300         # LSTM网络中的时间步（每个时间步处理图像的一行）
data_length = 300       # 每个时间步输入数据的长度（这里就是图像的宽度）
learning_rate = 0.01    # 学习率

# 定义相关数据的占位符
X_ = tf.placeholder(tf.float32, [None, 300, 300])     # 输入数据
Y_ = tf.placeholder(tf.float32, [None, 1])            # 数据集的类标
inputs = tf.reshape(X_, [-1, time_step, data_length])  # dynamic_rnn的输入数据(batch_size, max_time, ...)

# 验证集
validate_data = {X_: data_train, Y_: labels_train}
# 测试集
test_data = {X_: data_test, Y_: labels_test}

# 定义一个两层的LSTM模型
lstm_layers = rnn.MultiRNNCell([rnn.BasicLSTMCell(num_units=num)
                                for num in [300, 300]], state_is_tuple=True)
# 定义一个两层的GRU模型
# gru_layers = rnn.MultiRNNCell([rnn.GRUCell(num_units=num)
#                                 for num in [100, 100]], state_is_tuple=True)

outputs, h_ = tf.nn.dynamic_rnn(lstm_layers, inputs, dtype=tf.float32)
# outputs, h_ = tf.nn.dynamic_rnn(gru_layers, inputs, dtype=tf.float32)
output = tf.layers.dense(outputs[:, -1, :], 2)  # 获取LSTM网络的最后输出状态

print("------------------------ loss ---------------------------")
# 定义交叉熵损失函数和优化器
# loss = tf.losses.huber_loss(labels=Y_, predictions=output)  # compute cost
# train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
# triplet loss
margin = 0.002
data_triplet_train = tf.reshape(outputs, [-1, 300*300])
# print(data_triplet_train)
loss = batch_hard_triplet_loss(labels_train, data_triplet_train, margin, squared=False)
#loss, fraction_positive_triplets= batch_all_triplet_loss(labels_train, data_triplet_train, margin, squared=False)
# tf.summary.scalar('loss', loss)
# print(loss)
global_step = tf.train.get_global_step()
train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
# predictions=output
predictions=tf.argmax(output, axis=1)
print(predictions)
# 计算准确率
# accuracy = tf.metrics.accuracy(
#     labels=Y_, predictions=tf.argmax(output, axis=1))[1]
accuracy = tf.metrics.accuracy(
    labels=tf.argmax(Y_, axis=1), predictions=tf.argmax(output, axis=1))[1]

# 初始化变量
sess = tf.Session()
init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
sess.run(init)
print(Y_)
print(output)

for step in range(300):
    # 获取一个batch的训练数据
    train_x = data_train
    train_y = labels_train
    train_op_ = sess.run(train_op, {X_: train_x, Y_: train_y})

    prediction1, loss_ = sess.run([predictions, loss], {X_: train_x, Y_: train_y})
    loss_ = sess.run(loss, {X_: train_x, Y_: train_y})
    # print(loss_)
    output_result, Y_result = sess.run([output, Y_], {X_: train_x, Y_: train_y})

    # print(train_y)
    # print(prediction1)
    # prediction_positive=0
    # for i in range(len(prediction1)):
    #     if(prediction1[i]==train_y[i]):
    #         prediction_positive+=1
    # print("prediction_positive:")
    # print(prediction_positive)

    # 在验证集上计算准确率
    if step % 50 == 0:
        val_acc = sess.run(accuracy, feed_dict=validate_data)
        print('train loss: %.4f' % loss_, '| val accuracy: %.2f' % val_acc)

# 计算测试集上的准确率
test_acc = sess.run(accuracy, feed_dict=test_data)
print('test accuracy %.4f' % test_acc)

# cm = metrics.confusion_matrix(labels_test, predictions)
