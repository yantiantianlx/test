import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

class test_net:
    def __init__(self):
        self.input = tf.placeholder(tf.float32, [None, 28*28])
        self.x = tf.reshape(self.input, [-1, 28, 28, 1]) / 255.
        self.label = tf.placeholder(tf.float32, [None, 10])
        self.keep_prob = tf.placeholder(tf.float32)
        self.is_train = tf.placeholder(tf.bool)
        self.bn_update_op = []

        self.W_b_init()
        self.build_net_without_bn()
        self.build_net_bn_before_relu()
        self.build_net_bn_after_relu()
        self.add_visualization()

    def W_b_init(self):
        xavier_conv = tf.contrib.layers.xavier_initializer_conv2d()
        xavier = tf.contrib.layers.xavier_initializer()
        self.W_conv1 = xavier_conv([5, 5, 1, 6])
        self.W_conv2 = xavier_conv([3,3,6,10])
        self.W_fc1 = xavier([360, 84])
        self.W_fc2 = xavier([84, 10])

    def bn(self, input):
        dimension = len(input.shape)
        mean, var = tf.nn.moments(input, axes=list(range(dimension - 1)))
        avg_mean = tf.Variable(tf.zeros([input.shape[dimension - 1]]))
        avg_var = tf.Variable(tf.zeros(input.shape[dimension - 1]))
        decay = 0.9997
        self.bn_update_op.append(avg_mean.assign(avg_mean.value() * decay + mean * (1-decay)) )
        self.bn_update_op.append(avg_var.assign(avg_var.value() * decay + var * (1-decay)))

        offset = tf.Variable(tf.zeros([input.shape[dimension - 1]]))
        scale = tf.Variable(tf.ones([input.shape[dimension - 1]]))
        mean, var = tf.cond(self.is_train, lambda :(mean,var), lambda : (avg_mean, avg_var))
        bn = tf.nn.batch_normalization(input, mean, var, offset, scale, 0.001)
        return bn

    def build_net_without_bn(self):
        W_conv1 = tf.Variable(self.W_conv1)
        b_conv1 = tf.Variable(tf.constant(0., shape=[6]))
        conv1 = tf.nn.conv2d(self.x, W_conv1, [1,1,1,1], 'SAME')
        relu1 = tf.nn.relu(conv1 + b_conv1)
        pool1 = tf.nn.max_pool(relu1, [1,3,3,1], [1,2,2,1], 'VALID')

        W_conv2 = tf.Variable(self.W_conv2)
        b_conv2 = tf.Variable(tf.constant(0., shape=[10]))
        conv2 = tf.nn.conv2d(pool1, W_conv2, [1,1,1,1], 'SAME')
        relu2 = tf.nn.relu(conv2 + b_conv2)
        pool2 = tf.nn.max_pool(relu2, [1,3,3,1], [1,2,2,1], 'VALID')

        pool2_flatten = tf.reshape(pool2, [-1, 360])
        W_fc1 = tf.Variable(self.W_fc1)
        b_fc1 = tf.Variable(tf.constant(0., shape=[84]))
        fc1 = tf.nn.relu(tf.matmul(pool2_flatten, W_fc1) + b_fc1)
        fc1_dropout = tf.nn.dropout(fc1, self.keep_prob)

        W_fc2 = tf.Variable(self.W_fc2)
        b_fc2 = tf.Variable(tf.constant(0., shape=[10]))
        fc2 = tf.matmul(fc1_dropout, W_fc2) + b_fc2
        self.y_without_bn = tf.nn.softmax(fc2)
        self.loss_without_bn = tf.reduce_mean(-tf.reduce_sum(self.label*tf.log(self.y_without_bn), reduction_indices=[1]))
        self.train_without_bn = tf.train.RMSPropOptimizer(1e-4).minimize(self.loss_without_bn)

        correct_prediction = tf.equal(tf.argmax(self.y_without_bn,1), tf.argmax(self.label,1))
        self.accuracy_without_bn = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def build_net_bn_before_relu(self):
        W_conv1 = tf.Variable(self.W_conv1)
        conv1 = tf.nn.conv2d(self.x, W_conv1, [1,1,1,1], 'SAME')
        bn1 = self.bn(conv1)
        relu1 = tf.nn.relu(bn1)
        pool1 = tf.nn.max_pool(relu1, [1,3,3,1], [1,2,2,1], 'VALID')

        W_conv2 = tf.Variable(self.W_conv2)
        conv2 = tf.nn.conv2d(pool1, W_conv2, [1,1,1,1], 'SAME')
        bn2 = self.bn(conv2)
        relu2 = tf.nn.relu(bn2)
        pool2 = tf.nn.max_pool(relu2, [1,3,3,1], [1,2,2,1], 'VALID')

        pool2_flatten = tf.reshape(pool2, [-1, 360])
        W_fc1 = tf.Variable(self.W_fc1)
        bn3 = self.bn(tf.matmul(pool2_flatten, W_fc1))
        fc1 = tf.nn.relu(bn3)
        fc1_dropout = tf.nn.dropout(fc1, self.keep_prob)

        W_fc2 = tf.Variable(self.W_fc2)
        b_fc2 = tf.Variable(tf.constant(0., shape=[10]))
        fc2 = tf.matmul(fc1_dropout, W_fc2) + b_fc2
        self.y_bn_before_relu = tf.nn.softmax(fc2)
        self.loss_bn_before_relu = tf.reduce_mean(-tf.reduce_sum(self.label*tf.log(self.y_bn_before_relu), reduction_indices=[1]))
        self.train_bn_before_relu = tf.train.RMSPropOptimizer(1e-4).minimize(self.loss_bn_before_relu)

        correct_prediction = tf.equal(tf.argmax(self.y_bn_before_relu, 1), tf.argmax(self.label, 1))
        self.accuracy_bn_before_relu = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def build_net_bn_after_relu(self):
        W_conv1 = tf.Variable(self.W_conv1)
        conv1 = tf.nn.conv2d(self.x, W_conv1, [1,1,1,1], 'SAME')
        relu1 = tf.nn.relu(conv1)
        bn1 = self.bn(relu1)
        pool1 = tf.nn.max_pool(bn1, [1,3,3,1], [1,2,2,1], 'VALID')

        W_conv2 = tf.Variable(self.W_conv2)
        conv2 = tf.nn.conv2d(pool1, W_conv2, [1,1,1,1], 'SAME')
        relu2 = tf.nn.relu(conv2)
        bn2 = self.bn(relu2)
        pool2 = tf.nn.max_pool(bn2, [1,3,3,1], [1,2,2,1], 'VALID')

        pool2_flatten = tf.reshape(pool2, [-1, 360])
        W_fc1 = tf.Variable(self.W_fc1)
        fc1 = tf.nn.relu(tf.matmul(pool2_flatten, W_fc1))
        bn3 = self.bn(fc1)
        fc1_dropout = tf.nn.dropout(bn3, self.keep_prob)

        W_fc2 = tf.Variable(self.W_fc2)
        b_fc2 = tf.Variable(tf.constant(0., shape=[10]))
        fc2 = tf.matmul(fc1_dropout, W_fc2) + b_fc2
        self.y_bn_after_relu = tf.nn.softmax(fc2)
        self.loss_bn_after_relu = tf.reduce_mean(-tf.reduce_sum(self.label*tf.log(self.y_bn_after_relu), reduction_indices=[1]))
        self.train_bn_after_relu = tf.train.RMSPropOptimizer(1e-4).minimize(self.loss_bn_after_relu)

        correct_prediction = tf.equal(tf.argmax(self.y_bn_after_relu, 1), tf.argmax(self.label, 1))
        self.accuracy_bn_after_relu = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def add_visualization(self):
        accuracy1_sum = tf.summary.scalar('accuracy_without_bn', self.accuracy_without_bn)
        accuracy2_sum = tf.summary.scalar('accuracy_bn_before_relu', self.accuracy_bn_before_relu)
        accuracy3_sum = tf.summary.scalar('accuracy_bn_after_relu', self.accuracy_bn_after_relu)

        loss1_sum = tf.summary.scalar('loss_without_bn', self.loss_without_bn)
        loss2_sum = tf.summary.scalar('loss_bn_before_relu', self.loss_bn_before_relu)
        loss3_sum = tf.summary.scalar('loss_bn_after_relu', self.loss_bn_after_relu)
        self.merge = tf.summary.merge([loss1_sum, loss2_sum, loss3_sum])
        self.merge_test = tf.summary.merge([accuracy1_sum,accuracy2_sum,accuracy3_sum])

net = test_net()
tf_sum_writer = tf.summary.FileWriter('logs/')

tf_config = tf.ConfigProto()#log_device_placement=True)
tf_config.gpu_options.allow_growth = True
with tf.Session(config = tf_config) as sess:
    tf_sum_writer.add_graph(sess.graph)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())


    test_step = 0
    for train_step in range(int(1e6)):
        print(train_step)
        batch_input, batch_target = mnist.train.next_batch(32)
        _,_,_, summary,_=sess.run([net.train_without_bn,net.train_bn_before_relu,net.train_bn_after_relu,
                                          net.merge, net.bn_update_op],
            feed_dict={net.input:batch_input,net.label:batch_target,net.keep_prob:0.5, net.is_train:True})
        tf_sum_writer.add_summary(summary, train_step)

        if train_step % 1000 ==0 :
            test_input = mnist.test.images[:1000]
            test_target = mnist.test.labels[:1000]
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            _,_,_,net_test_sum = sess.run([net.accuracy_without_bn, net.accuracy_bn_before_relu, net.accuracy_bn_after_relu, net.merge_test],
                 feed_dict={net.input: test_input, net.label: test_target, net.keep_prob: 1, net.is_train:False},
                                    options=run_options, run_metadata=run_metadata)
            tf_sum_writer.add_run_metadata(run_metadata, 'step%03d' %  train_step)
            tf_sum_writer.add_summary(net_test_sum, test_step)
            test_step += 1
    tf_sum_writer.close()





