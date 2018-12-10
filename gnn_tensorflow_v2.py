import argparse
import os.path
import glob
import numpy as np
import time
from termcolor import *
import tensorflow as tf


GRAPH_CONV_LAYER_CHANNEL = 32
CONV1D_1_OUTPUT = 16
CONV1D_2_OUTPUT = 32
CONV1D_1_FILTER_WIDTH = GRAPH_CONV_LAYER_CHANNEL * 3
CONV1D_2_FILTER_WIDTH = 5
DENSE_NODES = 128
DROP_OUTPUT_KEEP_PROB = 0.5


parser = argparse.ArgumentParser(description="GNN(graph neural network)-tensorflow")
parser.add_argument("--data", type=str, help="name of data", default="mutag")
parser.add_argument("-E", "--epoch", type=int, default=100, help="pass through all training set call a EPOCH")
parser.add_argument("-r", "--learning_rate", type=float, default=0.0001, help="learning rate")
args = parser.parse_args()


def load_networks():
    print("load data...")
    file_list = []
    file_glob_pattern = os.path.join("graph_data", args.data, "mutag*.graph")
    file_list.extend(glob.glob(file_glob_pattern))

    edges_set, labels_set, nodes_size_list = {}, {}, {}
    for file in file_list:
        base_name = os.path.basename(file)
        edges_set[base_name] = []
        with open(file, "r") as f:
            line, read = f.readline(), False
            while line:
                if line.startswith("#c - Class"):
                    labels_set[base_name] = (int(f.readline().strip()))
                    break
                if read:
                    edges_set[base_name].append([int(x) for x in line.strip().split(",")[:2]])
                if line.startswith("#e - edge labels"):
                    read = True
                line = f.readline()
        nodes_size_list[base_name] = max(max(np.array(edges_set[base_name])[:, 0]), max(np.array(edges_set[base_name])[:, 1]))

    A, Y, count = [], [], 0
    for key, value in edges_set.items():
        A.append(np.zeros([nodes_size_list[key], nodes_size_list[key]], dtype=np.float32))
        for edge in value:
            A[count][edge[0] - 1][edge[1] - 1] = 1.
        Y.append([labels_set[key]])
        count += 1
    A, Y = np.array(A), np.array(Y)
    Y = np.where(Y == -1, 0, 1)
    print("\tpositive examples: %d, negative examples: %d." % (np.sum(Y == 0), np.sum(Y == 1)))
    print("\tX.shape: %s, Y.shape: %s" % (A.shape, Y.shape))
    # get A_tilde
    for index, x in enumerate(A):
        A[index] = x + np.eye(x.shape[0])
    # get D_inverse
    D_inverse = []
    for x in A:
        D_inverse.append(np.linalg.inv(np.diag(np.sum(x, axis=1))))
    nodes_list = [x for x in nodes_size_list.values()]
    print("\tmax graph size: %d, min graph size: %d.", max(nodes_list), min(nodes_list))
    return np.array(D_inverse), A, Y, nodes_list # A is A_tilde


def create_attribution(attribute, dimension, nodes_size_list, A_title):
    """
    :param attribute: 'label' or 'attribute' or 'degree'
    :param dimension: dimension of the initial attribute of node
    :param graph_size_list: nodes of each graph in the graph data
    :return: X
    """
    if dimension is None:
        if attribute == "degree":
            print("\tX: normalized node degree.")
            degree_normalized = []
            for graph in A_title:
                degree_total = np.sum(graph, axis=1)
                degree_normalized.append(np.divide(degree_total, np.sum(degree_total)).reshape(-1, 1))
            return np.array(degree_normalized)
        elif attribute == "label":
            print("\tX: np.ones([node_size, 1]).")
            return np.array([np.ones([x, 1]) for x in nodes_size_list])
        elif attribute == "onehot":
            print("\tX: np.eye().")
            return np.array([np.eye(x) for x in nodes_size_list])
    else:
        # read attribute.
        pass


def split_train_test(D_inverse, A_tilde, X, Y, nodes_size_list, rate):
    print("split training and test data...")
    state = np.random.get_state()
    np.random.shuffle(D_inverse)
    np.random.set_state(state)
    np.random.shuffle(A_tilde)
    np.random.set_state(state)
    np.random.shuffle(X)
    np.random.set_state(state)
    np.random.shuffle(Y)
    np.random.set_state(state)
    np.random.shuffle(nodes_size_list)
    data_size = Y.shape[0]
    training_set_size, test_set_size = int(data_size * (1 - rate)), int(data_size * rate)
    D_inverse_train, D_inverse_test = D_inverse[: training_set_size], D_inverse[training_set_size:]
    A_tilde_train, A_tilde_test = A_tilde[: training_set_size], A_tilde[training_set_size:]
    X_train, X_test = X[: training_set_size], X[training_set_size:]
    Y_train, Y_test = Y[: training_set_size], Y[training_set_size:]
    nodes_size_list_train, nodes_size_list_test = nodes_size_list[: training_set_size], nodes_size_list[training_set_size:]
    print("\tabout train: positive examples(%d): %d, negative examples: %d."
          % (training_set_size, np.sum(Y_train == 1), np.sum(Y_train == 0)))
    print("\tabout test: positive examples(%d): %d, negative examples: %d."
          % (test_set_size, np.sum(Y_test == 1), np.sum(Y_test == 0)))
    return D_inverse_train, D_inverse_test, A_tilde_train, A_tilde_test, X_train, X_test, Y_train, Y_test, \
           nodes_size_list_train, nodes_size_list_test


def variable_summary(var):
    var_mean = tf.reduce_mean(var)
    var_variance = tf.square(tf.reduce_mean(tf.square(var - tf.reduce_mean(var))))
    var_max = tf.reduce_max(var)
    var_min = tf.reduce_min(var)
    return var_mean, var_variance, var_max, var_min


def GNN(X_train, D_inverse_train, A_tilde_train, Y_train, nodes_size_list_train, top_k, initial_channels,
        X_test, D_inverse_test, A_tilde_test, Y_test, nodes_size_list_test):
    # placeholder
    D_inverse_pl = tf.placeholder(dtype=tf.float32, shape=[None, None])
    A_tilde_pl = tf.placeholder(dtype=tf.float32, shape=[None, None])
    X_pl = tf.placeholder(dtype=tf.float32, shape=[None, initial_channels])
    Y_pl = tf.placeholder(dtype=tf.int32, shape=[1], name="Y-placeholder")
    node_size_pl = tf.placeholder(dtype=tf.int32, shape=[], name="node-size-placeholder")
    is_train = tf.placeholder(dtype=tf.uint8, shape=[], name="is-train-or-test")

    # trainable parameters of graph convolution layer
    graph_weight_1 = tf.Variable(tf.truncated_normal(shape=[initial_channels, GRAPH_CONV_LAYER_CHANNEL],
                                                     stddev=0.1, dtype=tf.float32))
    graph_weight_2 = tf.Variable(tf.truncated_normal(shape=[GRAPH_CONV_LAYER_CHANNEL, GRAPH_CONV_LAYER_CHANNEL],
                                                     stddev=0.1, dtype=tf.float32))
    graph_weight_3 = tf.Variable(tf.truncated_normal(shape=[GRAPH_CONV_LAYER_CHANNEL, GRAPH_CONV_LAYER_CHANNEL],
                                                     stddev=0.1, dtype=tf.float32))
    graph_weight_4 = tf.Variable(tf.truncated_normal(shape=[GRAPH_CONV_LAYER_CHANNEL, 1], stddev=0.1, dtype=tf.float32))

    # GRAPH CONVOLUTION LAYER 1
    gl_1_XxW = tf.matmul(X_pl, graph_weight_1)                  # shape=(node_size/None, 32)
    gl_1_AxXxW = tf.matmul(A_tilde_pl, gl_1_XxW)                # shape=(node_size/None, 32)
    Z_1 = tf.nn.tanh(tf.matmul(D_inverse_pl, gl_1_AxXxW))       # shape=(node_size/None, 32)
    # GRAPH CONVOLUTION LAYER 2
    gl_2_XxW = tf.matmul(Z_1, graph_weight_2)                   # shape=(node_size/None, 32)
    gl_2_AxXxW = tf.matmul(A_tilde_pl, gl_2_XxW)                # shape=(node_size/None, 32)
    Z_2 = tf.nn.tanh(tf.matmul(D_inverse_pl, gl_2_AxXxW))       # shape=(node_size/None, 32)
    # GRAPH CONVOLUTION LAYER 3
    gl_3_XxW = tf.matmul(Z_2, graph_weight_3)                   # shape=(node_size/None, 32)
    gl_3_AxXxW = tf.matmul(A_tilde_pl, gl_3_XxW)                # shape=(node_size/None, 32)
    Z_3 = tf.nn.tanh(tf.matmul(D_inverse_pl, gl_3_AxXxW))       # shape=(node_size/None, 32)
    # GRAPH CONVOLUTION LAYER 4
    gl_4_XxW = tf.matmul(Z_3, graph_weight_4)                   # shape=(node_size/None, 1)
    gl_4_AxXxW = tf.matmul(A_tilde_pl, gl_4_XxW)                # shape=(node_size/None, 1)
    Z_4 = tf.nn.tanh(tf.matmul(D_inverse_pl, gl_4_AxXxW))       # shape=(node_size/None, 1)
    graph_conv_output = tf.concat([Z_1, Z_2, Z_3], axis=1)      # shape=(node_size/None, 32 + 32 + 32)

    var_mean, var_variance, var_max, var_min = variable_summary(graph_weight_1)

    # SORT POOLING LAYER
    graph_conv_output_stored = tf.gather(graph_conv_output, tf.nn.top_k(Z_4[:, 0], node_size_pl).indices)

    # the unifying is done by deleting the last n-k rows if n > k;
    # or adding k-n zero rows if n < k.
    graph_conv_output_top_k = tf.cond(tf.less(node_size_pl, top_k),
                                      lambda: tf.concat(axis=0,
                                                        values=[graph_conv_output_stored,
                                                                tf.zeros(dtype=tf.float32,
                                                                         shape=[top_k-node_size_pl,
                                                                                GRAPH_CONV_LAYER_CHANNEL*3])]),
                                      lambda: tf.slice(graph_conv_output_stored, begin=[0, 0], size=[top_k, -1]))
    # assert graph_conv_output_top_k.shape == [top_k, 32*3]

    graph_conv_output_flatten = tf.reshape(graph_conv_output_top_k, shape=[1, GRAPH_CONV_LAYER_CHANNEL*3*top_k, 1])
    assert graph_conv_output_flatten.shape == [1, GRAPH_CONV_LAYER_CHANNEL*3*top_k, 1]

    # 1-D CONVOLUTION LAYER: (filter_width, in_channel, out_channel)
    conv1d_kernel_1 = tf.Variable(tf.truncated_normal(shape=[CONV1D_1_FILTER_WIDTH, 1, CONV1D_1_OUTPUT],
                                                      stddev=0.1, dtype=tf.float32))
    conv_1d_a = tf.nn.conv1d(graph_conv_output_flatten, conv1d_kernel_1, stride=CONV1D_1_FILTER_WIDTH, padding="VALID")
    assert conv_1d_a.shape == [1, top_k, CONV1D_1_OUTPUT]
    conv1d_kernel_2 = tf.Variable(tf.truncated_normal(shape=[CONV1D_2_FILTER_WIDTH, CONV1D_1_OUTPUT, CONV1D_2_OUTPUT],
                                                      stddev=0.1, dtype=tf.float32))
    conv_1d_b = tf.nn.conv1d(conv_1d_a, conv1d_kernel_2, stride=1, padding="VALID")
    assert conv_1d_b.shape == [1, top_k - CONV1D_2_FILTER_WIDTH + 1, CONV1D_2_OUTPUT]

    conv_output_flatten = tf.layers.flatten(conv_1d_b)

    # DENSE LAYER
    weight_1 = tf.Variable(tf.truncated_normal(shape=[int(conv_output_flatten.shape[1]), DENSE_NODES], stddev=0.1))
    bias_1 = tf.Variable(tf.zeros(shape=[DENSE_NODES]))
    dense_z = tf.nn.relu(tf.matmul(conv_output_flatten, weight_1) + bias_1)
    if is_train == 1:
        dense_z = tf.layers.dropout(dense_z, DROP_OUTPUT_KEEP_PROB)

    weight_2 = tf.Variable(tf.truncated_normal(shape=[DENSE_NODES, 2]))
    bias_2 = tf.Variable(tf.zeros(shape=[2]))
    pre_y = tf.matmul(dense_z, weight_2) + bias_2

    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Y_pl, logits=pre_y))

    train_op = tf.train.AdamOptimizer(0.0001).minimize(loss)

    train_data_size = X_train.shape[0]
    test_data_size = X_test.shape[0]

    with tf.Session() as sess:
        print("\nstart training gnn.")
        start_t = time.time()
        sess.run(tf.global_variables_initializer())
        batch_index = 0
        for step in range(args.epoch * train_data_size):
            batch_index = batch_index + 1 if batch_index < train_data_size - 1 else 0
            feed_dict = {D_inverse_pl: D_inverse_train[batch_index],
                         A_tilde_pl: A_tilde_train[batch_index],
                         X_pl: X_train[batch_index],
                         Y_pl: Y_train[batch_index],
                         node_size_pl: nodes_size_list_train[batch_index],
                         is_train: 1
                         }
            loss_value, _ = sess.run([loss, train_op], feed_dict=feed_dict)
            if step % 1000 == 0:
                train_acc = 0
                for i in range(train_data_size):
                    feed_dict = {D_inverse_pl: D_inverse_train[i],
                                 A_tilde_pl: A_tilde_train[i],
                                 X_pl: X_train[i],
                                 Y_pl: Y_train[i],
                                 node_size_pl: nodes_size_list_train[i],
                                 is_train: 0
                                 }
                    pre_y_value = sess.run(pre_y, feed_dict=feed_dict)
                    if np.argmax(pre_y_value, 1) == Y_train[i]:
                        train_acc += 1
                train_acc = train_acc / train_data_size

                test_acc = 0
                for i in range(test_data_size):
                    feed_dict = {D_inverse_pl: D_inverse_test[i],
                                 A_tilde_pl: A_tilde_test[i],
                                 X_pl: X_test[i],
                                 Y_pl: Y_test[i],
                                 node_size_pl: nodes_size_list_test[i],
                                 is_train: 0
                                 }
                    pre_y_value = sess.run(pre_y, feed_dict=feed_dict)
                    if np.argmax(pre_y_value, 1) == Y_test[i]:
                        test_acc += 1
                test_acc = test_acc / test_data_size

                # mean_value, var_value, max_value, min_value = sess.run([var_mean,
                #                                                         var_variance,
                #                                                         var_max,
                #                                                         var_min],
                #                                                        feed_dict=feed_dict)
                # print("debug: mean: %f, variance: %f, max: %f, min: %f." %
                #       (mean_value, var_value, max_value, min_value))

                print("After %5s step, the loss is %f, training acc %f, test acc %f."
                      % (step, loss_value, train_acc, test_acc))
        end_t = time.time()
        print("time consumption: ", end_t - start_t)


if __name__ == "__main__":
    D_inverse, A_tilde, Y, nodes_size_list, = load_networks()
    X = create_attribution("degree", None, nodes_size_list, A_tilde)
    D_inverse_train, D_inverse_test, A_tilde_train, A_tilde_test, X_train, X_test, Y_train, Y_test, \
    nodes_size_list_train, nodes_size_list_test = split_train_test(D_inverse, A_tilde, X, Y, nodes_size_list, 0.1)
    initial_feature_dimension = 1
    GNN(X_train, D_inverse_train, A_tilde_train, Y_train, nodes_size_list_train, 25, initial_feature_dimension,
        X_test, D_inverse_test, A_tilde_test, Y_test, nodes_size_list_test)

