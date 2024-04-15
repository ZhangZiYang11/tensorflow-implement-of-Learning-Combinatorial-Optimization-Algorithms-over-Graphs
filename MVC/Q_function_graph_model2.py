import tensorflow as tf

def Q_func(x, adj, w, p, T, initialization_stddev,adjust_matrix,multiply_matrix,aux,
           scope, reuse=False, pre_pooling_mlp_layers = 1, post_pooling_mlp_layers = 1):
    """
    x:      B x n_vertices.
    Placeholder for the current state of the solution.
    Each row of x is a binary vector encoding
    which vertices are included in the current partial solution.
    adj:    n_vertices x n_vertices.
    A placeholder for the adjacency matrix of the graph.
    w:      n_vertices x n_vertice.
    A placeholder fot the weights matrix of the graph.
    """
    #x用来表示当前解，为与顶点数相同维数的列表，值为1表示在部分解中0表示不在，adj为邻接矩阵（元素为bool值），w为权重矩阵，注意x的shape为[None,n]，None表示采样的数量。
    #adj和w的shape为[None,n,n]其中None为采样的数量
    #p为论文中生成的embedding的维数，initialization_stddev为初始化时的标准差
    #T为迭代次数，即S2V层数，论文中为4层
    #sample_true_nodes为nparray，指示各样本的实际点的个数
    #adjust_matrix用来将多余点的embedding重置为0，mutiply_matrix为最大维数的全1矩阵
    with tf.variable_scope(scope, reuse=reuse):
        with tf.variable_scope('thetas'):
            #theta为神经网络中参数，对应论文中θ，通过tf.variable设定为可优化变量，tf.random_normal()生成指定形状的正态分布张量，均值为0，标准差为stddev
            global theta1
            theta1 = tf.Variable(tf.random_normal([p], stddev=initialization_stddev), name='theta1')
            theta2 = tf.Variable(tf.random_normal([p, p], stddev=initialization_stddev), name='theta2')
            theta3 = tf.Variable(tf.random_normal([p, p], stddev=initialization_stddev), name='theta3')
            theta4 = tf.Variable(tf.random_normal([p], stddev=initialization_stddev), name='theta4')
            theta5 = tf.Variable(tf.random_normal([2 * p + 3], stddev=initialization_stddev), name='theta5')
            theta6 = tf.Variable(tf.random_normal([p, p], stddev=initialization_stddev), name='theta6')
            theta7 = tf.Variable(tf.random_normal([p, p], stddev=initialization_stddev), name='theta7')


        with tf.variable_scope('pre_pooling_MLP', reuse=reuse):
            Ws_pre_pooling = []; bs_pre_pooling = []
            for i in range(pre_pooling_mlp_layers):
                Ws_pre_pooling.append(tf.Variable(tf.random_normal([p, p], stddev=initialization_stddev),
                                          name='W_MLP_pre_pooling_' + str(i)))
                bs_pre_pooling.append(tf.Variable(tf.random_normal([p], stddev=initialization_stddev),
                                          name='b_MLP_pre_pooling_' + str(i)))

            Ws_post_pooling = []; bs_post_pooling = []
            for i in range(post_pooling_mlp_layers):
                Ws_post_pooling.append(tf.Variable(tf.random_normal([p, p], stddev=initialization_stddev),
                                          name='W_MLP_post_pooling_' + str(i)))
                bs_post_pooling.append(tf.Variable(tf.random_normal([p], stddev=initialization_stddev),
                                          name='b_MLP_post_pooling_' + str(i)))

        # Define the mus
        # Loop over t
        for t in range(T):
            # First part of mu
            mu_part1 = tf.einsum('iv,k->ivk', x, theta1)

            # Second part of mu
            if t != 0:
                # Add some non linear transformation of the neighbors' embedding before pooling
                with tf.variable_scope('pre_pooling_MLP', reuse=reuse):
                    for i in range(pre_pooling_mlp_layers):
                        mu = tf.nn.relu(tf.einsum('kl,ivk->ivl', Ws_pre_pooling[i], mu) + bs_pre_pooling[i])
                        mu = tf.einsum('ivu,iuk->ivk', adjust_matrix,mu)

                mu_part2 = tf.einsum('kl,ivk->ivl', theta2, tf.einsum('ivu,iuk->ivk', adj, mu))
                # Add some non linear transformations of the pooled neighbors' embeddings
                with tf.variable_scope('post_pooling_MLP', reuse=reuse):
                    for i in range(post_pooling_mlp_layers):
                        mu_part2 = tf.nn.relu(tf.einsum('kl,ivk->ivl', Ws_post_pooling[i], mu_part2) + bs_post_pooling[i])

            # Third part of mu
            mu_part3_0 = tf.einsum('ikvu->ikv', tf.nn.relu(tf.einsum('k,ivu->ikvu', theta4, w)))
            mu_part3_1 = tf.einsum('kl,ilv->ivk', theta3, mu_part3_0)

            # All all of the parts of mu and apply ReLui
            if t != 0:
                mu = tf.nn.relu(tf.add(mu_part1 + mu_part2, mu_part3_1, name='mu_' + str(t)))
                mu = tf.einsum('ivu,iuk->ivk', adjust_matrix,mu)
            else:
                mu = tf.nn.relu(tf.add(mu_part1, mu_part3_1, name='mu_' + str(t)))
                mu = tf.einsum('ivu,iuk->ivk', adjust_matrix,mu)

        # Define the Qs
        Q_part1 = tf.einsum('kl,ivk->ivl', theta6, tf.einsum('ivu,iuk->ivk',multiply_matrix, mu))
        Q_part2 = tf.einsum('kl,ivk->ivl', theta7, mu)
        return tf.identity(tf.einsum('iv,ivk->ik',tf.einsum('k,ivk->iv', theta5,
                                     tf.concat([tf.nn.relu(tf.concat([Q_part1, Q_part2], axis=2)),aux], axis=2)),adjust_matrix),
                           name='Q')

def target_Q_func(x, adj, w, p, T, initialization_stddev,adjust_matrix,multiply_matrix,aux,
           scope, reuse=False, pre_pooling_mlp_layers = 1, post_pooling_mlp_layers = 1):
    with tf.variable_scope(scope, reuse=False):
        with tf.variable_scope('thetas'):
            #theta为神经网络中参数，对应论文中θ，通过tf.variable设定为可优化变量，tf.random_normal()生成指定形状的正态分布张量，均值为0，标准差为stddev
            global theta1
            theta1 = tf.Variable(tf.random_normal([p], stddev=initialization_stddev), trainable=False,name='theta1')
            theta2 = tf.Variable(tf.random_normal([p, p], stddev=initialization_stddev),  trainable=False,name='theta2')
            theta3 = tf.Variable(tf.random_normal([p, p], stddev=initialization_stddev), trainable=False, name='theta3')
            theta4 = tf.Variable(tf.random_normal([p], stddev=initialization_stddev), trainable=False, name='theta4')
            theta5 = tf.Variable(tf.random_normal([2 * p + 3], stddev=initialization_stddev), trainable=False, name='theta5')
            theta6 = tf.Variable(tf.random_normal([p, p], stddev=initialization_stddev), trainable=False, name='theta6')
            theta7 = tf.Variable(tf.random_normal([p, p], stddev=initialization_stddev), trainable=False, name='theta7')


        with tf.variable_scope('pre_pooling_MLP', reuse=False):
            Ws_pre_pooling = []; bs_pre_pooling = []
            for i in range(pre_pooling_mlp_layers):
                Ws_pre_pooling.append(tf.Variable(tf.random_normal([p, p], stddev=initialization_stddev), trainable=False,
                                          name='W_MLP_pre_pooling_' + str(i)))
                bs_pre_pooling.append(tf.Variable(tf.random_normal([p], stddev=initialization_stddev), trainable=False,
                                          name='b_MLP_pre_pooling_' + str(i)))

            Ws_post_pooling = []; bs_post_pooling = []
            for i in range(post_pooling_mlp_layers):
                Ws_post_pooling.append(tf.Variable(tf.random_normal([p, p], stddev=initialization_stddev), trainable=False,
                                          name='W_MLP_post_pooling_' + str(i)))
                bs_post_pooling.append(tf.Variable(tf.random_normal([p], stddev=initialization_stddev), trainable=False,
                                          name='b_MLP_post_pooling_' + str(i)))

        # Define the mus
        # Loop over t
        for t in range(T):
            # First part of mu
            mu_part1 = tf.einsum('iv,k->ivk', x, theta1)

            # Second part of mu
            if t != 0:
                # Add some non linear transformation of the neighbors' embedding before pooling
                with tf.variable_scope('pre_pooling_MLP', reuse=False):
                    for i in range(pre_pooling_mlp_layers):
                        mu = tf.nn.relu(tf.einsum('kl,ivk->ivl', Ws_pre_pooling[i], mu) + bs_pre_pooling[i])
                        mu = tf.einsum('ivu,iuk->ivk', adjust_matrix,mu)

                mu_part2 = tf.einsum('kl,ivk->ivl', theta2, tf.einsum('ivu,iuk->ivk', adj, mu))
                # Add some non linear transformations of the pooled neighbors' embeddings
                with tf.variable_scope('post_pooling_MLP', reuse=False):
                    for i in range(post_pooling_mlp_layers):
                        mu_part2 = tf.nn.relu(tf.einsum('kl,ivk->ivl', Ws_post_pooling[i], mu_part2) + bs_post_pooling[i])

            # Third part of mu
            mu_part3_0 = tf.einsum('ikvu->ikv', tf.nn.relu(tf.einsum('k,ivu->ikvu', theta4, w)))
            mu_part3_1 = tf.einsum('kl,ilv->ivk', theta3, mu_part3_0)

            # All all of the parts of mu and apply ReLui
            if t != 0:
                mu = tf.nn.relu(tf.add(mu_part1 + mu_part2, mu_part3_1, name='mu_' + str(t)))
                mu = tf.einsum('ivu,iuk->ivk', adjust_matrix,mu)
            else:
                mu = tf.nn.relu(tf.add(mu_part1, mu_part3_1, name='mu_' + str(t)))
                mu = tf.einsum('ivu,iuk->ivk', adjust_matrix,mu)

        # Define the Qs
        Q_part1 = tf.einsum('kl,ivk->ivl', theta6, tf.einsum('ivu,iuk->ivk',multiply_matrix, mu))
        Q_part2 = tf.einsum('kl,ivk->ivl', theta7, mu)
        return tf.identity(tf.einsum('iv,ivk->ik',tf.einsum('k,ivk->iv', theta5,
                                     tf.concat([tf.nn.relu(tf.concat([Q_part1, Q_part2], axis=2)),aux], axis=2)),adjust_matrix),
                           name='Q')