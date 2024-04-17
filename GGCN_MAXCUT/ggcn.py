import tensorflow as tf

#总体上和GGCN的区别在于：1.feat输入不同，2.没使用BN，而是多乘了一次参数，因为不是全连接图没法做BN，并且GGCN原论文也没有进行BN。
#3.GGCN源码中边进行更新时对于两端点乘以相同的参数，这里乘的是不同的参数，两端点对于边ij是有区别的，认为乘以不同参数会好一点。
#注意GGCN源码中边ij和边ji的初始embedding是有区别的，原因在于ij的knn关系，但是每次更新更新是一样的
def init_layer(node_feat, adj, w, p, initialization_stddev,
               edge_feat, scope):
    '''
    第一层处理，将node_feat和edge_feat转化成，scope用来指示double dqn
    由于适用于maxcut问题，点和边的初始特征沿用原形式
    点特征为2维，只是表示是否在部分解内
    边特征为4维，边ij的feature eij第一维为i是否在部分解内，在则为1；第二维为权重；第三维为ij是否为割边；最后一维为偏置1
    注意，每个边都有两个embedding，edge_feat[i]表示以i为起点的所有边的embedding，计算i的embedding时只用edge_feat[i]的所有embedding即可
    '''

    node_init = tf.Variable(tf.random_normal([2,p],stddev=initialization_stddev),name='node_init')
    edge_init = tf.Variable(tf.random_normal([4,p],stddev=initialization_stddev),name='edge_init')
    edge_feat_init = tf.einsum('ivlk,kj->ivlj',edge_feat, edge_init)
    node_feat_init = tf.einsum('ivk,kj->ivj',node_feat,node_init)
    
    return node_feat_init, edge_feat_init

def ggcn_layer(node_feat, adj, w, p, initialization_stddev,adjust_matrix,edge_feat,
           layer_num, scope):
    
#     此为一层的ggcn_layer网络，由于每一层需要的参数不同，所以用layer_num指示layer，用scope指示double dqn
#     一次迭代返回node_feat和edge_feat
#     由于不是完全图不能使用batch normalization，并且GGCN的原论文也没有使用BN，此处采取多乘一次参数处理。
    
    w1 = tf.Variable(tf.random_normal([p,p],stddev=initialization_stddev),name='w1'+str(layer_num))
    w2 = tf.Variable(tf.random_normal([p,p],stddev=initialization_stddev),name='w2'+str(layer_num))
    w3 = tf.Variable(tf.random_normal([p,p],stddev=initialization_stddev),name='w3'+str(layer_num))
    w4 = tf.Variable(tf.random_normal([p,p],stddev=initialization_stddev),name='w4'+str(layer_num))
    w5 = tf.Variable(tf.random_normal([p,p],stddev=initialization_stddev),name='w5'+str(layer_num))
    b1 = tf.Variable(tf.zeros([p]),name='b1'+str(layer_num))
    b2 = tf.Variable(tf.zeros([p]),name='b2'+str(layer_num))
    b3 = tf.Variable(tf.zeros([p]),name='b3'+str(layer_num))
    b4 = tf.Variable(tf.zeros([p]),name='b4'+str(layer_num))
    b5 = tf.Variable(tf.zeros([p]),name='b5'+str(layer_num))#GGCN源码中是有偏置的，这里偏置处理完之后需要对边和点特征进行调整，不存在的边和点embedding应该调整成0
    #此处沿用GGCN论文中的参数符号，注意GGCN代码中w4和w5是一样的，但是此处不一样。
    #因为每个边都有两个embedding，对于eij来说两个顶点是不一样的。实际上认为应该无影响。
    bn1 = tf.Variable(tf.random_normal([p,p],stddev=initialization_stddev),name='bn1'+str(layer_num))
    bn2 = tf.Variable(tf.random_normal([p,p],stddev=initialization_stddev),name='bn2'+str(layer_num))
    bn1_b = tf.Variable(tf.zeros([p]),name='bn1_b'+str(layer_num))
    bn2_b = tf.Variable(tf.zeros([p]),name='bn2_b'+str(layer_num))
    #bn1，bn2为代替batch normalization多乘的参数，同样是有偏置的
    e_in = edge_feat
    x_in =node_feat
    e_tmp = (tf.einsum('ivlk,kj->ivlj',e_in,w3) + b3) + tf.expand_dims(tf.einsum('ivk,kj->ivj',x_in,w4)+b4,axis=2) + tf.expand_dims(tf.einsum('ivk,kj->ivj',x_in,w5)+b5,axis=1)
    # 计算edge_feat进行BN之前的操作，(batch_size,n,n,p) +(batch_size,n,1,p)+(batch_size,1,n,p)
    # (batch_size,n,n,p) +(batch_size,n,1,p)会使每个(n,p)加上(1,p)，即为第i行的edge_feat都加上node i的feat
    # (batch_size,n,n,p) +(batch_size,1,n,p)会使每个(n,n,p)加上(1,n,p)，即为第i行的每一维分别加上对应node的feat
    e_bn = tf.nn.relu(tf.einsum('ivlk,kj->ivlj',e_tmp,bn2)+bn2_b)
    e = e_in + e_bn
    #先计算好edge_feat，再计算node_feat
    #TODO
    e = tf.expand_dims(adj,axis=-1)*e
    #此处缺少将edge_feat中去除不存在的边，因为有偏置的存在所以是必须的
    edge_gate = tf.nn.sigmoid(e)#element-wise的sigmoid处理，与GGCN一致
    edge_gate = tf.expand_dims(adj,axis=-1)*edge_gate
    gatevx = tf.expand_dims(tf.einsum('ivk,kj->ivj',x_in,w2) + b2,axis=1)*edge_gate
    #(batch_size,1,n,p)和(batch_size,n,n,p)相乘，此处计算好了分子上的部分
    x_new = (tf.einsum('ivk,kj->ivj',x_in,w1) + b1) + tf.einsum('ivjk->ivk',gatevx)/(1e-20+tf.einsum('ivjk->ivk',edge_gate))
    x_bn = tf.nn.relu(tf.einsum('ivk,kj->ivj',x_new,bn1)+bn1_b)
    x = x_in + x_bn
    #TODO
    x = tf.einsum('ivk,iuv->iuk',x,adjust_matrix)
    #缺少将node_feat中去除不存在的点
    return x, e

def ggcn(node_feat, adj, w, p, initialization_stddev,adjust_matrix,multiply_matrix,edge_feat,
           layer_num, scope):
    x, e = init_layer(node_feat, adj, w, p,initialization_stddev,edge_feat,scope)
    for i in range(layer_num):
        x, e = ggcn_layer(x, adj, w, p, initialization_stddev, adjust_matrix, e, i, scope)
    return x, e

def q_func(node_feat, adj, w, p, initialization_stddev,adjust_matrix,multiply_matrix,edge_feat,
           layer_num,scope):
    #得到点和边的embedding之后，直接计算Q函数，沿用S2V代码的Q函数计算方法
    reg_hidden = 100
    with tf.variable_scope(scope):
        x, e = ggcn(node_feat, adj, w, p, initialization_stddev,adjust_matrix,multiply_matrix,edge_feat,
            layer_num, scope)
        h1_weight = tf.Variable(tf.random_normal([2*p, reg_hidden], stddev=initialization_stddev), name='h1_weight')
        h1_weight1 = tf.Variable(tf.random_normal([reg_hidden, reg_hidden], stddev=initialization_stddev), name='h1_weight1')
        h1_weight2 = tf.Variable(tf.random_normal([reg_hidden, reg_hidden], stddev=initialization_stddev), name='h1_weight2')
        h1_weight3 = tf.Variable(tf.random_normal([reg_hidden, reg_hidden], stddev=initialization_stddev), name='h1_weight3')
        b1_weight1 = tf.Variable(tf.random_normal([reg_hidden], stddev=initialization_stddev), name='b1_weight1')
        b1_weight2 = tf.Variable(tf.random_normal([reg_hidden], stddev=initialization_stddev), name='b1_weight2')
        b1_weight3 = tf.Variable(tf.random_normal([reg_hidden], stddev=initialization_stddev), name='b1_weight3')

        h2_weight = tf.Variable(tf.random_normal([reg_hidden ], stddev=initialization_stddev), name='h2_weight')
        y_potential = tf.einsum('ivu,iuk->ivk',multiply_matrix, x)
        embed_s_a = tf.concat([y_potential, x],axis=2)

        hidden = tf.einsum('ivk,kj->ivj', embed_s_a, h1_weight)
        hidden1 = tf.einsum('ivk,kj->ivj', hidden, h1_weight1)+b1_weight1
        hidden2 = tf.einsum('ivk,kj->ivj', hidden1, h1_weight2)+b1_weight2
        hidden3 = tf.einsum('ivk,kj->ivj', hidden2, h1_weight3)+b1_weight3
        last_output = tf.nn.relu(hidden3)
        q = tf.einsum('ivj,j->iv', last_output, h2_weight)

    return tf.identity(q)

