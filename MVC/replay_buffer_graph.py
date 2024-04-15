import numpy as np


class ReplayBuffer:

    def __init__(self, size, obs_size, n_nodes):#n_nodes应该是max值
        self.size = size
        self.obs = np.zeros([self.size, obs_size], dtype=np.float32)#生成2维的np列表，size实际为1000000，obs_size为每个样本的obs维数
        self.adj = np.zeros([self.size, n_nodes, n_nodes], dtype=np.float32)
        self.weight_matrix = np.zeros([self.size, n_nodes, n_nodes], dtype=np.float32)
        self.next_obs = np.zeros([self.size, obs_size], dtype=np.float32)
        self.actions = np.zeros([self.size], dtype=np.int32)
        self.n_q_values = np.zeros([self.size],dtype=np.float32)#用于储存n步之后的最大的q函数值
        self.n_acts = np.zeros([self.size],dtype=np.int32)
        self.rewards = np.zeros([self.size], dtype=np.float32)
        self.done = np.zeros([self.size], dtype=np.bool)
        self.transition_length = np.zeros([self.size], dtype=np.int32)
        self.true_nodes = np.zeros([self.size],dtype=np.int32)#用于指示buffer内每个样本中实际的点的个数
        self.adjust_matrix = np.zeros([self.size, n_nodes, n_nodes], dtype=np.float32)
        self.aux = np.zeros([self.size, n_nodes, 3], dtype=np.float32)

        self.num_in_buffer = 0
        self.next_idx = 0


    def store_transition(self, obs, adj, weight_matrix, action, n_q_value, n_act, reward, next_obs, done, transition_length, true_node, adjust_matrix, aux):
        self.obs[self.next_idx] = obs#刚初始化时将self的第一维改为相应的输入
        self.adj[self.next_idx] = adj
        self.weight_matrix[self.next_idx] = weight_matrix
        self.actions[self.next_idx] = action
        self.n_q_values[self.next_idx] = n_q_value
        self.n_acts[self.next_idx] = n_act
        self.rewards[self.next_idx] = reward
        self.next_obs[self.next_idx] = next_obs
        self.done[self.next_idx] = done
        self.transition_length[self.next_idx] = transition_length
        self.true_nodes[self.next_idx] = true_node
        self.adjust_matrix[self.next_idx] = adjust_matrix
        self.aux[self.next_idx] = aux

        self.num_in_buffer = min(self.size, self.num_in_buffer + 1)
        self.next_idx = (self.next_idx + 1) % self.size#初始化之后每一次调用此函数self.next_idx加1，self.num_in_buffer也加1

    def can_sample(self, batch_size):
        """Returns true if `batch_size` different transitions
         can be sampled from the buffer."""
        return batch_size + 1 <= self.num_in_buffer

    def sample(self, batch_size):
        assert self.can_sample(batch_size)#如果self.can_sample结果为True，即为batch_size + 1 <= self.num_in_buffer
        idxes = np.random.choice(self.num_in_buffer, batch_size)#选择batch_size个数的从0到self.num_in_buffer的随机数组成nparray
        #观察到obs对应位置的状态，就采取actions对应的行动，获得rewards对应的回报，之后状态转变为next_obs对应位置的状态

        return self.obs[idxes], \
               self.adj[idxes], \
               self.weight_matrix[idxes], \
               self.actions[idxes], \
               self.n_q_values[idxes],\
               self.n_acts[idxes],\
               self.rewards[idxes], \
               self.next_obs[idxes], \
               1 - np.array(self.done[idxes], dtype=np.int32), \
               self.transition_length[idxes],\
               self.true_nodes[idxes],\
               self.adjust_matrix[idxes],\
               self.aux[idxes]
                #因为这里是nparray，idxes为nparray，self.obs[idxes]即为取self.obs的idxes对应维组成新的nparray
