import numpy as np
import networkx as nx
import mvc_docplex
import random

class MVC_env:
    def __init__(self, number_nodes_min,number_nodes_max, p = 0.15, replay_penalty=-1000):
        self.number_nodes = number_nodes_max#节点数取最大值，不够维数的使用np.pad来补齐
        self.number_nodes_min = number_nodes_min
        self.p = p
        self.state_shape = [self.number_nodes]
        self.num_actions = self.number_nodes
        self.env_name = 'MVC'
        self.replay_penalty = replay_penalty

    def reset(self):
        self.acc_reward = 0
        self.real_nodes = random.randint(self.number_nodes_min, self.number_nodes)
        self.graph = nx.erdos_renyi_graph(n = self.real_nodes, p = self.p)
        self.nodes = list(self.graph.nodes)+[0]*(self.number_nodes-self.real_nodes)
        self.edges = list(self.graph.edges)
        self.state = np.array([0]*self.real_nodes+[1]*(self.number_nodes-self.real_nodes))
        self.real_state = np.zeros(self.number_nodes)
        self.adjacency_matrix = nx.to_numpy_matrix(self.graph)#返回邻接矩阵，每个元素值为0或1
        self.adjacency_matrix = np.pad(self.adjacency_matrix, ((0,self.number_nodes-self.real_nodes),(0,self.number_nodes-self.real_nodes)),'constant',constant_values=(0,0))
        #用0补齐矩阵到最大可能点数
        self.weight_matrix = self.adjacency_matrix
        self.adjust_matrix = np.pad(np.eye(self.real_nodes,dtype=np.float32),((0,self.num_actions-self.real_nodes),(0,self.num_actions-self.real_nodes)),'constant',constant_values=(0,0))
        #用于保证计算时只计算子图的embedding
        self.aux_init = np.array([[0,0,1]]*self.num_actions)

        if len(self.edges) == 0:
            self.reset()
        return self.real_nodes, self.state, self.adjacency_matrix, self.weight_matrix, self.adjust_matrix,self.aux_init
    
    def one_graph_reset(self):
        self.acc_reward = 0
        self.state = np.array([0]*self.real_nodes+[1]*(self.number_nodes-self.real_nodes))
        self.real_state = np.zeros(self.number_nodes)
        return self.real_nodes, self.state, self.adjacency_matrix, self.weight_matrix, self.adjust_matrix

    def is_done(self, state):
        done = True
        for e in self.edges:
            if state[e[0]] == 0 and state[e[1]] == 0:
                done = False#对于self.edges中的每条边，如果有一条边两个顶点state都是0，则为False，即为当每条边两个顶点都是一个0一个1时为True

        return done

    def aux(self, state):#根据状态生成辅助向量矩阵
        node_cover = 0
        for node_count in range(self.real_nodes):
            if state[node_count] ==1:
                node_cover += 1
        edge_cover = 0
        for edge_count in self.edges:
            if state[edge_count[0]] == 1 or state[edge_count[1]] == 1:
                edge_cover += 1
        return np.array([[node_cover/self.real_nodes, edge_cover/len(self.edges), 1]]*self.number_nodes)
    
    def adjust(self, state):#根据状态生成调整矩阵
        temp_adjust = np.pad(np.eye(self.real_nodes,dtype=np.float32),((0,self.number_nodes-self.real_nodes),(0,self.number_nodes-self.real_nodes)),'constant',constant_values=(0,0))
        for node_count in range(self.real_nodes):
            if state[node_count] == 1:
                temp_adjust[node_count][node_count] = 0
        return temp_adjust

    def adj(self, state):
        new_adj = self.adjacency_matrix.copy()
        for node_count in range(self.real_nodes):
            if state[node_count] == 1:
                for k in range(self.real_nodes):
                    new_adj[node_count][k] = 0
                    new_adj[k][node_count] = 0
        return new_adj
    
    def weight(self, state):
        new_weight = self.weight_matrix.copy()
        for node_count in range(self.real_nodes):
            if state[node_count] == 1:
                for k in range(self.real_nodes):
                    new_weight[node_count][k] = 0
                    new_weight[k][node_count] = 0
        return new_weight

    def step(self, action):
        if self.state[action] != 1:#每次action对state数组进行修改，要修改的位置为0时改为1
            self.state[action] = 1
            self.real_state[action] = 1
            rew=-1/self.num_actions
            if self.is_done(self.state):
                # rew = self.number_nodes_min
                rew = -1/self.num_actions
        else:
            rew = -self.replay_penalty

        self.acc_reward += rew

        return self.state, rew, self.is_done(self.state), self.adj(self.state), self.weight(self.state), self.adjust(self.state), self.aux(self.state)

    def accumulated_reward(self):
        return self.acc_reward

    def at_random_solution(self):#用于生成随机方案
        temp_state = np.zeros(self.number_nodes)
        while not self.is_done(temp_state):#当self.is_done值为False时进行循环
            temp_state[np.random.randint(self.number_nodes)] = 1
            #np.random.randint用于生成随机整数，np.random.randint(self.number_nodes)生成从0到self.number_nodes的随机整数
        return -np.sum(temp_state), temp_state

    def optimal_solution(self):
        solution = mvc_docplex.mvc_opt(self.real_nodes, self.edges)
        opt_rew = 0
        for count in solution:
            if count == 1:
                opt_rew -= 1
        return opt_rew, solution
    