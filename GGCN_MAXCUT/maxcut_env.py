import numpy as np
import networkx as nx
import maxcut_docplex
import random

class maxcut_env:
    def __init__(self, number_nodes_min, number_nodes_max, p=0.15, replay_penalty=-1000):
        self.number_nodes = number_nodes_max
        self.number_nodes_min = number_nodes_min
        self.p = p
        self.state_shape = [self.number_nodes]
        self.num_actions = self.number_nodes
        self.env_name = 'maxcut'
        self.replay_penalty = replay_penalty
    
    def reset(self):
        self.max_rew = 0
        self.acc_reward = 0
        self.real_nodes = random.randint(self.number_nodes_min, self.number_nodes)
        self.graph = nx.erdos_renyi_graph(n = self.real_nodes, p = self.p)
        self.nodes = list(self.graph.nodes)+[0]*(self.number_nodes-self.real_nodes)
        self.edges = list(self.graph.edges)
        self.state = np.array([0]*self.real_nodes+[1]*(self.number_nodes-self.real_nodes))
        self.real_state = np.zeros(self.number_nodes)
        self.adjacency_matrix = nx.to_numpy_matrix(self.graph)#返回邻接矩阵，每个元素值为0或1
        self.adjacency_matrix = np.pad(self.adjacency_matrix, ((0,self.number_nodes-self.real_nodes),(0,self.number_nodes-self.real_nodes)),'constant',constant_values=(0,0))
        self.weights = [random.random() for edge in self.edges]
        self.weight_matrix = np.zeros((self.number_nodes, self.number_nodes))
        for edge_idx in range(len(self.edges)):
            self.weight_matrix[self.edges[edge_idx][0]][self.edges[edge_idx][1]] = self.weights[edge_idx]
            self.weight_matrix[self.edges[edge_idx][1]][self.edges[edge_idx][0]] = self.weights[edge_idx]
        self.adjust_matrix = np.pad(np.eye(self.real_nodes,dtype=np.float32),((0,self.num_actions-self.real_nodes),(0,self.num_actions-self.real_nodes)),'constant',constant_values=(0,0))
        self.edge_feat = np.array([[[0,0,0,0]]*self.num_actions]*self.num_actions,dtype=np.float32)
        self.edge_feat_update()
        self.node_feat = np.array([[0,0]]*self.num_actions,dtype=np.float32)
        self.node_feat_update()

        self.sum_weight = 0
        for weight in self.weights:
            self.sum_weight += weight
        self.aux_init = np.array([[0,0,0,1]]*self.num_actions)
        
        return self.real_nodes, self.state,self.node_feat, self.adjacency_matrix, self.weight_matrix,self.adjust_matrix,self.edge_feat,self.aux_init
    
    def is_done(self, state):
        done = True
        cost1 = self.cost(state)
        for node in range(self.real_nodes):
            if state[node] == 0:
                # temp_state = state.copy()
                # temp_state[node] = 1
                # cost2 = self.cost(temp_state)
                # if cost1 < cost2:
                done = False
        return done
    
    def cost(self,state):
        cut_cost = 0
        for edge_idx in range(len(self.edges)):
            if state[self.edges[edge_idx][0]] != state[self.edges[edge_idx][1]]:
                cut_cost += self.weights[edge_idx]
        return cut_cost
    
    def step_rew(self, a):#假设采取行动a时的奖励
        temp_state = self.state.copy()
        temp_state[a] = 1
        return (self.cost(temp_state)-self.cost(self.state))/self.num_actions
    
    def node_feat_update(self):
        for node_count1 in range(self.real_nodes):
            self.node_feat[node_count1][0] = 1
            if self.state[node_count1] == 1:
                self.node_feat[node_count1][1] = 0
            if self.state[node_count1] == 0:
                self.node_feat[node_count1][1] = 1
                # self.node_feat[node_count1][2] = self.step_rew(node_count1)

    def edge_feat_update(self):
        for node_count1 in range(self.real_nodes):
            for node_count2 in range(self.real_nodes):
                if self.weight_matrix[node_count1][node_count2] != 0:
                    same_sign = 0
                    if self.state[node_count1] != self.state[node_count2]:
                        same_sign = 1.0
                    self.edge_feat[node_count1][node_count2] = np.array([self.state[node_count1],self.weight_matrix[node_count1][node_count2],same_sign,1])

    def aux(self, state):
        node_cover = 0
        for node_count in range(self.real_nodes):
            if state[node_count] ==1:
                node_cover += 1
        edge_cover = 0
        for edge_count in self.edges:
            if state[edge_count[0]] == 1 or state[edge_count[1]] == 1:
                edge_cover += 1
        return np.array([[0,0,0,1]]*self.num_actions)
    #np.array([[node_cover/self.real_nodes, edge_cover/len(self.edges),self.cost(state)/self.sum_weight,1]]*self.number_nodes)

    def step(self, action):
        if self.state[action] != 1:#每次action对state数组进行修改，要修改的位置为0时改为1
            old_state = self.state.copy()
            self.state[action] = 1
            self.real_state[action] = 1
            rew = self.cost(self.state) - self.cost(old_state)
        else:
            rew = -self.replay_penalty
        
        rew = rew/self.num_actions
        self.acc_reward = self.cost(self.state)
        if self.acc_reward > self.max_rew:
            self.max_rew = self.acc_reward

        self.node_feat_update()
        self.edge_feat_update()

        return self.state,self.node_feat, rew, self.is_done(self.state), self.adjacency_matrix, self.weight_matrix,self.adjust_matrix,self.edge_feat,self.aux(self.state)
    
    def max_accumulated_reward(self):
        return self.max_rew
    
    def accumulated_reward(self):
        return 0
    
    def get_e2n_param(self):#得到用来edge 2 node用来平均的参数，n维nparray每一维为对应点所连边数的倒数
        e2n_param = np.zeros(self.number_nodes)
        adjsum = np.sum(self.adjacency_matrix,axis=1)
        for node_count in range(self.real_nodes):
            if e2n_param[node_count] != 0:
                e2n_param[node_count] = 1/adjsum[node_count]
        return e2n_param

    def optimal_solution(self):
        solution = maxcut_docplex.maxcut_opt(self.real_nodes, self.edges, self.weights)
        return self.cost(solution), solution


    

            