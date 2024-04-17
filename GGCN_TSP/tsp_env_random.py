import numpy as np
import networkx as nx
from tsp_solver.greedy import solve_tsp
import random

class tsp_env_random:
    def __init__(self, number_nodes_min, number_nodes_max):
        self.number_nodes = number_nodes_max
        self.number_nodes_min = number_nodes_min
        self.num_actions = self.number_nodes
        self.state_shape = [self.number_nodes]
        self.env_name = 'tsp_random'

    def reset(self):
        self.acc_reward = 0
        self.real_nodes = random.randint(self.number_nodes_min, self.number_nodes)
        self.x_coor = np.zeros(self.real_nodes)
        self.y_coor = np.zeros(self.real_nodes)
        for i in range(self.real_nodes):
            self.x_coor[i] = random.uniform(0,1)
            self.y_coor[i] = random.uniform(0,1)
        self.state = np.array([0]*self.real_nodes+[1]*(self.number_nodes-self.real_nodes))
        self.state[0] = 1
        self.real_state = np.zeros(self.real_nodes)
        self.real_state[0] = 1

        self.ordered_action = [0]#强制以0为起点

        self.adjacency_matrix = np.zeros([self.num_actions, self.num_actions])
        self.weight_matrix = np.zeros([self.num_actions, self.num_actions])
        for i in range(self.real_nodes):
            for j in range(self.real_nodes):
                self.weight_matrix[i][j] = np.power((self.x_coor[i]-self.x_coor[j])*(self.x_coor[i]-self.x_coor[j]) + (self.y_coor[i]-self.y_coor[j])*(self.y_coor[i]-self.y_coor[j]),0.5)
        self.knn_weight = self.weight_matrix.copy()
        self.knn_update(2)

        self.adjust_matrix = np.pad(np.eye(self.real_nodes,dtype=np.float32),((0,self.num_actions-self.real_nodes),(0,self.num_actions-self.real_nodes)),'constant',constant_values=(0,0))
        self.edge_feat = np.array([[[0,0,0,0]]*self.num_actions]*self.num_actions,dtype=np.float32)
        self.edge_feat_update()
        self.node_feat = np.array([[0,0,0,0]]*self.num_actions,dtype=np.float32)
        self.node_feat_update()

        return self.real_nodes, self.state,self.node_feat, self.adjacency_matrix, self.weight_matrix,self.adjust_matrix,self.edge_feat
    
    def is_done(self, state):
        done = True
        for node in range(self.real_nodes):
            if state[node] == 0:
                done = False
        return done
    
    def knn_update(self,k):#构建k nearest graph，不考虑可能变成不连通图的情况
        for node_count1 in range(self.real_nodes):
            sort = self.weight_matrix[node_count1].argsort()[-k:]
            for sort_idx in sort:
                self.adjacency_matrix[node_count1][sort_idx] = 1
                self.adjacency_matrix[sort_idx][node_count1] = 1
            self.adjacency_matrix[node_count1][node_count1] = 0
        self.knn_weight = np.multiply(self.adjacency_matrix, self.weight_matrix)


    def node_feat_update(self):
        for node_count1 in range(self.real_nodes):
            self.node_feat[node_count1] = np.array([self.x_coor[node_count1], self.y_coor[node_count1], self.state[node_count1], 1])
    
    def edge_feat_update(self):
        for node_count1 in range(self.real_nodes):
            for node_count2 in range(self.real_nodes):
                if self.adjacency_matrix[node_count1][node_count2] != 0:
                    same_sign = 0
                    if self.state[node_count1] != self.state[node_count2]:
                        same_sign = 1.0
                    self.edge_feat[node_count1][node_count2] = np.array([self.state[node_count1],self.weight_matrix[node_count1][node_count2],same_sign,1])
            self.edge_feat[node_count1][node_count1] = np.array([0,0,0,0],dtype=np.float32)
        #只保留k_nearest图的边的embedding


    def distance(self,ordered_list):
        distance = 0
        for i in range(0, len(ordered_list)):
            distance = distance + self.weight_matrix[ordered_list[i-1]][ordered_list[i]]
        return distance

    def step(self, action):
        dis_change = []
        
        for i in range(0,len(self.ordered_action)):
            dis_change.append(self.weight_matrix[self.ordered_action[i]][self.ordered_action[i-1]]-self.weight_matrix[self.ordered_action[i]][action]-self.weight_matrix[self.ordered_action[i-1]][action])

        index = np.argmax(dis_change)#增量最小的点添加处
        if index == 0:
            index = -1
        
        old_order_action = self.ordered_action.copy()
        self.ordered_action.insert(index, action)

        self.state[action] = 1
        self.real_state[action] = 1
        
        self.node_feat_update()
        self.edge_feat_update()

        rew = self.distance(old_order_action)-self.distance(self.ordered_action)
        rew = rew/self.num_actions
        #reward为负增量（鼓励选择增量小的点添加）
        return self.state,self.node_feat, rew, self.is_done(self.state), self.adjacency_matrix, self.weight_matrix,self.adjust_matrix,self.edge_feat
    
    def accumulated_reward(self):
        return 0

    def optimal_solution(self):
        solution = solve_tsp(self.weight_matrix)
        return self.distance(solution), solution