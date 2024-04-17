import random
import numpy as np
import tensorflow as tf

import dqn_graph_model as dqn
from dqn_utils import *
import ggcn
import tsp_env_random

env = tsp_env_random.tsp_env_random(30,30)
dqn.learn(
    env,
    q_func=ggcn.q_func,
    pre_pooling_mlp_layers=0,
    post_pooling_mlp_layers=0,
    n_hidden_units=64, T=4,
    initialization_stddev=1e-2,
    exploration=LinearSchedule(100000,0.05,1),
    stopping_criterion=None,
    replay_buffer_size=100000,
    batch_size=64,
    gamma=0.1,
    learning_starts=2000,
    learning_freq=4,
    target_update_freq=2000,
    grad_norm_clipping=10,
    double_DQN=True,
    n_steps_ahead=1,
    learning_rate_start=1e-4,
    pre_training=None,
    test_mode= False,
    one_graph_test= False
)