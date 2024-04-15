import random
import numpy as np
import tensorflow as tf

import dqn_graph_model as dqn
from dqn_utils import *
import ggcn as Q_function_graph_model
import maxcut_env

env = maxcut_env.maxcut_env(10,10)
dqn.learn(
    env,
    q_func=Q_function_graph_model.q_func,
    pre_pooling_mlp_layers=0,
    post_pooling_mlp_layers=0,
    n_hidden_units=64, T=3,
    initialization_stddev=1e-3,
    exploration=LinearSchedule(100000,0.05,1),
    stopping_criterion=None,
    replay_buffer_size=100000,
    batch_size=128,
    gamma=1.,
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