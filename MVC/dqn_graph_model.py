import sys
import gym.spaces
import itertools
import numpy as np
import random
import logz
import tensorflow                as tf
import tensorflow.contrib.layers as layers
from collections import namedtuple
from dqn_utils import *
import replay_buffer_graph
import time
import Q_function_graph_model2 as Q_function_graph_model
import mvc_env

OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs", "lr_schedule"])

"""
learn slightly modified to pass the task name as an argument
so that it is easier to record data.
"""

def learn(env,
          q_func,
          target_q_func,
          pre_pooling_mlp_layers,
          post_pooling_mlp_layers,
          n_hidden_units,
          T=4,
          initialization_stddev=1e-4,
          exploration=LinearSchedule(1000000, 0.1),
          stopping_criterion=None,
          replay_buffer_size=1000000,
          batch_size=32,
          gamma=0.99,
          learning_starts=50000,
          learning_freq=4,
          frame_history_len=4,
          target_update_freq=10000,
          grad_norm_clipping=10,
          double_DQN=True,
          n_steps_ahead=3,
          learning_rate_start=1e-3,
          pre_training=None,
          one_graph_test=False
         ):
    
    """Run Deep Q-learning algorithm.

    You can specify your own convnet using q_func.

    All schedules are w.r.t. total number of steps taken in the environment.

    Parameters
    ----------
    env: gym.Env
        gym environment to train on.
    q_func: function
        Model to use for computing the q function. It should accept the
        following named arguments:
            img_in: tf.Tensor
                tensorflow tensor representing the input image
            num_actions: int
                number of actions
            scope: str
                scope in which all the model related variables
                should be created
            reuse: bool
                whether previously created variables should be reused.
    optimizer_spec: OptimizerSpec
        Specifying the constructor and kwargs, as well as learning rate schedule
        for the optimizer
    session: tf.Session
        tensorflow session to use.
    exploration: rl_algs.deepq.utils.schedules.Schedule
        schedule for probability of chosing random action.
    stopping_criterion: (env, t) -> bool
        should return true when it's ok for the RL algorithm to stop.
        takes in env and the number of steps executed so far.
    replay_buffer_size: int
        How many memories to store in the replay buffer.
    batch_size: int
        How many transitions to sample each time experience is replayed.
    gamma: float
        Discount Factor
    learning_starts: int
        After how many environment steps to start replaying experiences
    learning_freq: int
        How many steps of environment to take between every experience replay
    frame_history_len: int
        How many past frames to include as input to the model.
    target_update_freq: int
        How many experience replay rounds (not steps!) to perform between
        each update to the target Q network
    grad_norm_clipping: float or None
        If not None gradients' norms are clipped to this value.
    """
    exp_name = env.env_name
    logz.configure_output_dir('/data/' + exp_name + time.strftime('%Y-%m-%d-%H-%M-%S'))
    ###############
    # BUILD MODEL #
    ###############

    input_shape = env.state_shape
    num_actions = env.num_actions


    num_min = env.number_nodes_min
    test_env = mvc_env.MVC_env(num_min,num_actions)
    # filename = "test2 10-15.txt"
    filename = env.env_name + str(env.number_nodes_min) + '-' + str(env.number_nodes) + \
                    'buffer' + str(replay_buffer_size) + '-' +\
                        "target_update_freq"+str(target_update_freq)+\
                            '-'+ time.strftime('%m-%d-%Y-%H-%M-%S')+"-"+"doubleDQN="+ str(double_DQN) +".txt"

    # set up placeholders
    # placeholder for current observation (or state)
    obs_t_ph              = tf.placeholder(tf.float32, [None] + list(input_shape))#占位符，tf.placeholer(dtype, shape=None, name=None),shape为[None]表示形状不指定
    # placeholder for current action
    act_t_ph              = tf.placeholder(tf.int32, [None], name='act_t_ph')
    n_act_ph              = tf.placeholder(tf.int32, [None])
    # placeholder for current reward
    rew_t_ph              = tf.placeholder(tf.float32, [None])
    # placeholder for next observation (or state)
    obs_tp1_ph            = tf.placeholder(tf.float32, [None] + list(input_shape))
    # placeholder for end of episode mask
    # this value is 1 if the next state corresponds to the end of an episode,
    # in which case there is no Q-value at the next state; at the end of an
    # episode, only the current state reward contributes to the target, not the
    # next state Q-value (i.e. target is just rew_t_ph, not rew_t_ph + gamma * q_tp1)
    done_mask_ph          = tf.placeholder(tf.float32, [None])#值为0时代表此episode结束，Q函数的值为当前步可产生的reward
    transition_length_ph = tf.placeholder(tf.float32, [None])
    adjust_matrix_ph = tf.placeholder(tf.float32,[None, env.number_nodes, env.number_nodes])
    multiply_matrix_ph = tf.placeholder(tf.float32,[None, env.number_nodes, env.number_nodes])
    multiply_matrix = np.ones((batch_size, env.number_nodes, env.number_nodes),dtype=np.float32)
    multiply_choose = np.ones((num_actions,num_actions),dtype=np.float32)
    aux_ph = tf.placeholder(tf.float32, [None, num_actions, 3])

    # Graphs specific placeholder
    adj_ph = tf.placeholder(tf.float32, [None, env.number_nodes, env.number_nodes],
                            name='adj_ph')
    graph_weights_ph = tf.placeholder(tf.float32,
                                      [None, env.number_nodes, env.number_nodes],
                                      name='graph_weights_ph')

    with tf.variable_scope('q_eval') as scope_eval:
        q_func_net = q_func(x=obs_t_ph,
                            adj=adj_ph,
                            w=graph_weights_ph,
                            p=n_hidden_units, T=T, initialization_stddev=initialization_stddev, adjust_matrix=adjust_matrix_ph,multiply_matrix=multiply_matrix_ph,
                            aux=aux_ph,
                            scope="q_func", reuse=True,
                            pre_pooling_mlp_layers=pre_pooling_mlp_layers,
                            post_pooling_mlp_layers=post_pooling_mlp_layers)
        scope_eval.reuse_variables()
        q_func_net_argmax_target = q_func(x=obs_tp1_ph,
                                        adj=adj_ph,
                                        w=graph_weights_ph,
                                        p=n_hidden_units, T=T, initialization_stddev=initialization_stddev,adjust_matrix=adjust_matrix_ph,multiply_matrix=multiply_matrix_ph,
                                        aux=aux_ph,
                                        scope="q_func", reuse=True,
                                        pre_pooling_mlp_layers=pre_pooling_mlp_layers,
                                        post_pooling_mlp_layers=post_pooling_mlp_layers)
    target_q_func_net = target_q_func(x=obs_tp1_ph,
                               adj=adj_ph,
                               w=graph_weights_ph,
                               p=n_hidden_units, T=T, initialization_stddev=initialization_stddev,adjust_matrix=adjust_matrix_ph,multiply_matrix=multiply_matrix_ph,
                               aux=aux_ph,
                               scope="target_q_func", reuse=False,
                               pre_pooling_mlp_layers=pre_pooling_mlp_layers,
                               post_pooling_mlp_layers=post_pooling_mlp_layers)

    q_target_ph = tf.placeholder(tf.float32,[None])
    if not double_DQN:
        target_y = rew_t_ph + tf.pow(gamma, transition_length_ph) *\
                              done_mask_ph * tf.reduce_sum(q_func_net_argmax_target *\
                                 tf.one_hot(tf.argmax(q_func_net_argmax_target, axis = 1),
                                            depth=num_actions),\
                                 axis=1)
    else:
        target_y = rew_t_ph + tf.pow(gamma,transition_length_ph)*done_mask_ph*q_target_ph

    actual_y = tf.reduce_sum(tf.multiply(q_func_net, tf.one_hot(act_t_ph, depth=num_actions)), axis=1)
    total_error =tf.reduce_mean(tf.squared_difference(target_y, actual_y))
    q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                    scope='q_func')
    target_q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                           scope='target_q_func')

    training_error_summ_sy = tf.summary.scalar('training_total_error', total_error)

    # construct optimization op (with gradient clipping)
    # learning_rate_ph = tf.placeholder(tf.float32, (), name="learning_rate")
    global_step = tf.placeholder(tf.int32,(),name='global_step')
    learning_rate = tf.train.exponential_decay(learning_rate=learning_rate_start, global_step=global_step, decay_steps=2000, decay_rate=0.95, staircase=True)
    optimizer = tf.train.RMSPropOptimizer(learning_rate_start)
    train_fn = optimizer.minimize(total_error)

    # update_target_fn will be called periodically to copy Q network to target Q network
    update_target_fn = []
    for var, var_target in zip(sorted(q_func_vars,        key=lambda v: v.name),
                               sorted(target_q_func_vars, key=lambda v: v.name)):
        update_target_fn.append(var_target.assign(var))
    update_target_fn = tf.group(*update_target_fn)

    # construct the replay buffer
    replay_buffer = replay_buffer_graph.ReplayBuffer(replay_buffer_size, obs_size=input_shape[0],
                                                     n_nodes=input_shape[0])

    # Model saver
    saver = tf.train.Saver()

    # Create session, initialize variables
    session = tf.InteractiveSession()

    log_files_name = env.env_name + str(env.number_nodes_min) + '-' + str(env.number_nodes) + \
                    'buffer' + str(replay_buffer_size) + '-'+ time.strftime('%m-%d-%Y-%H-%M-%S')+"-"+"doubleDQN="+ str(double_DQN) 

    writer = tf.summary.FileWriter('/tmp/' + log_files_name,
                                   session.graph)
    if pre_training ==None:
        tf.global_variables_initializer().run()
    else:
        saver.restore(session, '/tmp/saved_models/'+pre_training )

    ###############
    # RUN ENV     #
    ###############
    model_initialized = False
    if pre_training != None:
        model_initialized = True
    num_param_updates = 0
    mean_episode_reward      = -float('nan')
    best_mean_episode_reward = -float('inf')
    real_nodes, temp_observations, adj_choose, w_choose, adjust_choose, aux_choose = env.reset()
    observations  = [temp_observations]
    adjs = [adj_choose]
    weights = [w_choose]
    adjusts = [adjust_choose]
    auxs = [aux_choose]

    LOG_EVERY_N_STEPS = 10000

    episode_total_rewards = []
    episode_total_optimal_rewards = []
    episode_total_at_random_rewards = []
    accuracies = []
    done = False
    actions_list = []
    rewards_list = [0]

    for t in itertools.count():#从0开始步长为1的无限序列
        ### 1. Check stopping criterion
        if stopping_criterion is not None and stopping_criterion(env, t):
            break

        ### 2. Step the env and store the transition
        import random
        from numpy import array

        if done:
            if one_graph_test != True:
                real_nodes, temp_observations, adj_choose, w_choose, adjust_choose, aux_choose = env.reset()
            else:
                real_nodes, temp_observations, adj_choose, w_choose, adjust_choose, aux_choose = env.one_graph_reset()
            observations = [temp_observations]
            actions_list = []
            rewards_list = [0]#rewards_list用于储存每一步的accumulated reward
            adjs = [adj_choose]
            weights = [w_choose]
            adjusts = [adjust_choose]
            auxs = [aux_choose]

        # Choose action
        if model_initialized:
            epsilon = exploration.value(t)
            q_values=session.run(q_func_net, feed_dict={obs_t_ph: observations[-1][None],
                                                        adj_ph: adj_choose[None],
                                                        graph_weights_ph: w_choose[None],adjust_matrix_ph:adjust_choose[None],multiply_matrix_ph:multiply_choose[None],
                                                        aux_ph:aux_choose[None]
                                                        })
            #此处observation[-1][None]为在observation[-1]的外面加上一层括号，例如a=[1,2,3]np array，a[None]为[[1,2,3]]
            #这样产生的q_value为[[....]]，中间为当前状态所对应的q value

            action = np.argmax((q_values[0] * (1 - observations[-1]) - 1e5 * observations[-1]))
            r = random.random()
            if r <= epsilon:
                all_possible_action = list(range(real_nodes))
                # other_actions = [x for x in all_possible_action if x != action]
                if env.env_name == 'MVC':
                    other_actions = [x for x in all_possible_action if env.state[x] != 1]
                else:
                    other_actions = [x for x in all_possible_action if observations[-1][x] != 1]
                action = np.array(random.choice(other_actions))
        else:
            action = np.array(random.choice(list(range(real_nodes))))

        next_obs, step_reward, done, adj_choose, w_choose, adjust_choose, aux_choose = env.step(action)
        observations.append(next_obs)
        adjs.append(adj_choose)
        weights.append(w_choose)
        adjusts.append(adjust_choose)
        auxs.append(aux_choose)
        actions_list.append(action)
        rewards_list.append(step_reward)
        eval_n_q_values=session.run(q_func_net, feed_dict={obs_t_ph: next_obs[None],
                                                        adj_ph: adj_choose[None],
                                                        graph_weights_ph: w_choose[None],adjust_matrix_ph:adjust_choose[None],multiply_matrix_ph:multiply_choose[None],
                                                        aux_ph:aux_choose[None]})
        n_action = np.argmax((eval_n_q_values[0] * (1 - next_obs) - 1e5 * next_obs))
        target_n_q_value = session.run(target_q_func_net,feed_dict={obs_tp1_ph:next_obs[None],
                                                                adj_ph: adj_choose[None],
                                                                graph_weights_ph: w_choose[None],adjust_matrix_ph:adjust_choose[None],multiply_matrix_ph:multiply_choose[None],
                                                                aux_ph:aux_choose[None]})[0][n_action]
        if len(observations) > n_steps_ahead:
            reward = 0
            for temp_count in range(n_steps_ahead):
                reward += rewards_list[-temp_count-1]*np.power(gamma,(n_steps_ahead-temp_count-1))
            replay_buffer.store_transition(observations[-(n_steps_ahead + 1)], adjs[-(n_steps_ahead + 1)],
                                           weights[-(n_steps_ahead + 1)], actions_list[-n_steps_ahead], target_n_q_value, n_action,
                                             reward, observations[-1], done, n_steps_ahead,real_nodes,
                                             adjusts[-(n_steps_ahead + 1)], auxs[-(n_steps_ahead + 1)])
            
        if done and 1 < len(observations) <= n_steps_ahead:
            reward = 0
            for temp_count in range(len(observations)-1):
                reward += rewards_list[temp_count]*np.power(gamma,temp_count-1)
            replay_buffer.store_transition(observations[0], adjs[0], weights[0],
                                           actions_list[0], 0, n_action ,reward, observations[-1], done, len(observations) - 1,real_nodes,
                                           adjusts[0], auxs[0])


        # at this point, the environment should have been advanced one step (and
        # reset if done was true), and last_obs should point to the new latest
        # observation

        ### 3. Perform experience replay and train the network.
        # note that this is only done if the replay buffer contains enough samples
        # for us to learn something useful -- until then, the model will not be
        # initialized and random actions should be taken

        if (t > learning_starts and
                t % learning_freq == 0 and
                replay_buffer.can_sample(batch_size)):

            obs_t_batch, adj_batch, graph_weights_batch, act_batch, n_q_values_batch,n_actions_batch,\
            rew_batch, obs_tp1_batch, done_mask_batch, transition_length_batch, true_nodes_batch, adjust_batch, aux_batch= replay_buffer.sample(batch_size)
            
            if not(model_initialized):
                initialize_interdependent_variables(session, tf.global_variables(), {
                            obs_t_ph: obs_t_batch,
                            obs_tp1_ph: obs_tp1_batch,
                        })
                model_initialized=True

            training_error_summ, _ = session.run([training_error_summ_sy, train_fn],
                                                 feed_dict={obs_t_ph: obs_t_batch,
                                                            adj_ph: adj_batch,
                                                            graph_weights_ph: graph_weights_batch,
                                                            act_t_ph: act_batch,q_target_ph:n_q_values_batch,n_act_ph:n_actions_batch,
                                                            rew_t_ph:rew_batch,
                                                            obs_tp1_ph: obs_tp1_batch,
                                                            done_mask_ph: done_mask_batch,
                                                            transition_length_ph: transition_length_batch, global_step:t,
                                                            adjust_matrix_ph:adjust_batch, multiply_matrix_ph:multiply_matrix,
                                                            aux_ph: aux_batch})


            if t % 100:
                writer.add_summary(training_error_summ, t)
                writer.flush()

            if num_param_updates%target_update_freq == 0:
                session.run(update_target_fn)
            num_param_updates += 1

            #####

            if t % 1000 == 0 :
                true_num = 0
                total_num = 0
                total_opt_rew = 0
                total_model_rew = 0 

                if one_graph_test != True:
                    for test_ep in range(100):
                        test_done = False
                        test_real_nodes, test_x, test_adj, test_w, test_adjust, test_aux = test_env.reset()
                        for test_step in range(num_actions):
                            if test_done:
                                # print('t= ' + str(t) + '\n')
                                # print(test_env.adjacency_matrix)
                                # print('\n')
                                # print("solution:  " +str(test_x) + '\n')
                                model_rew = -np.sum(test_env.real_state)
                                opt_rew, opt_solution= test_env.optimal_solution()
                                total_opt_rew += opt_rew
                                total_model_rew += model_rew
                                if model_rew == opt_rew:
                                    true_num += 1
                                total_num += 1
                                break
                            test_q_values=session.run(q_func_net, feed_dict={obs_t_ph: test_x[None],
                                                                    adj_ph: test_adj[None],
                                                                    graph_weights_ph: test_w[None],adjust_matrix_ph:test_adjust[None],multiply_matrix_ph:multiply_choose[None],
                                                                    aux_ph: test_aux[None]})
                            test_action = np.argmax((test_q_values[0] * (1 - test_x) - 1e5 * test_x))
                            test_x, test_reward, test_done, test_adj, test_w, test_adjust, test_aux= test_env.step(test_action)
                    appro_ratio = total_model_rew/total_opt_rew
                    file = open(filename, 'a')
                    file.write("t=" +str(t)+'\n')
                    file.write('true test=' + str(true_num)+'\n')
                    file.write('total test=' + str(total_num)+'\n')
                    file.write("total opt rew=" + str(total_opt_rew) + '\n')
                    file.write("total model rew=" + str(total_model_rew) + '\n')
                    file.write(time.strftime('%m-%d-%Y-%H-%M-%S') + '\n')
                    file.write('appro_ratio = '+ str(appro_ratio)+'\n')
                    file.write('epsilon = '+str(epsilon)+'\n')
                    file.write('\n')

                
                sys.stdout.flush()