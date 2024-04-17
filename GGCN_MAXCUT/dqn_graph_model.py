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
import maxcut_env

OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs", "lr_schedule"])

"""
learn slightly modified to pass the task name as an argument
so that it is easier to record data.
"""

def learn(env,
          q_func,
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
          test_mode = False,
          one_graph_test=True
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
    test_env = maxcut_env.maxcut_env(num_min,num_actions)
    # filename = "test2 10-15.txt"
    filename = env.env_name + str(env.number_nodes_min) + '-' + str(env.number_nodes) + \
                    'buffer' + str(replay_buffer_size) + '-' +\
                        "target_update_freq"+str(target_update_freq)+\
                            '-'+ time.strftime('%m-%d-%Y-%H-%M-%S')+"-"+"doubleDQN="+ str(double_DQN) +".txt"

    # set up placeholders
    # placeholder for current observation (or state)
    obs_t_ph              = tf.placeholder(tf.float32, [None] + list(input_shape)+[2])#占位符，tf.placeholer(dtype, shape=None, name=None),shape为[None]表示形状不指定
    # placeholder for current action
    act_t_ph              = tf.placeholder(tf.int32, [None], name='act_t_ph')
    n_act_ph              = tf.placeholder(tf.int32, [None])
    # placeholder for current reward
    rew_t_ph              = tf.placeholder(tf.float32, [None])
    # placeholder for next observation (or state)
    obs_tp1_ph            = tf.placeholder(tf.float32, [None] + list(input_shape)+[2])
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
    edge_feat_ph = tf.placeholder(tf.float32, [None, num_actions, num_actions, 4])
    aux_ph = tf.placeholder(tf.float32, [None, num_actions, 4])
    e2n_param_ph = tf.placeholder(tf.float32, [None, num_actions])

    # Graphs specific placeholder
    adj_ph = tf.placeholder(tf.float32, [None, env.number_nodes, env.number_nodes],
                            name='adj_ph')
    graph_weights_ph = tf.placeholder(tf.float32,
                                      [None, env.number_nodes, env.number_nodes],
                                      name='graph_weights_ph')

    
    q_func_net = q_func(node_feat=obs_t_ph,
                        adj=adj_ph,
                        w=graph_weights_ph, p=n_hidden_units, initialization_stddev=initialization_stddev,
                        adjust_matrix = adjust_matrix_ph, multiply_matrix=multiply_matrix_ph,
                        edge_feat=edge_feat_ph,layer_num=T,scope='q_func'
                        )

    target_q_func_net = q_func(node_feat=obs_tp1_ph,
                            adj=adj_ph,
                            w=graph_weights_ph, p=n_hidden_units, initialization_stddev=initialization_stddev,
                            adjust_matrix = adjust_matrix_ph, multiply_matrix=multiply_matrix_ph,
                            edge_feat=edge_feat_ph,layer_num=T,scope='target_q_func'
                            )

    q_target_ph = tf.placeholder(tf.float32,[None])
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
    # learning_rate = tf.train.exponential_decay(learning_rate=learning_rate_start, global_step=global_step, decay_steps=2000, decay_rate=0.95, staircase=True)
    optimizer = tf.train.AdamOptimizer(learning_rate_start)
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
    log_files_name = 'DQN_' + str(env.env_name) + \
                     '-lf=' + str(learning_freq) + \
                     '-b=' + str(batch_size) + '-' + \
                     time.strftime('%m-%d-%Y-%H-%M-%S')
    log_files_name = env.env_name + str(env.number_nodes_min) + '-' + str(env.number_nodes) + \
                    'buffer' + str(replay_buffer_size) + '-'+ time.strftime('%m-%d-%Y-%H-%M-%S')+"-"+"doubleDQN="+ str(double_DQN) 

    writer = tf.summary.FileWriter('/tmp/' + log_files_name,
                                   session.graph)
    if pre_training ==None:
        tf.global_variables_initializer().run()
    else:
        saver.restore(session, '/tmp/saved_models/'+pre_training )
    # saver.save(session, '/tmp/saved_models/' + log_files_name +'/')


    if test_mode == True:
        file = open(filename, 'a')
        test_real_nodes, test_x = test_env.reset()
        file.write(str(test_env.edges))
        file.write('\n')
        adjust_test = np.pad(np.eye(test_real_nodes,dtype=np.float32),((0,num_actions-test_real_nodes),(0,num_actions-test_real_nodes)),'constant',constant_values=(0,0))
        for test_step in range(num_actions):
            test_q_values=session.run(q_func_net, feed_dict={obs_t_ph: test_x[None],
                                                                adj_ph: test_env.adjacency_matrix[None],
                                                                graph_weights_ph: test_env.weight_matrix[None],adjust_matrix_ph:adjust_test[None],multiply_matrix_ph:multiply_choose[None]})
            test_action = np.argmax((test_q_values[0] * (1 - test_x) - 1e5 * test_x))
            test_x, test_reward, test_done = test_env.step(test_action)
            file.write(str(test_q_values)+'\n')
            file.write(str(test_x)+'\n')
            file.write('\n')
        opt_rew, opt_solution= test_env.optimal_solution()
        file.write(str(opt_solution))

        file.close()
    ###############
    # RUN ENV     #
    ###############
    model_initialized = True
    if pre_training != None:
        model_initialized = True
    num_param_updates = 0
    mean_episode_reward      = -float('nan')
    best_mean_episode_reward = -float('inf')
    real_nodes, now_state, node_feat_choose, adj_choose, w_choose, adjust_choose, edge_feat_choose, aux_choose= env.reset()
    e2n_param_choose = env.get_e2n_param()

    LOG_EVERY_N_STEPS = 10000

    episode_total_rewards = []
    episode_total_optimal_rewards = []
    episode_total_at_random_rewards = []
    done = False


    for t in itertools.count():#从0开始步长为1的无限序列
        ### 1. Check stopping criterion
        if stopping_criterion is not None and stopping_criterion(env, t):
            break

        ### 2. Step the env and store the transition
        import random
        from numpy import array

        if done:
            if one_graph_test != True:
                real_nodes, now_state, node_feat_choose, adj_choose, w_choose, adjust_choose, edge_feat_choose, aux_choose = env.reset()
                e2n_param_choose = env.get_e2n_param()
            else:
                real_nodes, now_state, node_feat_choose, adj_choose, w_choose, adjust_choose, edge_feat_choose, aux_choose = env.one_graph_reset()
                e2n_param_choose = env.get_e2n_param()

        # Choose action
        epsilon = exploration.value(t)

        q_values=session.run(q_func_net, feed_dict={obs_t_ph: node_feat_choose[None],
                                                    adj_ph: adj_choose[None],
                                                    graph_weights_ph: w_choose[None],adjust_matrix_ph:adjust_choose[None],multiply_matrix_ph:multiply_choose[None],
                                                    edge_feat_ph:edge_feat_choose[None],aux_ph:aux_choose[None],e2n_param_ph:e2n_param_choose[None]
                                                    })
        action_num = env.action_count
        #此处observation[-1][None]为在observation[-1]的外面加上一层括号，例如a=[1,2,3]np array，a[None]为[[1,2,3]]
        #这样产生的q_value为[[....]]，中间为当前状态所对应的q value

        action = np.argmax((q_values[0] * (1 - now_state) - 1e5 * now_state))
        r = random.random()
        if r <= epsilon:
            all_possible_action = list(range(real_nodes))
            # other_actions = [x for x in all_possible_action if x != action]
            if env.env_name == 'MVC':
                other_actions = [x for x in all_possible_action if env.state[x] != 1]
            else:
                other_actions = [x for x in all_possible_action if now_state[x] != 1]
            action = np.array(random.choice(other_actions))

        now_state, next_node_feat_choose, step_reward, done, adj_choose, w_choose, adjust_choose, next_edge_feat_choose, next_aux_choose = env.step(action)
        
        eval_n_q_values=session.run(target_q_func_net, feed_dict={obs_tp1_ph: next_node_feat_choose[None],
                                                        adj_ph: adj_choose[None],
                                                        graph_weights_ph: w_choose[None],adjust_matrix_ph:adjust_choose[None],multiply_matrix_ph:multiply_choose[None],
                                                        edge_feat_ph:next_edge_feat_choose[None],aux_ph:next_aux_choose[None],e2n_param_ph:e2n_param_choose[None]})
        n_action = np.argmax((eval_n_q_values[0] * (1 - now_state) - 1e5 * now_state))
        target_n_q_value = eval_n_q_values[0][n_action]
        
        # if target_n_q_value < 0:
        #     target_n_q_value = 0

        if num_param_updates ==20000:
            abc=1
        
        replay_buffer.store_transition(node_feat_choose, adj_choose,
                                        w_choose, action, target_n_q_value, n_action,
                                            step_reward, next_node_feat_choose, done, n_steps_ahead,real_nodes,
                                            adjust_choose, edge_feat_choose, aux_choose, e2n_param_choose)

        node_feat_choose = next_node_feat_choose
        edge_feat_choose = next_edge_feat_choose
        aux_choose = next_aux_choose

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
            rew_batch, obs_tp1_batch, done_mask_batch, transition_length_batch, true_nodes_batch, adjust_batch, edge_feat_batch,\
                aux_batch, e2n_param_batch= replay_buffer.sample(batch_size)
            
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
                                                            edge_feat_ph: edge_feat_batch, aux_ph:aux_batch, e2n_param_ph:e2n_param_batch})

            # if t>10000:
            #     exploration = ConstantSchedule(0.05)
            # import pdb; pdb.set_trace()
            if t % 100:
                writer.add_summary(training_error_summ, t)
                writer.flush()

            if num_param_updates%target_update_freq == 0:
                session.run(update_target_fn)
            num_param_updates += 1

            #####

            ### 4. Log progress
            # episode_rewards = get_wrapper_by_name(env, "Monitor").get_episode_rewards()
            if done:
                episode_total_rewards.append(env.accumulated_reward())
                episode_total_optimal_rewards.append(0)#env.optimal_solution()[0]
                episode_total_at_random_rewards.append(0)


            if len(episode_total_rewards) > 0:
                mean_episode_reward = np.mean(np.array(episode_total_rewards)[-1000:])
                mean_optimal_episode_reward = np.mean(np.array(episode_total_optimal_rewards)[-1000:])
                mean_at_random_episode_reward = np.mean(np.array(episode_total_at_random_rewards)[-1000:])
                if env.env_name == 'TSP':
                    mean_approx_ratio = np.mean(np.array(episode_total_rewards)[-1000:] /
                                                np.mean(np.array(episode_total_optimal_rewards)[-1000:]))

            if len(episode_total_rewards) > 1000:
                best_mean_episode_reward = max(best_mean_episode_reward, mean_episode_reward)

            if t % LOG_EVERY_N_STEPS == 0 and model_initialized:
                # Save the model
                saver.save(session, '/tmp/saved_models/test/' + log_files_name + str(t))
                # Display and log episode stats
                # logz.log_tabular("Timestep", t)
                # logz.log_tabular("AtRandomAverageReturn", mean_at_random_episode_reward)
                # logz.log_tabular("AverageReturn", mean_episode_reward)
                # logz.log_tabular("OptimalAverageReturn", mean_optimal_episode_reward)
                # if env.env_name == 'TSP':
                #     logz.log_tabular("ApproxRatio", mean_approx_ratio)
                # logz.log_tabular("MaxReturn", best_mean_episode_reward)
                # logz.log_tabular("Episodes", len(episode_total_rewards))
                # logz.log_tabular("Exploration", exploration.value(t))
                # logz.dump_tabular()

            if t % 1000 == 0 :
                test = True
                true_num = 0
                total_num = 0
                total_opt_rew = 0
                total_model_rew = 0 

                if one_graph_test != True:
                    for test_ep in range(100):
                        test_done=False
                        test_real_nodes, test_state, test_x, test_adj, test_w, test_adjust, test_edge_feat, test_aux = test_env.reset()
                        e2n_param_test = test_env.get_e2n_param()
                        for test_step in range(num_actions + 10):
                            if test_done:
                                # print('t= ' + str(t) + '\n')
                                # print(test_env.adjacency_matrix)
                                # print('\n')
                                # print("solution:  " +str(test_x) + '\n')
                                model_rew = test_env.max_accumulated_reward()
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
                                                                    edge_feat_ph: test_edge_feat[None], aux_ph:test_aux[None], e2n_param_ph:e2n_param_test[None]})
                            test_action = np.argmax((test_q_values[0] * (1 - test_state) - 1e5 * test_state))
                            test_state, test_x, test_reward, test_done, test_adj, test_w, test_adjust, test_edge_feat, test_aux= test_env.step(test_action)
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
                else:
                    appro_ratio = 2
                    real_nodes, x = env.one_graph_reset()
                    adjust_test = np.pad(np.eye(real_nodes,dtype=np.float32),((0,num_actions-real_nodes),(0,num_actions-real_nodes)),'constant',constant_values=(0,0))
                    file = open(filename, 'a')
                    opt_rew, opt_solution= env.optimal_solution()
                    for test_step in range(num_actions):
                        test_q_values=session.run(q_func_net, feed_dict={obs_t_ph: x[None],
                                                                adj_ph: env.adjacency_matrix[None],
                                                                graph_weights_ph: env.weight_matrix[None],adjust_matrix_ph:adjust_test[None],multiply_matrix_ph:multiply_choose[None]})
                        test_action = np.argmax((test_q_values[0] * (1 - x) - 1e5 * x))
                        x, test_reward, test_done = env.step(test_action)
                        file.write(str(test_q_values)+'\n')
                        file.write(str(x)+'\n')
                        if test_done:
                            file.write('-------------------------------------------------\n')
                            file.write('t='+str(t)+'\n')
                            file.write(str(list(x))+'\n')
                            file.write(str(opt_solution+[1]*(num_actions-real_nodes))+'\n')
                            file.write('--------------------------------------------------------\n')
                            file.write('\n')
                            file.write('\n')
                            file.write('\n')
                            env.one_graph_reset()
                            break
                    file.close()
                if appro_ratio <= 1.02:
                    saver.save(session, '/tmp/saved_models/trained/' + log_files_name + str(t))
                
                # sys.stdout.flush()