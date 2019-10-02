import tensorflow as tf
import numpy as np

from DeepQNetwork import DeepQNetwork


class Agent(object):
    def __init__(self, alpha, gamma, mem_size, n_actions, epsilon, batch_size,
                 replace_target=10000, input_dims=(210, 160, 4),
                 q_next_dir='tmp/q_next', q_eval_dir='tmp/q_eval'):
        self.action_space = [i for i in range(n_actions)]
        self.n_actions = n_actions
        self.gamma = gamma
        self.mem_size = mem_size
        self.mem_cntr = 0
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.replace_target = replace_target
        self.q_next = DeepQNetwork(alpha, n_actions, input_dims=input_dims,
                                   name='q_next', chkpt_dir=q_next_dir)
        self.q_eval = DeepQNetwork(alpha, n_actions, input_dims=input_dims,
                                   name='q_eval', chkpt_dir=q_eval_dir)
        self.state_memory = np.zeros((self.mem_size, *input_dims))
        self.new_state_memory = np.zeros((self.mem_size, *input_dims))
        self.action_memory = np.zeros((self.mem_size, self.n_actions),
                                      dtype=np.int8)
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.int8)

    def store_transition(self, state, action, reward, state_, terminal):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        actions = np.zeros(self.n_actions)
        actions[action] = 1.0
        self.action_memory[index] = actions
        self.reward_memory[index] = reward
        self.new_state_memory[index] = state_
        self.terminal_memory[index] = 1 - terminal
        self.mem_cntr += 1

    def choose_action(self, state):
        rand = np.random.random()
        if rand < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            actions = self.q_eval.sess.run(self.q_eval.Q_values,
                                           feed_dict={self.q_eval.input: state})
            action = np.argmax(actions)
        return action

    def learn(self):
        if self.mem_cntr % self.replace_target == 0:
            self.update_graph()

        max_mem = self.mem_cntr if self.mem_cntr < self.mem_size else self.mem_size

        batch = np.random.choice(max_mem, self.batch_size)

        state_batch = self.state_memory[batch]
        action_batch = self.action_memory[batch]
        action_values = np.array([0, 1, 2], dtype=np.int8)
        action_indices = np.dot(action_batch, action_values)
        reward_batch = self.reward_memory[batch]
        new_state_batch = self.new_state_memory[batch]
        terminal_batch = self.terminal_memory[batch]

        q_eval = self.q_eval.sess.run(self.q_eval.Q_values,
                                      feed_dict={self.q_eval.input: state_batch})
        q_next = self.q_next.sess.run(self.q_next.Q_values,
                                      feed_dict={self.q_next.input: new_state_batch})

        q_target = q_eval.copy()
        idx = np.arange(self.batch_size)
        q_target[idx, action_indices] = reward_batch + \
                                        self.gamma * np.max(q_next, axis=1) * terminal_batch

        # q_target = np.zeros(self.batch_size)
        # q_target = reward_batch + self.gamma*np.max(q_next, axis=1)*terminal_batch

        _ = self.q_eval.sess.run(self.q_eval.train_op,
                                 feed_dict={self.q_eval.input: state_batch,
                                            self.q_eval.actions: action_batch,
                                            self.q_eval.q_target: q_target})

        if self.mem_cntr > 25000:  # 200000:
            if self.epsilon > 0.05:
                self.epsilon -= 4e-7
            elif self.epsilon <= 0.05:
                self.epsilon = 0.05

    def save_models(self):
        self.q_eval.save_checkpoint()
        self.q_next.save_checkpoint()

    def load_models(self):
        self.q_eval.load_checkpoint()
        self.q_next.load_checkpoint()

    def update_graph(self):
        t_params = self.q_next.params
        e_params = self.q_eval.params

        for t, e in zip(t_params, e_params):
            self.q_eval.sess.run(tf.assign(t, e))