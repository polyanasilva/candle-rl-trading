import numpy as np
from collections import defaultdict

class ReinforcementAgent:
    def __init__(self, states, actions, alpha=0.1, gamma=0.95, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.states = states
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_table = defaultdict(lambda: np.zeros(len(actions)))

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.actions)
        else:
            return self.actions[np.argmax(self.q_table[state])]

    def learn(self, state, action, reward, next_state):
        action_idx = self.actions.index(action)
        best_next_action_idx = np.argmax(self.q_table[next_state])
        td_target = reward + self.gamma * self.q_table[next_state][best_next_action_idx]
        td_error = td_target - self.q_table[state][action_idx]
        self.q_table[state][action_idx] += self.alpha * td_error

    def train(self, episodes, get_state_reward_next):
        rewards = []
        for _ in range(episodes):
            total_reward = 0
            for state, action, reward, next_state in get_state_reward_next():
                self.learn(state, action, reward, next_state)
                total_reward += reward
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            rewards.append(total_reward)
        return rewards

    def get_optimal_policy(self):
        return {state: self.actions[np.argmax(q_values)] for state, q_values in self.q_table.items()}

    def get_q_table(self):
        return dict(self.q_table)

    def action_distribution(self):
        dist = defaultdict(int)
        for state in self.q_table:
            best_action = self.actions[np.argmax(self.q_table[state])]
            dist[best_action] += 1
        return dict(dist)
