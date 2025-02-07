# File: src/models/reinforcement_learning.py
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.ml_model import PricePredictor

class QLearningAgent:
    def __init__(self, state_size, action_size, alpha=0.1, gamma=0.95, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.alpha = alpha        # Learning rate
        self.gamma = gamma        # Discount factor
        self.epsilon = epsilon    # Exploration rate
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        # Q-table initialized as a dictionary for simplicity
        self.q_table = {}
    
    def get_q(self, state, action):
        return self.q_table.get((state, action), 0.0)
    
    def choose_action(self, state):
        """
        Choose action using epsilon-greedy strategy.
        For simplicity, encode state as a hashable tuple.
        """
        state_key = tuple(state)
        if np.random.rand() < self.epsilon:
            action = np.random.choice(self.action_size)
        else:
            # Choose action with highest Q-value
            q_vals = [self.get_q(state_key, a) for a in range(self.action_size)]
            action = int(np.argmax(q_vals))
        return action
    
    def learn(self, state, action, reward, next_state, done):
        state_key = tuple(state)
        next_state_key = tuple(next_state)
        current_q = self.get_q(state_key, action)
        # Next max Q-value:
        next_qs = [self.get_q(next_state_key, a) for a in range(self.action_size)]
        max_next_q = max(next_qs) if next_qs else 0.0
        # Q-learning update
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q * (1 - int(done)) - current_q)
        self.q_table[(state_key, action)] = new_q
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Example usage for ML Model:
if __name__ == "__main__":
    X = np.array([[100], [101], [102], [103], [104]])
    y = np.array([101, 102, 103, 104, 105])
    predictor = PricePredictor()
    predictor.train(X, y)
    pred = predictor.predict(np.array([[105]]))
    print(f"Prediction for input 105: {pred[0]:.2f}")

# Example usage for Q-Learning Agent:
if __name__ == "__main__":
    agent = QLearningAgent(state_size=1, action_size=3)
    state = [100]
    action = agent.choose_action(state)
    print("Chosen action:", action)
    agent.learn(state, action, reward=1, next_state=[101], done=False)
