import numpy as np
import pandas as pd
import random
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from collections import deque

print("="*50)
print("MEMULAI EVALUASI KOMPARATIF (BASELINE vs Q-LEARNING vs DQN)")
print("="*50)

# ==========================================
# 1. LOAD DATA & ENVIRONMENT UTAMA
# ==========================================
def load_real_data(file_path):
    df = pd.read_csv(file_path)
    df_product = df[df['product_id'] == "aca2eb7d00ea1a7b8ebd4e68314663af"].copy() 
    df_product = df_product.sort_values('Date')
    df_product['pct_change_qty'] = df_product['Quantity_Sold'].pct_change()
    df_product['pct_change_price'] = df_product['Price'].pct_change()
    df_product['elasticity'] = abs(df_product['pct_change_qty'] / df_product['pct_change_price'])
    df_product.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_product['elasticity'] = df_product['elasticity'].fillna(1.5) 
    df_product.rename(columns={'Competitor_Price': 'competitor_price'}, inplace=True)
    return df_product.reset_index(drop=True)

class DynamicPricingEnv:
    def __init__(self, data):
        self.data = data
        self.max_days = len(data) - 1
        
    def reset(self):
        self.current_day = 0
        self.current_price = self.data.iloc[0]['Price']
        self.base_cost = self.current_price * 0.7
        self.current_demand = self.data.iloc[0]['Quantity_Sold']
        return self._get_state()
        
    def _get_state(self):
        today = self.data.iloc[self.current_day]
        return np.array([self.current_price/100000.0, self.current_demand/10.0, today['elasticity'], today['competitor_price']/100000.0], dtype=np.float32)
        
    def step(self, action):
        today = self.data.iloc[self.current_day]
        base_price = today['Price']
        
        if action == 0: self.current_price *= 0.95
        elif action == 2: self.current_price *= 1.05
            
        # Guardrails (Batas Bawah dan Atas)
        self.current_price = max(self.base_cost * 1.05, min(self.current_price, base_price * 1.50))
            
        demand_change = - (((self.current_price - base_price) / base_price) * today['elasticity'])
        self.current_demand = max(1, int(today['Quantity_Sold'] * (1 + demand_change)))
        
        profit = (self.current_price - self.base_cost) * self.current_demand
        reward = profit / 100000.0
        if self.current_price > today['competitor_price'] * 1.1: reward -= 1.0 
            
        self.current_day += 1
        return self._get_state(), float(reward), self.current_day >= self.max_days, float(profit)

# ==========================================
# 2. AGEN Q-LEARNING TRADISIONAL (TABULAR)
# ==========================================
class QLearningAgent:
    def __init__(self):
        self.q_table = {}
        self.alpha = 0.1
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.96
        self.epsilon_min = 0.05

    def discretize(self, state):
        return (round(state[0], 1), round(state[1], 0), round(state[2], 1), round(state[3], 1))

    def act(self, state):
        state_d = self.discretize(state)
        if np.random.rand() <= self.epsilon: return random.choice([0, 1, 2])
        if state_d not in self.q_table: self.q_table[state_d] = np.zeros(3)
        return np.argmax(self.q_table[state_d])

    def learn(self, state, action, reward, next_state):
        s_d, ns_d = self.discretize(state), self.discretize(next_state)
        if s_d not in self.q_table: self.q_table[s_d] = np.zeros(3)
        if ns_d not in self.q_table: self.q_table[ns_d] = np.zeros(3)
        
        td_target = reward + self.gamma * np.max(self.q_table[ns_d])
        self.q_table[s_d][action] += self.alpha * (td_target - self.q_table[s_d][action])

# ==========================================
# 3. AGEN DEEP Q-NETWORK (DQN)
# ==========================================
class DQNAgent:
    def __init__(self):
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    
        self.epsilon = 1.0  
        self.epsilon_decay = 0.96 
        self.epsilon_min = 0.05
        self.model = self._build()
        self.target_model = self._build()
        self.target_model.set_weights(self.model.get_weights())

    def _build(self):
        m = Sequential([Input(shape=(4,)), Dense(32, activation='relu'), Dense(16, activation='relu'), Dense(3, activation='linear')])
        m.compile(loss='mse', optimizer=Adam(learning_rate=0.0005))
        return m

    def act(self, state):
        if np.random.rand() <= self.epsilon: return random.choice([0, 1, 2])
        return np.argmax(self.model.predict(np.reshape(state, [1, 4]), verbose=0)[0])

    def replay(self):
        if len(self.memory) < 32: return
        batch = random.sample(self.memory, 32)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        
        targets = rewards + self.gamma * np.amax(self.target_model.predict(next_states, verbose=0), axis=1) * (1 - dones)
        target_f = self.model.predict(states, verbose=0)
        for i, a in enumerate(actions): target_f[i][a] = targets[i]
        self.model.fit(states, target_f, epochs=1, verbose=0)

# ==========================================
# 4. EKSEKUSI TRAINING & PENGUMPULAN DATA
# ==========================================
df = load_real_data("dataset_olist_siap_train.csv")
env = DynamicPricingEnv(df)
episodes = 50 # Set ke 50 agar cepat. Ubah ke 200/300 untuk hasil jurnal final.

# A. Uji Baseline (Harga Statis - Action 1 Terus)
print("\n[1/3] Menghitung Profit Baseline (Harga Statis)...")
env.reset()
profit_baseline = sum(env.step(1)[3] for _ in range(env.max_days))
baseline_scores = [profit_baseline] * episodes

# B. Uji Q-Learning
print(f"\n[2/3] Melatih Q-Learning ({episodes} Episode)...")
agent_q = QLearningAgent()
scores_q = []
for e in range(episodes):
    s, done, total = env.reset(), False, 0
    while not done:
        a = agent_q.act(s)
        ns, r, done, p = env.step(a)
        agent_q.learn(s, a, r, ns)
        s, total = ns, total + p
    if agent_q.epsilon > agent_q.epsilon_min: agent_q.epsilon *= agent_q.epsilon_decay
    scores_q.append(total)

# C. Uji DQN
print(f"\n[3/3] Melatih Deep Q-Network ({episodes} Episode)...")
agent_dqn = DQNAgent()
scores_dqn = []
for e in range(episodes):
    s, done, total = env.reset(), False, 0
    while not done:
        a = agent_dqn.act(s)
        ns, r, done, p = env.step(a)
        agent_dqn.memory.append((s, a, r, ns, done))
        s, total = ns, total + p
        agent_dqn.replay()
    if agent_dqn.epsilon > agent_dqn.epsilon_min: agent_dqn.epsilon *= agent_dqn.epsilon_decay
    agent_dqn.target_model.set_weights(agent_dqn.model.get_weights())
    scores_dqn.append(total)

# ==========================================
# 5. GENERATE ULTIMATE GRAPH
# ==========================================
print("\nMenyiapkan Grafik Komparasi...")
plt.figure(figsize=(12, 6))

window = max(1, episodes // 10)
ma_q = pd.Series(scores_q).rolling(window=window, min_periods=1).mean()
ma_dqn = pd.Series(scores_dqn).rolling(window=window, min_periods=1).mean()

plt.plot(range(1, episodes+1), baseline_scores, color='gray', linestyle='--', linewidth=2, label='Baseline (Harga Statis)')
plt.plot(range(1, episodes+1), ma_q, color='orange', linewidth=2.5, label='Q-Learning (Tradisional)')
plt.plot(range(1, episodes+1), ma_dqn, color='green', linewidth=3, label='Deep Q-Network (DQN - Proposed)')

plt.title('Perbandingan Performa Model Dynamic Pricing (DQN vs Q-Learning vs Baseline)', fontsize=14, fontweight='bold')
plt.xlabel('Episode Training', fontsize=12)
plt.ylabel('Total Profit per Episode (Rp)', fontsize=12)
plt.legend(loc='lower right', fontsize=11, shadow=True)
plt.grid(True, linestyle=':', alpha=0.7)
plt.tight_layout()

nama_gambar = 'grafik_komparasi_final.png'
plt.savefig(nama_gambar, dpi=300) # Resolusi tinggi untuk Jurnal Sinta
print(f"Grafik pamungkas berhasil disimpan sebagai: '{nama_gambar}'")
plt.show()