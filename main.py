import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt
import os


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Menggunakan device komputasi: {device}\n")


def load_real_data(file_path, target_product="P001"):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} tidak ditemukan! Pastikan nama file CSV sudah benar.")
        
    df = pd.read_csv(file_path)
    
    df_product = df[df['product_id'] == target_product].copy() 
    df_product = df_product.sort_values('Date')
    
    
    df_product['pct_change_qty'] = df_product['Quantity_Sold'].pct_change()
    df_product['pct_change_price'] = df_product['Price'].pct_change()
    df_product['elasticity'] = abs(df_product['pct_change_qty'] / df_product['pct_change_price'])
    
    
    df_product.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_product['elasticity'] = df_product['elasticity'].fillna(1.5) 
    
    df_product.rename(columns={'Competitor_Price': 'competitor_price'}, inplace=True)
    
    df_product = df_product.reset_index(drop=True)
    df_product['day'] = df_product.index
    
    print(f"--- Dataset Siap (Produk {target_product}, Total Hari: {len(df_product)}) ---")
    return df_product


class DynamicPricingEnv:
    def __init__(self, data):
        self.data = data
        self.max_days = len(data) - 1
        self.current_day = 0
        
        initial_price = self.data.iloc[0]['Price']
        self.base_cost = initial_price * 0.7 
        self.current_price = initial_price
        self.current_demand = self.data.iloc[0]['Quantity_Sold']
        
        self.action_space = [0, 1, 2] 
        
    def reset(self):
        self.current_day = 0
        self.current_price = self.data.iloc[0]['Price']
        self.current_demand = self.data.iloc[0]['Quantity_Sold']
        return self._get_state()
        
    def _get_state(self):
        today_data = self.data.iloc[self.current_day]
        
        state = np.array([
            self.current_price / 100000.0,  
            self.current_demand / 10.0,     
            today_data['elasticity'], 
            today_data['competitor_price'] / 100000.0
        ], dtype=np.float32)
        return state
        
    def step(self, action):
        today_data = self.data.iloc[self.current_day]
        elasticity = today_data['elasticity']
        base_price_today = today_data['Price'] 
        base_demand_today = today_data['Quantity_Sold']
        
        if action == 0:
            self.current_price *= 0.95
        elif action == 2:
            self.current_price *= 1.05
            
        price_diff_ratio = (self.current_price - base_price_today) / base_price_today
        demand_change = - (price_diff_ratio * elasticity)
        
        self.current_demand = max(1, int(base_demand_today * (1 + demand_change)))
        
       
        profit = (self.current_price - self.base_cost) * self.current_demand
        
        
        reward = profit / 100000.0
        
        # Reward Shaping: Penalti
        if self.current_price > today_data['competitor_price'] * 1.1:
            reward -= 5.0 # Setara penalti 500rb
            
        self.current_day += 1
        done = self.current_day >= self.max_days
        
        return self._get_state(), float(reward), done, float(profit)

# 4. Arsitektur Deep Q-Network
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# 5. Agen Reinforcement Learning (Tuning Hyperparameter)
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=5000) # Memori diperbesar
        
        self.gamma = 0.95    
        self.epsilon = 1.0   
        self.epsilon_min = 0.05 # Jangan sampai terlalu kecil
        self.epsilon_decay = 0.985 # Eksplorasi lebih lambat
        self.learning_rate = 0.0005 # Learning rate diperhalus
        self.batch_size = 64 # Sekali belajar dari 64 sampel
        
        self.model = DQN(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.choice(self.action_space) if hasattr(self, 'action_space') else random.choice([0, 1, 2])
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            action_values = self.model(state_tensor)
        return torch.argmax(action_values[0]).item()

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        minibatch = random.sample(self.memory, self.batch_size)
        
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(device)
                target = (reward + self.gamma * torch.max(self.model(next_state_tensor)[0]).item())
            
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            target_f = self.model(state_tensor)
            target_f_clone = target_f.clone()
            target_f_clone[0][action] = target
            
            self.optimizer.zero_grad()
            loss = self.criterion(target_f, target_f_clone)
            loss.backward()
            self.optimizer.step()
            
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# 6. Eksekusi Utama
if __name__ == "__main__":
    NAMA_FILE_CSV = "dataset_olist_siap_train.csv" 
    
    try:
        df_real = load_real_data(NAMA_FILE_CSV, target_product="aca2eb7d00ea1a7b8ebd4e68314663af")
        env = DynamicPricingEnv(df_real)
        
        state_size = 4
        action_size = 3
        agent = DQNAgent(state_size, action_size)
        
        episodes = 300 # NAIK JADI 300 EPISODE
        scores = []
        
        print("\nMemulai Training DQN (300 Episode)...")
        print("Bisa ditinggal ngopi dulu bentar, GPU lagi kerja keras!")
        print("-" * 50)
        
        for e in range(episodes):
            state = env.reset()
            total_profit = 0
            done = False
            
            while not done:
                action = agent.act(state)
                next_state, reward, done, profit = env.step(action)
                total_profit += profit
                
                agent.remember(state, action, reward, next_state, done)
                state = next_state
                agent.replay()
                
            scores.append(total_profit)
            
            # Print log setiap 20 episode
            if (e + 1) % 20 == 0 or e == 0:
                print(f"Episode: {e+1:03d}/{episodes} | Total Profit: Rp {int(total_profit):,} | Eksplorasi: {agent.epsilon:.3f}")
            
        print("\nTraining Selesai! Menyiapkan grafik hasil...")
        
        # 7. Plotting Hasil Training (Lebih Halus)
        plt.figure(figsize=(12, 6))
        # Plot garis biru tipis untuk data mentah
        plt.plot(range(1, episodes + 1), scores, marker='', linestyle='-', color='b', alpha=0.3, label='Profit per Episode')
        
        # Tambahkan Moving Average (Garis merah) biar trennya kelihatan jelas
        window_size = 20
        moving_avg = pd.Series(scores).rolling(window=window_size).mean()
        plt.plot(range(1, episodes + 1), moving_avg, color='red', linewidth=2, label=f'Trend (MA {window_size})')
        
        plt.title('DQN Training Performance - Dynamic Pricing (Tuned)', fontsize=14)
        plt.xlabel('Episode', fontsize=12)
        plt.ylabel('Total Profit (Rp)', fontsize=12)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        plt.savefig('dqn_training_result_tuned.png')
        print("Grafik berhasil disimpan sebagai 'dqn_training_result_tuned.png' di folder project.")
        plt.show()

    except Exception as e:
        print(f"Terjadi Kesalahan: {e}")