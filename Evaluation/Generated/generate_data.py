import pandas as pd
import numpy as np
import os

np.random.seed(2025)

# Parameters
num_timesteps = 51
num_samples_per_timestep = 1000
base_prob = 0.05  # initial connection probability
prob_increase = (1.0 - base_prob) / num_timesteps  # probability increase per timestep

data = []

src_id = 1
tgt_id = 100000
helper_id = 200000
rand_id = 300000
# tgt_id = src_id + num_timesteps * num_samples_per_timestep
# helper_id = tgt_id + num_timesteps * num_samples_per_timestep
# rand_id = helper_id + num_timesteps * num_samples_per_timestep
for t in range(1, num_timesteps):
    for i in range(num_samples_per_timestep):
        prob = min(base_prob + t * prob_increase, 1.0)
        # connected = np.random.rand() < prob
        data.append({
            'time': t+2,
            'node1': src_id,
            'node2': tgt_id,
            'connected': 1,
            'label': t+2
        })
        #Randomly set timestamp of helper edge to be before or at the same time as src-tgt edge
        ts = int((t + 1) * np.random.rand())
        data.append({
            'time': ts,
            'node1': helper_id,
            'node2': tgt_id,
            'connected': 1, # if connect_to_src else 1,
            'label': ''
        })
        for id in [src_id, tgt_id]:
            for _ in range(np.random.randint(0, 4)):
                data.append({
                    'time': t+1,
                    'node1': rand_id,
                    'node2': id,
                    'connected': 1,
                    'label': ''
                })
                curr_rand_id = rand_id
                rand_id += 1
                for _ in range(np.random.randint(0, 4)):
                    data.append({
                        'time': t,
                        'node1': rand_id,
                        'node2': curr_rand_id,
                        'connected': 1,
                        'label': ''
                    })
                    rand_id += 1
        src_id += 1
        tgt_id += 1
        helper_id += 1

df = pd.DataFrame(data)
df = df.rename(columns={'node1': 'u', 'node2': 'i', 'time': 'ts'})
#Remove non-connected edges
df = df[df['connected'] == 1].reset_index(drop=True)
df = df.drop(columns=['connected'])

# Add index column
df = df.reset_index(names='idx')    
df["idx"] = df["idx"] + 1 


# Add numpy array with random edge features
#edge_features = np.random.rand(len(df) + 1, 1)  # random features for each edge
edge_features = np.ones((len(df) + 1, 1))  # features for each edge
# Identify helper events by features
edge_features[1:,:][(df['u'] < 3 * num_timesteps * num_samples_per_timestep + 1) & (df['u'] > 1 * num_timesteps * num_samples_per_timestep), 0] = 2.0  # First feature indicates helper edges
# Add numpy array with random node features
node_features = np.zeros((rand_id+2, 3))  # 3 zero features for each node
# Add node names 
node_names = np.full((rand_id+2,), 'rnd', dtype=object)
node_names[1:100000] = 'A'
node_names[100000:200000] = 'B'
node_names[200000:300000] = 'H'


#Save data
os.makedirs("Data/generated", exist_ok=True)
df.to_csv("Data/generated/ml_generated.csv")
np.save("Data/generated/ml_generated.npy", edge_features)
np.save("Data/generated/ml_generated_node.npy", node_features)
np.save("Data/generated/generated_node_names.npy", node_names)
print(f"Generated data saved to Data/generated/ml_generated.csv and corresponding features. {len(df)} edges created.")