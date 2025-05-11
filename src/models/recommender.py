import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from collections import defaultdict
from tqdm import tqdm

class Recommender:
    def __init__(self):
        self.user_ids = {}
        self.item_ids = {}
        self.interaction_matrix = None
        self.weights = {'clicks': 1, 'carts': 2, 'orders': 3}
    
    def _get_or_add_id(self, key, id_dict):
        if key not in id_dict:
            id_dict[key] = len(id_dict)
        return id_dict[key]
    
    def fit(self, df, batch_size=100):  # Move batch_size to fit parameter
        print("Processing data in batches...")
        interactions = defaultdict(lambda: defaultdict(float))
        
        # Process in batches using the passed batch_size parameter
        for start in tqdm(range(0, len(df), batch_size)):  # Use parameter directly
            end = min(start + batch_size, len(df))
            batch = df.iloc[start:end]
            
            for _, row in batch.iterrows():
                user_id = self._get_or_add_id(row['session'], self.user_ids)
                
                for event in row['events']:
                    item_id = self._get_or_add_id(event['aid'], self.item_ids)
                    weight = self.weights.get(event['type'], 0)
                    interactions[user_id][item_id] += weight
            
            if len(interactions) > 10000:
                self._build_matrix(interactions)
                interactions.clear()
        
        if interactions:
            self._build_matrix(interactions)
        
        print(f"Final matrix shape: {self.interaction_matrix.shape}")
        return self