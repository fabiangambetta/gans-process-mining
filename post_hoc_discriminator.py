import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import csv
import re
import argparse
from datetime import datetime
from torch.utils.data import DataLoader, TensorDataset, random_split

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

DEVICE = get_device()

class TracePreprocessorEvaluator:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.attr_types = []  
        self.vocabs = []      
        self.stoi = []        
        self.itos = []        
        self.dims = []        
        self.log_delta_mins = {}
        self.log_delta_maxs = {}
        self.trace_start_times = []
        self.num_attributes = 0 
        self.max_len = 0
        self.total_dim = 0      

    def load_and_process_real(self):
        with open(self.csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            raw_rows = list(reader)
            
        parsed_data = []
        raw_attribute_sets = [] 
        
        for r_idx, row in enumerate(raw_rows):
            parsed_row = []
            prev_times = {} 
            
            for t, item in enumerate(row):
                item = item.strip()
                if not item: continue
                
                ts_match = re.search(r'\d{4}-\d{2}-\d{2}T', item)
                
                if ts_match:
                    ts_start_idx = ts_match.start()
                    cat_parts_str = item[:ts_start_idx].rstrip(':')
                    cat_parts = cat_parts_str.split(':') if cat_parts_str else []
                    timestamp_str = item[ts_start_idx:]
                    parts = cat_parts + [timestamp_str]
                else:
                    parts = item.split(":")
                
                parts = [p.strip() for p in parts if p.strip()]
                
                if self.num_attributes == 0:
                    self.num_attributes = len(parts)
                    raw_attribute_sets = [set() for _ in range(self.num_attributes)]
                    
                    for i, val in enumerate(parts):
                        if re.search(r'\d{4}-\d{2}-\d{2}T', val):
                            self.attr_types.append('time')
                            self.log_delta_mins[i] = float('inf')
                            self.log_delta_maxs[i] = float('-inf')
                        else:
                            self.attr_types.append('cat')
                
                if len(parts) == self.num_attributes:
                    parsed_row.append(parts)
                    for i, val in enumerate(parts):
                        if self.attr_types[i] == 'cat':
                            raw_attribute_sets[i].add(val)
                        elif self.attr_types[i] == 'time':
                            ts = datetime.fromisoformat(val).timestamp()
                            
                            if t == 0:
                                delta = 0.0 
                                if r_idx == 0: 
                                    self.trace_start_times.append(ts) 
                            else:
                                delta = max(0.0, ts - prev_times[i]) 
                            
                            prev_times[i] = ts
                            
                            log_delta = np.log(delta + 1.0)
                            self.log_delta_mins[i] = min(self.log_delta_mins[i], log_delta)
                            self.log_delta_maxs[i] = max(self.log_delta_maxs[i], log_delta)
            
            if len(parsed_row) > 0:
                parsed_data.append(parsed_row)
                self.max_len = max(self.max_len, len(parsed_row))
        
        for attr_idx in range(self.num_attributes):
            if self.attr_types[attr_idx] == 'cat':
                vocab = ["<PAD>"] + sorted(list(raw_attribute_sets[attr_idx]))
                self.vocabs.append(vocab)
                self.stoi.append({ch:i for i,ch in enumerate(vocab)})
                self.itos.append({i:ch for i,ch in enumerate(vocab)})
                self.dims.append(len(vocab))
            else:
                self.vocabs.append(None)
                self.stoi.append(None)
                self.itos.append(None)
                self.dims.append(1) 
                
                if self.log_delta_maxs[attr_idx] == self.log_delta_mins[attr_idx]:
                    self.log_delta_maxs[attr_idx] += 1.0 
            
        self.total_dim = sum(self.dims)

        data_matrix = np.zeros((len(parsed_data), self.max_len, self.total_dim), dtype=np.float32)
        
        for i, row in enumerate(parsed_data):
            prev_times = {}
            for t, event_parts in enumerate(row):
                current_offset = 0
                for attr_idx, val in enumerate(event_parts):
                    if self.attr_types[attr_idx] == 'cat':
                        if val in self.stoi[attr_idx]:
                            vocab_idx = self.stoi[attr_idx][val]
                            data_matrix[i, t, current_offset + vocab_idx] = 1.0
                            
                    elif self.attr_types[attr_idx] == 'time':
                        ts = datetime.fromisoformat(val).timestamp()
                        delta = 0.0 if t == 0 else max(0.0, ts - prev_times[attr_idx])
                        prev_times[attr_idx] = ts
                        
                        log_delta = np.log(delta + 1.0)
                        norm_val = (log_delta - self.log_delta_mins[attr_idx]) / (self.log_delta_maxs[attr_idx] - self.log_delta_mins[attr_idx])
                        data_matrix[i, t, current_offset] = norm_val
                        
                    current_offset += self.dims[attr_idx]
            
            for t in range(len(row), self.max_len):
                current_offset = 0
                for attr_idx in range(self.num_attributes):
                    if self.attr_types[attr_idx] == 'cat':
                        pad_idx = self.stoi[attr_idx]["<PAD>"]
                        data_matrix[i, t, current_offset + pad_idx] = 1.0
                    elif self.attr_types[attr_idx] == 'time':
                        data_matrix[i, t, current_offset] = 0.0 
                    current_offset += self.dims[attr_idx]
                
        return torch.tensor(data_matrix).to(DEVICE)

    def load_and_process_synthetic(self, synth_csv_path):
        with open(synth_csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            raw_rows = list(reader)
            
        parsed_data = []
        for r_idx, row in enumerate(raw_rows):
            parsed_row = []
            for item in row:
                item = item.strip()
                if not item: continue
                
                ts_match = re.search(r'\d{4}-\d{2}-\d{2}T', item)
                if ts_match:
                    ts_start_idx = ts_match.start()
                    cat_parts_str = item[:ts_start_idx].rstrip(':')
                    cat_parts = cat_parts_str.split(':') if cat_parts_str else []
                    timestamp_str = item[ts_start_idx:]
                    parts = cat_parts + [timestamp_str]
                else:
                    parts = item.split(":")
                
                parts = [p.strip() for p in parts if p.strip()]
                
                if len(parts) == self.num_attributes:
                    parsed_row.append(parts)

            if len(parsed_row) > 0:
                parsed_data.append(parsed_row[:self.max_len])

        data_matrix = np.zeros((len(parsed_data), self.max_len, self.total_dim), dtype=np.float32)
        
        for i, row in enumerate(parsed_data):
            prev_times = {}
            for t, event_parts in enumerate(row):
                current_offset = 0
                for attr_idx, val in enumerate(event_parts):
                    if self.attr_types[attr_idx] == 'cat':
                        vocab_idx = self.stoi[attr_idx].get(val, self.stoi[attr_idx]["<PAD>"])
                        data_matrix[i, t, current_offset + vocab_idx] = 1.0
                            
                    elif self.attr_types[attr_idx] == 'time':
                        try:
                            ts = datetime.fromisoformat(val).timestamp()
                            delta = 0.0 if t == 0 else max(0.0, ts - prev_times.get(attr_idx, ts))
                            prev_times[attr_idx] = ts
                            
                            log_delta = np.log(delta + 1.0)
                            norm_val = (log_delta - self.log_delta_mins[attr_idx]) / (self.log_delta_maxs[attr_idx] - self.log_delta_mins[attr_idx])
                            norm_val = max(0.0, min(1.0, norm_val))
                            data_matrix[i, t, current_offset] = norm_val
                        except ValueError:
                            data_matrix[i, t, current_offset] = 0.0

                    current_offset += self.dims[attr_idx]
            
            for t in range(len(row), self.max_len):
                current_offset = 0
                for attr_idx in range(self.num_attributes):
                    if self.attr_types[attr_idx] == 'cat':
                        pad_idx = self.stoi[attr_idx]["<PAD>"]
                        data_matrix[i, t, current_offset + pad_idx] = 1.0
                    elif self.attr_types[attr_idx] == 'time':
                        data_matrix[i, t, current_offset] = 0.0 
                    current_offset += self.dims[attr_idx]
                
        return torch.tensor(data_matrix).to(DEVICE)

class PostHocDiscriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2):
        super().__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=False)
        self.linear = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = out[:, -1, :] 
        return torch.sigmoid(self.linear(out))

def train_and_evaluate(real_data, synth_data, hidden_dim=32, batch_size=64, epochs=100, lr=1e-3):
    labels_real = torch.ones(real_data.size(0), 1).to(DEVICE)
    labels_synth = torch.zeros(synth_data.size(0), 1).to(DEVICE)

    X_all = torch.cat([real_data, synth_data], dim=0)
    Y_all = torch.cat([labels_real, labels_synth], dim=0)

    dataset = TensorDataset(X_all, Y_all)
    
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = PostHocDiscriminator(input_dim=real_data.size(-1), hidden_dim=hidden_dim).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    print(f"\nEntrenando PostHoc Discriminator (Train: {train_size} | Test: {test_size})")
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        for X_batch, Y_batch in train_loader:
            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, Y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, Y_batch in test_loader:
            predictions = model(X_batch)
            predicted_labels = (predictions > 0.5).float()
            correct += (predicted_labels == Y_batch).sum().item()
            total += Y_batch.size(0)

    accuracy = correct / total
    print(f"Accuracy Final en Test Set: {accuracy * 100:.2f}%")
    
    return accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", type=str, required=True)
    parser.add_argument("--synth", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=32)
    args = parser.parse_args()

    try:
        preprocessor = TracePreprocessorEvaluator(args.real)
        real_tensor = preprocessor.load_and_process_real()
        
        synth_tensor = preprocessor.load_and_process_synthetic(args.synth)
        
        train_and_evaluate(
            real_data=real_tensor, 
            synth_data=synth_tensor,
            hidden_dim=args.hidden_dim,
            batch_size=args.batch_size,
            epochs=args.epochs
        )

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()