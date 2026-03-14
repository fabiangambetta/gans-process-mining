import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import csv
from torch.utils.data import DataLoader, TensorDataset
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from datetime import datetime
import re
import time
import argparse
import itertools

def get_device():
    if torch.backends.mps.is_available():
        print("✅ Usando aceleración GPU Apple Metal (MPS)")
        return torch.device("mps")
    else:
        print("⚠️ MPS no disponible. Usando CPU")
        return torch.device("cpu")

DEVICE = get_device()

class TracePreprocessor2:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        
        self.vocabs = []
        self.stoi = []
        self.itos = []
        self.dims = []
        
        self.num_attributes = 0
        self.max_len = 0
        self.total_dim = 0
        
    def load_and_process(self):
        with open(self.csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            raw_rows = list(reader)
        
        parsed_data = []
        raw_attribute_sets = []
        
        for r_idx, row in enumerate(raw_rows):
            parsed_row = []
            for item in row:
                item = item.strip()
                if not item:
                    continue
                    
                if ":" in item:
                    parts = item.split(":")
                    parts = [p.strip() for p in parts]
                else:
                    parts = [item]
                
                if self.num_attributes == 0:
                    self.num_attributes = len(parts)
                    raw_attribute_sets = [set() for _ in range(self.num_attributes)]
                
                if len(parts) == self.num_attributes:
                    parsed_row.append(parts)
                    for i, val in enumerate(parts):
                        raw_attribute_sets[i].add(val)
            
            parsed_data.append(parsed_row)
            if len(parsed_row) > 0:
                self.max_len = max(self.max_len, len(parsed_row))
        
        for attr_set in raw_attribute_sets:
            vocab = ["<PAD>"] + sorted(list(attr_set))
            self.vocabs.append(vocab)
            self.stoi.append({ch:i for i,ch in enumerate(vocab)})
            self.itos.append({i:ch for i,ch in enumerate(vocab)})
            self.dims.append(len(vocab))
            
        self.total_dim = sum(self.dims)
        
        print(f"Detectados {self.num_attributes} atributos por evento.")
        for i, dim in enumerate(self.dims):
            print(f" - Atributo {i+1} Vocabulario ({dim}): {self.vocabs[i][:5]} ...")
        print(f"Longitud máxima: {self.max_len} | Dimensión Vector Total: {self.total_dim}")

        data_matrix = np.zeros((len(parsed_data), self.max_len, self.total_dim), dtype=np.float32)
        
        for i, row in enumerate(parsed_data):
            for t, event_parts in enumerate(row):
                current_offset = 0
                for attr_idx, val in enumerate(event_parts):
                    if val in self.stoi[attr_idx]:
                        vocab_idx = self.stoi[attr_idx][val]
                        data_matrix[i, t, current_offset + vocab_idx] = 1.0
                    
                    current_offset += self.dims[attr_idx]
            
            for t in range(len(row), self.max_len):
                current_offset = 0
                for attr_idx in range(self.num_attributes):
                    pad_idx = self.stoi[attr_idx]["<PAD>"]
                    data_matrix[i, t, current_offset + pad_idx] = 1.0
                    current_offset += self.dims[attr_idx]
                
        return torch.tensor(data_matrix).to(DEVICE)

    def decode_traces(self, generated_data):
        if isinstance(generated_data, torch.Tensor):
            generated_data = generated_data.cpu().detach().numpy()
            
        decoded_traces = []
        for i in range(len(generated_data)):
            trace = []
            for t in range(len(generated_data[i])):
                vector = generated_data[i][t]
                
                recovered_parts = []
                current_offset = 0
                
                for attr_idx in range(self.num_attributes):
                    dim = self.dims[attr_idx]
                    sub_vector = vector[current_offset : current_offset + dim]
                    
                    idx = np.argmax(sub_vector)
                    val_str = self.itos[attr_idx][idx]
                    recovered_parts.append(val_str)
                    
                    current_offset += dim
                
                if recovered_parts[0] == "<PAD>":
                    break
                
                if self.num_attributes == 1:
                    trace.append(recovered_parts[0])
                else:
                    trace.append(":".join(recovered_parts))
                
            decoded_traces.append(trace)
        return decoded_traces

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
        
    def load_and_process(self):
        with open(self.csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            raw_rows = list(reader)
            
        if len(raw_rows) > 0 and not any(char.isdigit() for char in raw_rows[0][0]):
            print("⚠️ Encabezado detectado y omitido.")
            raw_rows = raw_rows[1:]
        
        parsed_data = []
        raw_attribute_sets = [] 
        
        for r_idx, row in enumerate(raw_rows):
            parsed_row = []
            prev_times = {} 
            
            for t, item in enumerate(row):
                item = item.strip()
                if not item:  
                    continue
                
                ts_match = re.search(r'\d{4}-\d{2}-\d{2}T', item)
                
                if ts_match:
                    ts_start_idx = ts_match.start()

                    cat_parts_str = item[:ts_start_idx].rstrip(':')
                    cat_parts = cat_parts_str.split(':') if cat_parts_str else []
                    
                    timestamp_str = item[ts_start_idx:]
                    
                    parts = cat_parts + [timestamp_str]
                else:
                    parts = item.split(":")
                
                parts = [p.strip() for p in parts]

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
                            ts = parser.parse(val).timestamp()
                            
                            if t == 0:
                                delta = 0.0 
                                self.trace_start_times.append(ts) 
                            else:
                                delta = max(0.0, ts - prev_times[i]) 
                            
                            prev_times[i] = ts
                            
                            log_delta = np.log(delta + 1.0)
                            
                            if log_delta < self.log_delta_mins[i]: self.log_delta_mins[i] = log_delta
                            if log_delta > self.log_delta_maxs[i]: self.log_delta_maxs[i] = log_delta
            
            parsed_data.append(parsed_row)
            if len(parsed_row) > 0:
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
        
        print(f"Detectados {self.num_attributes} atributos por evento.")
        for i, dim in enumerate(self.dims):
            if self.attr_types[i] == 'time':
                print(f" - Atributo {i+1} [Timestamp Relativo] Log-Delta Max: {self.log_delta_maxs[i]:.2f}")
        print(f"Longitud máxima: {self.max_len} | Dimensión Vector Total: {self.total_dim}")

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
                        ts = parser.parse(val).timestamp()
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

    def decode_traces(self, generated_data):
        if isinstance(generated_data, torch.Tensor):
            generated_data = generated_data.cpu().detach().numpy()
            
        import random
        from datetime import datetime
        decoded_traces = []
        
        for i in range(len(generated_data)):
            trace = []
            
            base_start_time = random.choice(self.trace_start_times) if self.trace_start_times else datetime.now().timestamp()
            current_simulated_times = {idx: base_start_time for idx in range(self.num_attributes) if self.attr_types[idx] == 'time'}
            
            for t in range(len(generated_data[i])):
                vector = generated_data[i][t]
                recovered_parts = []
                current_offset = 0
                is_pad = False
                
                for attr_idx in range(self.num_attributes):
                    dim = self.dims[attr_idx]
                    sub_vector = vector[current_offset : current_offset + dim]
                    
                    if self.attr_types[attr_idx] == 'cat':
                        idx = np.argmax(sub_vector)
                        val_str = self.itos[attr_idx][idx]
                        if val_str == "<PAD>" and attr_idx == 0: 
                            is_pad = True
                        recovered_parts.append(val_str)
                        
                    elif self.attr_types[attr_idx] == 'time':
                        norm_val = np.clip(sub_vector[0], 0.0, 1.0) 
                        log_delta = norm_val * (self.log_delta_maxs[attr_idx] - self.log_delta_mins[attr_idx]) + self.log_delta_mins[attr_idx]
                        delta = np.exp(log_delta) - 1.0
                        
                        current_simulated_times[attr_idx] += delta
                        

                        dt_obj = datetime.fromtimestamp(current_simulated_times[attr_idx])
                        val_str = dt_obj.strftime("%Y-%m-%dT%H:%M:%S.000-03:00")
                        
                        recovered_parts.append(val_str)

                    current_offset += dim
                
                if is_pad:
                    break 
                
                if self.num_attributes == 1:
                    trace.append(recovered_parts[0])
                else:
                    trace.append(":".join(recovered_parts))
                
            if len(trace) > 0:
                decoded_traces.append(trace)
                
        return decoded_traces

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
        
    def load_and_process(self):
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
                if not item:  
                    continue
                    
                parts = [p.strip() for p in item.split(":")] if ":" in item else [item]
                
                if self.num_attributes == 0:
                    self.num_attributes = len(parts)
                    raw_attribute_sets = [set() for _ in range(self.num_attributes)]
                    
                    for i, val in enumerate(parts):
                        try:
                            datetime.fromisoformat(val)
                            self.attr_types.append('time')
                            self.log_delta_mins[i] = float('inf')
                            self.log_delta_maxs[i] = float('-inf')
                        except ValueError:
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
                                self.trace_start_times.append(ts)
                            else:
                                delta = max(0.0, ts - prev_times[i])
                            
                            prev_times[i] = ts
                            
                            log_delta = np.log(delta + 1.0)
                            
                            if log_delta < self.log_delta_mins[i]: self.log_delta_mins[i] = log_delta
                            if log_delta > self.log_delta_maxs[i]: self.log_delta_maxs[i] = log_delta
            
            parsed_data.append(parsed_row)
            if len(parsed_row) > 0:
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
        
        print(f"Detectados {self.num_attributes} atributos por evento.")
        for i, dim in enumerate(self.dims):
            if self.attr_types[i] == 'time':
                print(f" - Atributo {i+1} [Timestamp Relativo] Log-Delta Max: {self.log_delta_maxs[i]:.2f}")
        print(f"Longitud máxima: {self.max_len} | Dimensión Vector Total: {self.total_dim}")

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

    def decode_traces(self, generated_data):
        if isinstance(generated_data, torch.Tensor):
            generated_data = generated_data.cpu().detach().numpy()
            
        import random
        decoded_traces = []
        
        for i in range(len(generated_data)):
            trace = []
            
            base_start_time = random.choice(self.trace_start_times) if self.trace_start_times else datetime.now().timestamp()
            current_simulated_times = {idx: base_start_time for idx in range(self.num_attributes) if self.attr_types[idx] == 'time'}
            
            for t in range(len(generated_data[i])):
                vector = generated_data[i][t]
                recovered_parts = []
                current_offset = 0
                
                is_pad = False
                
                for attr_idx in range(self.num_attributes):
                    dim = self.dims[attr_idx]
                    sub_vector = vector[current_offset : current_offset + dim]
                    
                    if self.attr_types[attr_idx] == 'cat':
                        idx = np.argmax(sub_vector)
                        val_str = self.itos[attr_idx][idx]
                        if val_str == "<PAD>" and attr_idx == 0: 
                            is_pad = True
                        recovered_parts.append(val_str)
                        
                    elif self.attr_types[attr_idx] == 'time':
                        norm_val = np.clip(sub_vector[0], 0.0, 1.0) 
                        
                        log_delta = norm_val * (self.log_delta_maxs[attr_idx] - self.log_delta_mins[attr_idx]) + self.log_delta_mins[attr_idx]
                        
                        delta = np.exp(log_delta) - 1.0
                        
                        current_simulated_times[attr_idx] += delta
                        
                        val_str = datetime.fromtimestamp(current_simulated_times[attr_idx]).isoformat(timespec='milliseconds')
                        recovered_parts.append(val_str)

                    current_offset += dim
                
                if is_pad:
                    break
                
                if self.num_attributes == 1:
                    trace.append(recovered_parts[0])
                else:
                    trace.append(":".join(recovered_parts))
                
            if len(trace) > 0:
                decoded_traces.append(trace)
                
        return decoded_traces

class TracePreprocessor:
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
        
    def load_and_process(self):
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
                self.vocabs.append(None); self.stoi.append(None); self.itos.append(None)
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
        
        print(f"\n✅ Detectados {self.num_attributes} atributos por evento.")
        for i, dim in enumerate(self.dims):
            if self.attr_types[i] == 'cat':
                muestra_vocab = self.vocabs[i][:5]
                print(f" - Atributo {i+1} (Categórico) | Vocabulario ({dim}): {muestra_vocab} ...")
            elif self.attr_types[i] == 'time':
                print(f" - Atributo {i+1} (Timestamp) | Dimensión (1) | Rango Log-Delta: [{self.log_delta_mins[i]:.2f}, {self.log_delta_maxs[i]:.2f}]")
        print(f"\n📊 Longitud máxima de secuencia: {self.max_len} | Dimensión del Vector Total: {self.total_dim}\n")
                
        return torch.tensor(data_matrix).to(DEVICE)

    def decode_traces(self, generated_data):
        if isinstance(generated_data, torch.Tensor):
            generated_data = generated_data.cpu().detach().numpy()
            
        decoded_traces = []
        
        for i in range(len(generated_data)):
            trace = []
            
            base_start_time = self.trace_start_times[i % len(self.trace_start_times)] if self.trace_start_times else datetime.now().timestamp()
            current_simulated_times = {idx: base_start_time for idx in range(self.num_attributes) if self.attr_types[idx] == 'time'}
            for t in range(len(generated_data[i])):
                vector = generated_data[i][t]
                recovered_parts = []
                current_offset = 0
                is_pad = False
                
                for attr_idx in range(self.num_attributes):
                    dim = self.dims[attr_idx]
                    sub_vector = vector[current_offset : current_offset + dim]
                    
                    if self.attr_types[attr_idx] == 'cat':
                        idx = np.argmax(sub_vector)
                        val_str = self.itos[attr_idx][idx]
                        if val_str == "<PAD>" and attr_idx == 0: 
                            is_pad = True
                        recovered_parts.append(val_str)
                        
                    elif self.attr_types[attr_idx] == 'time':
                        norm_val = np.clip(sub_vector[0], 0.0, 1.0) 
                        
                        log_delta = norm_val * (self.log_delta_maxs[attr_idx] - self.log_delta_mins[attr_idx]) + self.log_delta_mins[attr_idx]
                        delta = np.exp(log_delta) - 1.0
                        
                        current_simulated_times[attr_idx] += delta
                        
                        val_str = datetime.fromtimestamp(current_simulated_times[attr_idx]).astimezone().isoformat(timespec='milliseconds')
                        recovered_parts.append(val_str)

                    current_offset += dim
                
                if is_pad: break 
                
                trace.append(":".join(recovered_parts) if self.num_attributes > 1 else recovered_parts[0])
                
            if len(trace) > 0:
                decoded_traces.append(trace)
                
        return decoded_traces

class TimeGAN_Module(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim) 

    def forward(self, x):
        out, _ = self.rnn(x) 
        return self.linear(out) 

class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super().__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(hidden_dim * 2, 1) 

    def forward(self, x):
        out, _ = self.rnn(x)
        return torch.sigmoid(self.linear(out))


class TimeGAN:
    def __init__(self, feature_dim, max_seq_len, hidden_dim, z_dim, lr_autoencoder=1e-3, lr_discriminator=1e-3, lr_generator=1e-3, batch_size=128, num_layers=3):
        self.feature_dim = feature_dim
        self.max_seq_len = max_seq_len
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        
        self.embedder = TimeGAN_Module(feature_dim, hidden_dim, hidden_dim, num_layers).to(DEVICE)
        self.recovery = TimeGAN_Module(hidden_dim, hidden_dim, feature_dim, num_layers).to(DEVICE)
        self.generator = TimeGAN_Module(z_dim, hidden_dim, hidden_dim, num_layers).to(DEVICE)
        self.supervisor = TimeGAN_Module(hidden_dim, hidden_dim, hidden_dim, num_layers-1).to(DEVICE)
        self.discriminator = Discriminator(hidden_dim, hidden_dim, num_layers).to(DEVICE)

        self.e_opt = optim.Adam(list(self.embedder.parameters()) + list(self.recovery.parameters()), lr = lr_autoencoder)
        self.d_opt = optim.Adam(self.discriminator.parameters(), lr=lr_discriminator)
        self.g_opt = optim.Adam(list(self.generator.parameters()) + list(self.supervisor.parameters()), lr=lr_generator)

        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()

    def train(self, dataset, epochs_f1=300, epochs_f2=300, epochs_f3=300, loss_log_path=None):
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        print("🚀 Iniciando entrenamiento TimeGAN...")

        log_file = open(loss_log_path, 'w', encoding='utf-8') if loss_log_path else None
        if log_file:
            log_file.write("fase,epoca,loss_e,loss_s,loss_d,loss_g\n")

        print("\n--- Fase 1: Embedding & Recovery ---")
        
        for epoch in range(epochs_f1): 
            for X_batch in dataloader:
                X = X_batch[0]
                self.e_opt.zero_grad()
                
                H = self.embedder(X)
                X_tilde = self.recovery(H)
                
                loss_e = self.mse_loss(X_tilde, X)
                loss_e.backward()
                self.e_opt.step()
            
            if log_file: log_file.write(f"1,{epoch},{loss_e.item():.6f},,,\n")
            
            if epoch % 50 == 0:
                print(f"Epoch {epoch}: Loss E: {loss_e.item():.4f}")

        print("\n--- Fase 2: Entrenamiento Supervisado ---")
        
        for epoch in range(epochs_f2):
            for X_batch in dataloader:
                X = X_batch[0]
                self.g_opt.zero_grad()
                
                H = self.embedder(X).detach()
                H_hat_sup = self.supervisor(H) 
                
                loss_s = self.mse_loss(H_hat_sup[:, :-1, :], H[:, 1:, :])
                loss_s.backward()
                self.g_opt.step()
            
            if log_file: log_file.write(f"2,{epoch},,{loss_s.item():.6f},,\n")
                
            if epoch % 50 == 0:
                print(f"Epoch {epoch}: Loss S: {loss_s.item():.4f}")

        print("\n--- Fase 3: Entrenamiento Conjunto ---")
        
        for epoch in range(epochs_f3):
            for X_batch in dataloader:
                X = X_batch[0]
                batch_curr = X.size(0)
                seq_len = X.size(1)

                Z = torch.randn(batch_curr, seq_len, self.z_dim).to(DEVICE)

                self.g_opt.zero_grad()
                
                H_hat = self.generator(Z)
                H_hat_sup = self.supervisor(H_hat)
                H_real = self.embedder(X).detach()

                Y_fake = self.discriminator(H_hat_sup)
                
                loss_g_u = self.bce_loss(Y_fake, torch.ones_like(Y_fake))
                loss_g_s = self.mse_loss(H_hat_sup[:, :-1, :], H_hat[:, 1:, :])
                loss_v = self.mse_loss(H_hat_sup.mean(0), H_real.mean(0)) + self.mse_loss(H_hat_sup.std(0), H_real.std(0))

                loss_gen = loss_g_u + 10 * loss_g_s + 50 * loss_v
                loss_gen.backward()
                self.g_opt.step()

                self.d_opt.zero_grad()
                
                H_hat = self.generator(Z).detach()
                H_hat_sup = self.supervisor(H_hat).detach()
                H_real = self.embedder(X).detach()

                Y_fake = self.discriminator(H_hat_sup)
                Y_real = self.discriminator(H_real)

                loss_d_fake = self.bce_loss(Y_fake, torch.zeros_like(Y_fake))
                loss_d_real = self.bce_loss(Y_real, torch.full_like(Y_real, 0.9))
                
                loss_d = loss_d_fake + loss_d_real
                loss_d.backward()
                self.d_opt.step()

            if log_file: log_file.write(f"3,{epoch},,,{loss_d.item():.6f},{loss_gen.item():.6f}\n")

            if epoch % 50 == 0:
                print(f"Epoch {epoch}: Loss D: {loss_d.item():.4f} | Loss G: {loss_gen.item():.4f}")

    def generate(self, num_samples):
        self.generator.eval()
        self.recovery.eval()
        self.supervisor.eval()
        
        with torch.no_grad():
            Z = torch.randn(num_samples, self.max_seq_len, self.z_dim).to(DEVICE)
            H_hat = self.generator(Z)
            H_hat_sup = self.supervisor(H_hat)
            X_hat = self.recovery(H_hat_sup)
            
        return X_hat

def visualizar_datos(ori_data, generated_data, analysis='pca', output_name='plot'):
    if isinstance(ori_data, torch.Tensor):
        ori_data = ori_data.cpu().detach().numpy()
    if isinstance(generated_data, torch.Tensor):
        generated_data = generated_data.cpu().detach().numpy()

    anal_sample_no = min([1000, len(ori_data), len(generated_data)])
    idx = np.random.permutation(len(ori_data))[:anal_sample_no]

    ori_data = ori_data[idx]
    generated_data = generated_data[:anal_sample_no]

    no, seq_len, dim = ori_data.shape

    prep_data = np.mean(ori_data, axis=1)
    prep_data_hat = np.mean(generated_data, axis=1)

    colors = ["red" for i in range(anal_sample_no)] + ["blue" for i in range(anal_sample_no)]

    if analysis == 'pca':
        pca = PCA(n_components=2)
        pca.fit(prep_data)
        pca_results = pca.transform(prep_data)
        pca_hat_results = pca.transform(prep_data_hat)

        f, ax = plt.subplots(1)
        plt.scatter(pca_results[:, 0], pca_results[:, 1],
                    c=colors[:anal_sample_no], alpha=0.2, label="Original")
        plt.scatter(pca_hat_results[:, 0], pca_hat_results[:, 1],
                    c=colors[anal_sample_no:], alpha=0.2, label="Sintético")

        ax.legend()
        plt.title('PCA plot')
        plt.xlabel('x-pca')
        plt.ylabel('y_pca')
        filename = f'{output_name}_pca.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Guardado: {filename}")

    elif analysis == 'tsne':
        prep_data_final = np.concatenate((prep_data, prep_data_hat), axis=0)

        tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=300)
        tsne_results = tsne.fit_transform(prep_data_final)

        f, ax = plt.subplots(1)
        plt.scatter(tsne_results[:anal_sample_no, 0], tsne_results[:anal_sample_no, 1],
                    c=colors[:anal_sample_no], alpha=0.2, label="Original")
        plt.scatter(tsne_results[anal_sample_no:, 0], tsne_results[anal_sample_no:, 1],
                    c=colors[anal_sample_no:], alpha=0.2, label="Sintético")

        ax.legend()
        plt.title('t-SNE plot')
        plt.xlabel('x-tsne')
        plt.ylabel('y-tsne')
        filename = f'{output_name}_tsne.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Guardado: {filename}")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Entrenamiento de TimeGAN con Búsqueda de Hiperparámetros.")
    
    parser.add_argument("--file", type=str, required=True, help="Archivo CSV de entrada")
    parser.add_argument("--out", type=str, default="output", help="Nombre base para archivos de salida")
    
    parser.add_argument("--hidden_dim", type=int, nargs='+', default=[24], help="Dimensión latente (ej: --hidden_dim 24 32 64)")
    parser.add_argument("--z_dim", type=int, default=None, help="Dimensión del ruido (Se igualará al hidden_dim de la iteración si no se pasa)")
    
    parser.add_argument("--lr_e", type=float, nargs='+', default=[1e-3], help="Learning Rate Fase 1 (Autoencoder)")
    parser.add_argument("--lr_d", type=float, nargs='+', default=[1e-3], help="Learning Rate Discriminador")
    parser.add_argument("--lr_g", type=float, nargs='+', default=[1e-3], help="Learning Rate Generador")

    parser.add_argument("--batch_size", type=int, nargs='+', default=128, help="Tamaño del batch")
    parser.add_argument("--num_layers", type=int, nargs='+', default=3, help="Número de capas")
    
    parser.add_argument("--epochs_f1", type=int, default=300, help="Épocas para entrenear auto encoder")
    parser.add_argument("--epochs_f2", type=int, default=300, help="Épocas para entrenar supervisor")
    parser.add_argument("--epochs_f3", type=int, default=300, help="Épocas para entrenar conjunto (generador y discriminador)")
    
    parser.add_argument("--num_new", type=int, default=100, help="Cantidad de trazas a generar")

    args = parser.parse_args()

    try:
        FILE_NAME = args.file
        print(f"\n📂 Cargando {FILE_NAME}...")
        preprocessor = TracePreprocessor(FILE_NAME)
        data_tensor = preprocessor.load_and_process()
        dataset = TensorDataset(data_tensor)

        combinaciones = list(itertools.product(
            args.hidden_dim, 
            args.lr_e, 
            args.lr_d, 
            args.lr_g,
            args.batch_size,
            args.num_layers
        ))
        
        print(f"\n🔍 Iniciando Búsqueda de Hiperparámetros: Se ejecutarán {len(combinaciones)} combinaciones en total.")

        for idx, (h_dim, lr_e, lr_d, lr_g, batch_size, num_layers ) in enumerate(combinaciones, 1):
            
            z_dim_actual = args.z_dim if args.z_dim is not None else h_dim
            
            sufijo_exp = f"hdim{h_dim}_lre{lr_e}_lrd{lr_d}_lrg{lr_g}_bs{batch_size}_nl{num_layers}"
            base_out_name = f"{args.out}_{idx}"

            print(f"\n" + "="*50)
            print(f"🚀 EXPERIMENTO {idx}/{len(combinaciones)}: {sufijo_exp}")
            print(f"⚙️ Red: HIDDEN={h_dim}, Z_DIM={z_dim_actual} | LRs: E={lr_e}, D={lr_d}, G={lr_g}")
            print("="*50)

            start_time = time.time()

            timegan = TimeGAN(
                feature_dim=preprocessor.total_dim, 
                max_seq_len=preprocessor.max_len, 
                hidden_dim=h_dim, 
                z_dim=z_dim_actual,
                lr_autoencoder=lr_e,
                lr_discriminator=lr_d,
                lr_generator=lr_g,
                batch_size=batch_size,
                num_layers=num_layers
            )

            loss_file_path = f"{base_out_name}_losses.csv"
            
            timegan.train(
                dataset, 
                epochs_f1=args.epochs_f1, 
                epochs_f2=args.epochs_f2, 
                epochs_f3=args.epochs_f3,
                loss_log_path=loss_file_path
            )

            print(f"\nGenerando {args.num_new} trazas sintéticas...")
            synthetic_data = timegan.generate(args.num_new)
            
            visualizar_datos(data_tensor, synthetic_data, 'pca', base_out_name)
            visualizar_datos(data_tensor, synthetic_data, 'tsne', base_out_name)

            synthetic_traces = preprocessor.decode_traces(synthetic_data)
            out_file = f"{base_out_name}.csv"
            
            with open(out_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerows(synthetic_traces)
            end_time = time.time()
            exec_time_seconds = end_time - start_time
            
            metadata_file = f"{base_out_name}_metadata.txt"
            with open(metadata_file, "w", encoding="utf-8") as f_meta:
                f_meta.write(f"=== METADATOS DEL EXPERIMENTO ===\n")
                f_meta.write(f"Fecha de ejecución: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f_meta.write(f"Tiempo de entrenamiento y generación: {exec_time_seconds:.2f} segundos ({exec_time_seconds/60:.2f} minutos)\n\n")
                
                f_meta.write(f"=== HIPERPARÁMETROS ===\n")
                f_meta.write(f"Hidden Dim: {h_dim}\n")
                f_meta.write(f"Z Dim (Ruido): {z_dim_actual}\n")
                f_meta.write(f"Learning Rate E (Autoencoder): {lr_e}\n")
                f_meta.write(f"Learning Rate D (Discriminador): {lr_d}\n")
                f_meta.write(f"Learning Rate G (Generador): {lr_g}\n")
                f_meta.write(f"Épocas Fase 1 (Autoencoder): {args.epochs_f1}\n")
                f_meta.write(f"Épocas Fase 2 (Supervisor): {args.epochs_f2}\n")
                f_meta.write(f"Épocas Fase 3 (Conjunto): {args.epochs_f3}\n")
                f_meta.write(f"Batch Size: {batch_size}\n")
                f_meta.write(f"Capas RNN (Num Layers): {num_layers}\n\n")
                
                f_meta.write(f"=== DATOS DEL DATASET ===\n")
                f_meta.write(f"Archivo de entrada: {FILE_NAME}\n")
                f_meta.write(f"Trazas originales utilizadas: {len(data_tensor)}\n")
                f_meta.write(f"Trazas sintéticas generadas: {args.num_new}\n")
                f_meta.write(f"Atributos por evento detectados: {preprocessor.num_attributes}\n")
                f_meta.write(f"Longitud máxima de la secuencia (Max Len): {preprocessor.max_len}\n")
                f_meta.write(f"Dimensión del vector (Total Dim One-Hot): {preprocessor.total_dim}\n\n")

                f_meta.write(f"=== HARDWARE ===\n")
                f_meta.write(f"Dispositivo utilizado: {DEVICE}\n")

            weights_path = f"{base_out_name}_weights.pth"
            weights_dict = {
                'generator_state': timegan.generator.state_dict(),
                'supervisor_state': timegan.supervisor.state_dict(),
                'recovery_state': timegan.recovery.state_dict(),
                'architecture_params': {
                    'feature_dim': preprocessor.total_dim,
                    'max_seq_len': preprocessor.max_len,
                    'hidden_dim': h_dim,
                    'z_dim': z_dim_actual,
                    'num_layers': num_layers
                }
            }
            torch.save(weights_dict, weights_path)

            torch.save(data_tensor.cpu(), f"{base_out_name}_real_tensor.pt")
            torch.save(synthetic_data.cpu(), f"{base_out_name}_synth_tensor.pt")
                
            print(f"✅ Experimento {idx} completado. Resultados guardados con prefijo '{base_out_name}'")

        print(f"\n🎉 ¡Búsqueda de hiperparámetros finalizada con éxito!")

    except Exception as e:
        print(f"❌ Error inesperado: {e}")
        import traceback
        traceback.print_exc()