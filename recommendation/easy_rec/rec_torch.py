# import torch
# import pytorch_lightning as pl
# import multiprocessing
# from copy import deepcopy
# from . import model

# #TODO: pass this function inside easy_torch
# # Define a custom PyTorch Dataset class named DictDataset
# class DictDataset(torch.utils.data.Dataset):
#     """
#     Custom PyTorch Dataset class that takes a dictionary as input and returns items based on keys.

#     Args:
#         data (dict): Input dictionary containing data.

#     Returns:
#         dict: A dictionary where each key corresponds to a tensor item.
#     """
#     # Constructor to initialize the dataset with input data
#     def __init__(self, data):
#         self.data = data

#         # Convert each value in the data dictionary to a PyTorch tensor
#         # for key, value in self.data.items():
#         #     if isinstance(value, torch.Tensor):
#         #         self.data[key] = value.clone().detach()
#         #     else:
#         #         self.data[key] = torch.tensor(value)

#     # Method to get an item from the dataset at a given index
#     def __getitem__(self, index):
#         return {key: value[index] for key, value in self.data.items()}

#     # Method to get the length of the dataset
#     def __len__(self):
#         # Assumes that all values in the data dictionary have the same length
#         return len(self.data[list(self.data.keys())[0]])

# # class SequentialDataset(torch.utils.data.Dataset):
# #     def __init__(self, data):
# #         self.data = data
# #         super().__init__()

# #     def __getitem__(self, index):
# #         batch = super().__getitem__(index)
# #         for key,value in batch:
# #             print(key,value)
# #         print(sanaskjnas)

# # # Method to pair input and output sequences based on specified parameters
# # def pair_input_output(self, sequential_keys, padding_value, lookback, stride, lookforward, simultaneous_lookforward, out_seq_len, keep_last, drop_original=True):
# #     key_to_use = sequential_keys[0]
# #     max_len = self.data[key_to_use].shape[1]
# #     if out_seq_len is None: out_seq_len = max_len

# #     # Calculate input and output indices based on lookback, stride, and lookforward
# #     # input_indices = torch.stack([torch.arange(a-lookback,a) for a in range(max_len-lookforward, lookback-1, -stride)][::-1])
# #     input_indices = torch.stack([torch.arange(a-lookback,a) for a in range(max_len-lookforward-simultaneous_lookforward+1, max(lookback-1, max_len-lookforward-simultaneous_lookforward+1-out_seq_len), -stride)][::-1])
# #     # output_indices = torch.stack([torch.arange(a-lookback,a) for a in range(max_len, lookback-1+lookforward, -stride)][::-1])
# #     output_indices = torch.stack([torch.stack([torch.arange(b-simultaneous_lookforward+1,b+1) for b in torch.arange(a-lookback,a)]) for a in range(max_len, max(lookback-1+lookforward+simultaneous_lookforward-1,max_len-out_seq_len), -stride)][::-1])
    
# #     # Get non-sequential keys in the data dictionary
# #     non_sequential_keys = [key for key in self.data.keys() if key not in sequential_keys]

# #     # Process each sequential key
# #     for key in sequential_keys:
# #         # Create input and output sequences based on calculated indices
# #         self.data[f"in_{key}"] = self.data[key][:,input_indices]
# #         self.data[f"out_{key}"] = self.data[key][:,output_indices]

# #         # Remove output values where input is padding
# #         input_is_padding = torch.isclose(self.data[f"in_{key}"], padding_value*torch.ones_like(self.data[f"in_{key}"]))
# #         self.data[f"out_{key}"][input_is_padding] = padding_value

# #         # Remove rows where all input or all output is padding
# #         to_keep = torch.logical_and(
# #             torch.logical_not(input_is_padding.all(-1)),
# #             torch.logical_not(torch.isclose(self.data[f"out_{key}"], padding_value*torch.ones_like(self.data[f"out_{key}"])).all(-1).all(-1)))

# #         self.data[f"in_{key}"] = self.data[f"in_{key}"][to_keep]
# #         self.data[f"out_{key}"] = self.data[f"out_{key}"][to_keep]

# #         # Remove output values if index is before out_seq_len from the end
# #         # Option 1: keep same shape
# #         # self.data[f"out_{key}"][:, :-out_seq_len] = padding_value
# #         # Option 2: shorten array
# #         self.data[f"out_{key}"] = self.data[f"out_{key}"][:, max(-keep_last,-out_seq_len+self.data[f"out_{key}"].shape[-1]-1):]
# #         # Shorten by number of samples reserved to this split, also removing simultaneous_lookforward

# #         # Optional: Squeeze out the last dimension if simultaneous_lookforward is 1
# #         # if simultaneous_lookforward == 1:
# #         #     self.data[f"out_{key}"] = self.data[f"out_{key}"].squeeze(-1)

# #         # Optionally, drop the original key from the data dictionary
# #         if drop_original:
# #             del self.data[key]

# #     # Repeat the indices of non-dropped rows for non-sequential keys
# #     orig_rows_repeat = torch.where(to_keep)[0]

# #     # Process each non-sequential key
# #     for key in non_sequential_keys:
# #         self.data[key] = self.data[key][orig_rows_repeat]

# #TODO: which parent class to use?
# #TODO: move this into easy_torch
# class SequentialCollator:
#     def __init__(self,
#                  sequential_keys,
#                  lookback,
#                  padding_value=0, 
#                  left_pad=True, 
#                  lookforward=1, 
#                  simultaneous_lookforward=1,
#                  out_seq_len=None,
#                  keep_last = None,
#                  drop_original=True):
        
#         self.sequential_keys = sequential_keys
#         self.padding_value = padding_value
#         self.left_pad = left_pad
#         self.lookback = lookback
        
#         self.lookforward = lookforward
#         self.simultaneous_lookforward = simultaneous_lookforward
#         self.out_seq_len = out_seq_len
        
#         self.keep_last = keep_last
#         if keep_last is None:
#             self.keep_last = lookback

#         self.drop_original = drop_original

#         if self.left_pad:
#             self.pad_x_function = self.reverse
#             self.pad_out_func = self.flip
#         else:
#             self.pad_x_function = self.identity
#             self.pad_out_func = self.identity

#         self.needed_length = self.lookback + self.lookforward + self.simultaneous_lookforward
        
#     #Functions needed because AttributeError: Can't pickle local object 'SequentialCollator.__init__.<locals>.<lambda>'
#     def identity(self, x):
#         return x
    
#     def reverse(self, x):
#         return x[::-1]
    
#     def flip(self, x):
#         return x.flip(dims=[1])
    
#     def extra_pad(self, x):
#         if self.needed_length <= x.shape[1]:
#             return x
#         else:
#             return torch.cat([x, torch.zeros((x.shape[0], self.needed_length - x.shape[1]),dtype=x.dtype)],dim=1)

#     def __call__(self, batch):
#         out = {}

#         seq_lens = torch.tensor([len(x[self.sequential_keys[0]]) for x in batch])

#         # Pad the sequences in the data using specified parameters
#         for key in batch[0].keys():
#             if key in self.sequential_keys:
#                 out[key] = self.pad_list_of_tensors([x[key] for x in batch])
#             else:
#                 out[key] = torch.stack([torch.tensor(x[key]) for x in batch])
        
#         # Pair input and output sequences based on specified parameters
#         out = self.pair_input_output(out, seq_lens)

#         return out
    
#     # Method to pad a list of tensors and return the padded sequence as a tensor
#     def pad_list_of_tensors(self, list_of_tensors):
#         padded = torch.nn.utils.rnn.pad_sequence([torch.tensor(self.pad_x_function(x)) for x in list_of_tensors], batch_first=True, padding_value=self.padding_value)
        
#         padded = self.extra_pad(padded)

#         #Also add padding for simultaneous_lookforward
#         sim_lookf_pad = self.padding_value*torch.ones((len(padded),self.simultaneous_lookforward-1))
#         lookf_pad = self.padding_value*torch.ones((len(padded),self.lookforward))
#         padded = torch.concat([sim_lookf_pad,padded,lookf_pad],dim=1)

#         padded = self.pad_out_func(padded)

#         # Change type to type of first non-empy list in list of tensors
#         for x in list_of_tensors:
#             if len(x)>0: break
#         padded = padded.type(getattr(torch,str(type(x[0]).__name__)))

#         return padded
    
#     # Method to pair input and output sequences based on specified parameters
#     # Now based on left_padding; TODO: reverse array if opposite
#     def pair_input_output(self, data, seq_lens):
#         out_seq_len = seq_lens if self.out_seq_len is None else self.out_seq_len*torch.ones_like(seq_lens)

#         # decide current point t;
#         # input goes from t-lookback+1 to t;
#         # output goes from t+lookforward to t+lookforward+simultaneous_lookforward
#         output_poss_end_ids = seq_lens #-self.lookforward+1
#         output_poss_start_ids = torch.maximum(output_poss_end_ids-out_seq_len,torch.zeros_like(seq_lens))

#         input_poss_start_ids = output_poss_start_ids - self.lookforward
#         input_poss_end_ids = output_poss_end_ids - self.lookforward

#         true_starting_point = data[self.sequential_keys[0]].shape[1] - (seq_lens+(self.simultaneous_lookforward-1))
#         input_poss_start_ids += true_starting_point
#         input_poss_end_ids += true_starting_point

#         input_poss_start_ids = torch.minimum(input_poss_start_ids+(self.lookback-1),input_poss_end_ids-1)
#         input_poss_start_ids = torch.maximum(input_poss_start_ids,(self.lookback-1)*torch.ones_like(seq_lens))

#         #.int() floors the number, so max_len can't be selected (good, cause is out of bounds)
#         # Generate random indices for output sequences
#         rand = torch.randint(2**63 - 1, size=(len(seq_lens),))
#         current_index = (rand % (input_poss_end_ids - input_poss_start_ids) + input_poss_start_ids).int()
        
#         if (current_index < 0).any():
#             raise ValueError("Some current index is negative")

#         # subtract lookback
#         input_indices = current_index.unsqueeze(1) - (torch.arange(self.lookback).flip(dims=[0])).unsqueeze(0)

#         # Compute input indices based on output indices
#         output_indices = input_indices + self.lookforward

#         # Add simultaneous_lookforward
#         output_indices = output_indices.unsqueeze(2) + torch.arange(self.simultaneous_lookforward).unsqueeze(0).unsqueeze(0)

#         # Process each sequential key
#         for key in self.sequential_keys:
#             # Create input and output sequences based on calculated indices
#             data[f"in_{key}"] = data[key][torch.arange(data[key].shape[0]).unsqueeze(-1),input_indices]
#             data[f"out_{key}"] = data[key][torch.arange(data[key].shape[0]).unsqueeze(-1).unsqueeze(-1),output_indices]

#             # Remove output values if index is before out_seq_len from the end
#             # Option 1: keep same shape
#             # self.data[f"out_{key}"][:, :-out_seq_len] = padding_value
#             # Option 2: shorten array
#             to_keep = -min(self.keep_last,out_seq_len.max())
#             data[f"out_{key}"] = data[f"out_{key}"][:, to_keep:]
#             # Shorten by number of samples reserved to this split, also removing simultaneous_lookforward

#             # Optional: Squeeze out the last dimension if simultaneous_lookforward is 1
#             # if simultaneous_lookforward == 1:
#             #     self.data[f"out_{key}"] = self.data[f"out_{key}"].squeeze(-1)

#             if self.drop_original:
#                 del data[key]
        
#         return data
    
# class RecommendationSequentialCollator(SequentialCollator):
#     def __init__(self,
#                  num_items,
#                  primary_key="sid",
#                  relevance=None,
#                  normalize_relevance=False,
#                  num_positives = 1,
#                  num_negatives = 1,
#                  mask_value = None,
#                  mask_prob = 0,
#                  negatives_distribution = "uniform",
#                  id_key="uid",
#                  *args,**kwargs): #out_seq_len and padding_value are set by super().__init__
        
#         super().__init__(*args,**kwargs)

#         self.num_items = num_items
#         self.primary_key = primary_key
#         self.in_key = f"in_{primary_key}"
#         self.out_key = f"out_{primary_key}"
#         self.id_key = id_key

#         # Set the number of positives and negatives
#         self.num_positives = num_positives
#         self.num_negatives = num_negatives
        
#         negatives_distribution = 'uniform'

#         if negatives_distribution not in {"uniform",'dynamic'} and not isinstance(negatives_distribution, torch.Tensor): #modifica_g : aggiunto il caso di campionamento dinamico dei negativi  durante il training
#             raise NotImplementedError(f"Unsupported negatives_distribution: {negatives_distribution}")
#         self.negatives_distribution = negatives_distribution
#         if self.negatives_distribution == "uniform":
#             self.sample_from_negative_distribution = self.uniform_negatives
#         elif isinstance(negatives_distribution, torch.Tensor):
#             self.sample_from_negative_distribution = self.distr_negatives
#         elif self.negatives_distribution == 'dynamic': #modifica_g: gestione del caso campionamento dinamico dei negativi durante il training 
#             self.negatives_buffer = {} # initialize a buffer to save the sampled negtives at each epoch
#             #self.register_buffer('negatives_buffer', negatives_buffer)
#             self.sample_from_negative_distribution = self.dynamic_negatives
            
            
#         self.relevance = relevance
#         self.normalize_relevance = normalize_relevance
#         # Set a relevance function based on the specified relevance type and data shape
#         if relevance in {None, "fixed", "linear", "exponential"}:
#             # Generate a relevance tensor based on the specified type and data shape
#             self.relevance_function = self.generate_relevance_from_type
#         elif isinstance(relevance, str):
#             # Use an existing output key as the relevance function
#             self.relevance_function = self.get_relevance_from_data
#         else:
#             # Raise an error for unsupported relevance types
#             raise NotImplementedError(f"Unsupported relevance: {relevance}")

#         if mask_prob > 0 and mask_value is None:
#             mask_value = num_items+1
#         self.mask_value = mask_value
#         self.mask_prob = mask_prob

#     def __call__(self, batch):
#         out = super().__call__(batch)

#         out["relevance"] = self.relevance_function(out) # Add relevance scores to out

#         # Add negative samples to the out
#         timesteps = out[self.out_key].shape[1]
#         negatives = self.sample_negatives([x[self.primary_key] for x in batch], timesteps, []) #, [x[self.id_key] for x in batch])

#         out[self.in_key] = self.mask_input(out[self.in_key])

#         out_is_padding = torch.isclose(out[self.out_key], self.padding_value*torch.ones_like(out[self.out_key]))
#         not_to_use = out_is_padding
#         if self.mask_value is not None:
#             relevant_in = out[self.in_key][:, -out[self.out_key].shape[1]:]
#             input_is_not_masking = torch.logical_not(torch.isclose(relevant_in, self.mask_value*torch.ones_like(relevant_in)))
#             not_to_use = torch.logical_or(not_to_use, input_is_not_masking.unsqueeze(-1))

#         all_not_to_use = not_to_use.all(-1)

#         negatives[all_not_to_use] = self.padding_value # Pad negative if out (i.e. positives) is padding #.all(-1) for out because out can have multiple values, i.e. simultaneous_lookforward
#         # Masked tensor are a prototype in torch, but could solve many issues with padding and not wanting to compute losses or metrics on padding
#         #negatives = torch.masked.masked_tensor(negatives, out_is_padding.unsqueeze(-1))

#         out[self.out_key] = torch.cat([out[self.out_key], negatives], dim=-1) #concatenate negatives to out[out_key]
#         out["relevance"] = torch.cat([out["relevance"], torch.zeros_like(negatives)], dim=-1) #concatenate 0 relevance to relevance

#         out["relevance"][:,:,:not_to_use.shape[-1]][not_to_use] = float("nan") # Nan relevance if out is padding or input is not masked (if applicable)
#         out["relevance"][:,:,not_to_use.shape[-1]:][all_not_to_use] = float("nan") # Nan relevance if all out is padding

#         return out
    
#     def uniform_negatives(self, possible_negatives, n, *args):
#         return possible_negatives[torch.randint(0, len(possible_negatives), (n,))]
    
#     def distr_negatives(self, possible_negatives, n, *args):
#         distr = self.negatives_distribution[possible_negatives]
#         repl = True if len(possible_negatives) < n else True
#         return possible_negatives[torch.multinomial(distr, n, replacement=repl)]
    
#     def update_buffer(self,new_negative_buffer):
#         self.negatives_buffer = new_negative_buffer.copy()

#     def dynamic_negatives(self, possible_negatives, n, i):
#         # If the buffer is empty, sample uniformly
#         if len(self.negatives_buffer) == 0:
#             return self.uniform_negatives(possible_negatives, n)
        
#         # Get the negatives with score higher than the target in the previous epoch
#         new_negatives = torch.tensor(list(self.negatives_buffer[i]))

#         if len(new_negatives) < n:
#             new_negatives = torch.cat([new_negatives, self.uniform_negatives(possible_negatives, n-len(new_negatives))])
        
#         new_negatives = new_negatives[torch.randperm(len(new_negatives))] #shuffle new_negatives
#         #new_negatives = new_negatives.reshape(1, -1, self.num_negatives)

#         return new_negatives

#     # Method to sample negative items for a given set of indices
#     def sample_negatives(self, original_sequences, t=1, id_keys=[]):
#         id_keys = id_keys if len(id_keys)>0 else [None]*len(original_sequences)
#         if self.num_negatives == 0:
#             return torch.zeros(len(original_sequences), t, self.num_negatives, dtype=torch.long)
#         negatives = torch.zeros(len(original_sequences), self.num_negatives*t, dtype=torch.long)
#         for i, (orig_seq,id_key) in enumerate(zip(original_sequences,id_keys)): #TODO: parallelize this
#             # Get possible negative items that are not in the original sequence
#             possible_negatives = torch.tensor(list(set(range(1, self.num_items + 1)).difference(orig_seq))) #-1 is needed because index starts from 1
#             # Randomly sample num_negatives negative items
#             negatives[i] = self.sample_from_negative_distribution(possible_negatives, self.num_negatives*t, id_key)
#             #negatives[i] = possible_negatives[torch.randperm(len(possible_negatives))[:self.num_negatives*t]]
#         negatives = negatives.reshape(len(original_sequences), t, max(self.num_negatives,1))
#         return negatives
    
#     def mask_input(self, input):
#         if self.mask_prob == 0: return input
#         mask = torch.rand(input.shape) < self.mask_prob
#         if self.out_seq_len is not None:
#             mask[:,-self.out_seq_len:] = True #Always mask last items (for val / test purposes)
#             mask[:,:-self.out_seq_len] = False #Never mask first items
#         input_is_padding = torch.isclose(input, self.padding_value*torch.ones_like(input))
#         mask[input_is_padding] = False
#         input[mask] = self.mask_value
#         return input

#     # Function to generate a relevance tensor based on the specified relevance type and shape
#     def generate_relevance_from_type(self, complete_data):
#         relevance_type = self.relevance
#         data = complete_data[self.out_key]
#         shape = data.shape
#         if relevance_type is None or relevance_type == "fixed":
#             # Generate a tensor of ones if the relevance type is None or fixed
#             app = torch.ones(shape[-1])
#         else:
#             # Generate a tensor with values from 1 to 0 with equal spacing
#             app = torch.linspace(0, 1, shape[-1])[::-1]
#             # Adjust the tensor based on the relevance type
#             if relevance_type == "linear":
#                 pass
#             elif relevance_type == "exponential":
#                 app = torch.exp(app)

#         # Normalize the tensor
#         if self.normalize_relevance:
#             app /= torch.sum(app)

#         # Repeat the tensor to match the shape
#         app = app.repeat(*shape[:-1], 1)

#         return app
    
#     def get_relevance_from_data(self, complete_data):
#         return complete_data[self.relevance]


# def prepare_rec_datasets(data,
#                          split_keys={"train": ["sid", "timestamp", "rating", "uid"],
#                                      "val": ["sid", "timestamp", "rating", "uid"],
#                                      "test": ["sid", "timestamp", "rating", "uid"]},
#                          **dataset_params
#                          ):
#     """
#     Prepare recommendation datasets for training and evaluation.

#     Args:
#         data (dict): Input dictionary containing data.
#         split_keys (dict): Dictionary specifying keys for different splits.
#         **dataset_params: Additional parameters for dataset preparation.

#     Returns:
#         dict: Dictionary containing prepared recommendation datasets for each split.
#     """

#     datasets = {}
#     for split_name, data_keys in split_keys.items():
#         split_dataset_params = deepcopy(dataset_params)
#         # Select specific parameters for this split
#         for key, value in split_dataset_params.items():
#             if isinstance(value, dict):
#                 if split_name in value.keys():
#                     split_dataset_params[key] = value[split_name]
        
#         # Get data
#         data_to_use = {}
#         for key in data_keys:
#             if key in data:
#                 data_to_use[key] = data[key]
#             else: #If key is not in data, try to get it using split_name
#                 data_to_use[key] = data[f"{split_name}_{key}"]

#         # Create the DataLoader
#         datasets[split_name] = DictDataset(data_to_use, **split_dataset_params)

#     return datasets

# def prepare_rec_collators(split_keys = ["train", "val", "test"],
#                          collator_class = RecommendationSequentialCollator,
#                          **collator_params):
#     """
#     Prepare recommendation data collators for training and evaluation.

#     Args:
#         data (dict): Input dictionary containing data.
#         split_keys (list): List of split keys for data collators.
#         original_seq_key (str): Key for original sequences in the data.
#         **collator_params: Additional parameters for data collator preparation.

#     Returns:
#         dict: Dictionary containing prepared recommendation data collators for each split.
#     """

#     # Default collator parameters
#     default_collator_params = {}

#     # Combine default and custom collator parameters
#     collator_params = dict(list(default_collator_params.items()) + list(collator_params.items()))
    
#     collators = {}
#     for split_name in split_keys:
#         split_collator_params = deepcopy(collator_params)
#         # Select specific parameters for this split
#         for key, value in split_collator_params.items():
#             if isinstance(value, dict):
#                 if split_name in value.keys():
#                     split_collator_params[key] = value[split_name]
#         split_collator_params = change_num_negative_if_float(split_collator_params)

#         # orig_seq_id = original_seq_id if original_seq_id in data else f"{split_name}_{original_seq_id}"
#         # orig_seq_key = original_seq_key if original_seq_key in data else f"{split_name}_{original_seq_key}"
#         # original_seq = {k:v for k,v in zip(data[orig_seq_id],data[orig_seq_key])}

#         # Create the DataCollator
#         collators[split_name] = collator_class(**split_collator_params)

#     return collators

# def change_num_negative_if_float(collator_params):
#     if "num_negatives" in collator_params:
#         if isinstance(collator_params["num_negatives"], float):
#             collator_params["num_negatives"] = int(collator_params["num_negatives"]*collator_params["num_items"])
#         # elif isinstance(collator_params["num_negatives"], dict):
#         #     for key in collator_params["num_negatives"]:
#         #         if isinstance(collator_params["num_negatives"][key], float):
#         #             collator_params["num_negatives"][key] = int(collator_params["num_negatives"][key]*total_items)
#     return collator_params

# def prepare_rec_data_loaders(datasets,
#                              split_keys = ["train", "val", "test"],
#                              original_seq_key="sid",
#                              **loader_params):
#     """
#     Prepare recommendation data loaders for training and evaluation.

#     Args:
#         datasets (dict): Dictionary containing prepared recommendation datasets.
#         data (dict): Input dictionary containing data.
#         split_keys (list): List of split keys for data loaders.
#         original_seq_key (str): Key for original sequences in the data.
#         **loader_params: Additional parameters for data loader preparation.

#     Returns:
#         dict: Dictionary containing prepared recommendation data loaders for each split.
#     """                         
#     # TODO: dict instead of list
#     # I don't remember what I meant by this comment
    
#     # Default loader parameters
#     default_loader_params = {
#         "num_workers": multiprocessing.cpu_count(),
#         "pin_memory": True,
#         "persistent_workers": True,
#         "drop_last": {"train": False, "val": False, "test": False}, #TODO: check specifics about drop last: losing data?
#         "shuffle": {"train": True, "val": False, "test": False},
#     }
#     # Combine default and custom loader parameters
#     loader_params = dict(list(default_loader_params.items()) + list(loader_params.items()))
    
#     loaders = {}
#     for split_name in split_keys:
#         split_loader_params = deepcopy(loader_params)
#         # Select specific parameters for this split
#         for key, value in split_loader_params.items():
#             if isinstance(value, dict):
#                 if split_name in value.keys():
#                     split_loader_params[key] = value[split_name]
        
#         # Create the DataLoader
#         loaders[split_name] = torch.utils.data.DataLoader(datasets[split_name], **split_loader_params)

#     return loaders

# def create_rec_model(name, seed=42, additional_module=None, **model_params):
#     """
#     Create a recommendation model.

#     Args:
#         name (str): Name of the recommendation model.
#         seed (int): Random seed for weight initialization.
#         **model_params: Additional parameters for model creation.

#     Returns:
#         torch.nn.Module: Instance of the recommendation model.
#     """
#     # Set a random seed for weight initialization
#     #  pl.seed_everything(seed)
#     # Get the model from the model module
#     if hasattr(additional_module, name):
#         model_module = additional_module
#     elif hasattr(model, name):
#         model_module = model
#     else:
#         raise NotImplementedError(f"The model {name} is not found in model submodule or additional module")

#     return getattr(model_module, name)(**model_params)


import torch
import pytorch_lightning as pl
import multiprocessing
from copy import deepcopy
from . import model

import numpy as np

#TODO: pass this function inside easy_torch
# Define a custom PyTorch Dataset class named DictDataset
class DictDataset(torch.utils.data.Dataset):
    """
    Custom PyTorch Dataset class that takes a dictionary as input and returns items based on keys.

    Args:
        data (dict): Input dictionary containing data.

    Returns:
        dict: A dictionary where each key corresponds to a tensor item.
    """
    # Constructor to initialize the dataset with input data
    def __init__(self, data):
        self.data = data

        # Convert each value in the data dictionary to a PyTorch tensor
        # for key, value in self.data.items():
        #     if isinstance(value, torch.Tensor):
        #         self.data[key] = value.clone().detach()
        #     else:
        #         self.data[key] = torch.tensor(value)

    # Method to get an item from the dataset at a given index
    def __getitem__(self, index):
        return {key: value[index] for key, value in self.data.items()}

    # Method to get the length of the dataset
    def __len__(self):
        # Assumes that all values in the data dictionary have the same length
        return len(self.data[list(self.data.keys())[0]])

    def add_row(self, new_row):
        """
        Adds a new row of data to the dataset.

        Args:
            new_row (dict): A dictionary containing the new row to be added.
        """
        if not isinstance(new_row, dict):
            raise ValueError("New row must be a dictionary.")
        
        for key, value in new_row.items():
            if key not in self.data:
                raise KeyError(f"Key '{key}' not found in the dataset.")
            if key == 'sid':
                app = np.array([None])
                app[0] = value
                self.data[key] = np.append(self.data[key], app)
            else:
                self.data[key] = np.append(self.data[key], value)


# class SequentialDataset(torch.utils.data.Dataset):
#     def __init__(self, data):
#         self.data = data
#         super().__init__()

#     def __getitem__(self, index):
#         batch = super().__getitem__(index)
#         for key,value in batch:
#             print(key,value)
#         print(sanaskjnas)

# # Method to pair input and output sequences based on specified parameters
# def pair_input_output(self, sequential_keys, padding_value, lookback, stride, lookforward, simultaneous_lookforward, out_seq_len, keep_last, drop_original=True):
#     key_to_use = sequential_keys[0]
#     max_len = self.data[key_to_use].shape[1]
#     if out_seq_len is None: out_seq_len = max_len

#     # Calculate input and output indices based on lookback, stride, and lookforward
#     # input_indices = torch.stack([torch.arange(a-lookback,a) for a in range(max_len-lookforward, lookback-1, -stride)][::-1])
#     input_indices = torch.stack([torch.arange(a-lookback,a) for a in range(max_len-lookforward-simultaneous_lookforward+1, max(lookback-1, max_len-lookforward-simultaneous_lookforward+1-out_seq_len), -stride)][::-1])
#     # output_indices = torch.stack([torch.arange(a-lookback,a) for a in range(max_len, lookback-1+lookforward, -stride)][::-1])
#     output_indices = torch.stack([torch.stack([torch.arange(b-simultaneous_lookforward+1,b+1) for b in torch.arange(a-lookback,a)]) for a in range(max_len, max(lookback-1+lookforward+simultaneous_lookforward-1,max_len-out_seq_len), -stride)][::-1])
    
#     # Get non-sequential keys in the data dictionary
#     non_sequential_keys = [key for key in self.data.keys() if key not in sequential_keys]

#     # Process each sequential key
#     for key in sequential_keys:
#         # Create input and output sequences based on calculated indices
#         self.data[f"in_{key}"] = self.data[key][:,input_indices]
#         self.data[f"out_{key}"] = self.data[key][:,output_indices]

#         # Remove output values where input is padding
#         input_is_padding = torch.isclose(self.data[f"in_{key}"], padding_value*torch.ones_like(self.data[f"in_{key}"]))
#         self.data[f"out_{key}"][input_is_padding] = padding_value

#         # Remove rows where all input or all output is padding
#         to_keep = torch.logical_and(
#             torch.logical_not(input_is_padding.all(-1)),
#             torch.logical_not(torch.isclose(self.data[f"out_{key}"], padding_value*torch.ones_like(self.data[f"out_{key}"])).all(-1).all(-1)))

#         self.data[f"in_{key}"] = self.data[f"in_{key}"][to_keep]
#         self.data[f"out_{key}"] = self.data[f"out_{key}"][to_keep]

#         # Remove output values if index is before out_seq_len from the end
#         # Option 1: keep same shape
#         # self.data[f"out_{key}"][:, :-out_seq_len] = padding_value
#         # Option 2: shorten array
#         self.data[f"out_{key}"] = self.data[f"out_{key}"][:, max(-keep_last,-out_seq_len+self.data[f"out_{key}"].shape[-1]-1):]
#         # Shorten by number of samples reserved to this split, also removing simultaneous_lookforward

#         # Optional: Squeeze out the last dimension if simultaneous_lookforward is 1
#         # if simultaneous_lookforward == 1:
#         #     self.data[f"out_{key}"] = self.data[f"out_{key}"].squeeze(-1)

#         # Optionally, drop the original key from the data dictionary
#         if drop_original:
#             del self.data[key]

#     # Repeat the indices of non-dropped rows for non-sequential keys
#     orig_rows_repeat = torch.where(to_keep)[0]

#     # Process each non-sequential key
#     for key in non_sequential_keys:
#         self.data[key] = self.data[key][orig_rows_repeat]

#TODO: which parent class to use?
#TODO: move this into easy_torch
class SequentialCollator:
    def __init__(self,
                 sequential_keys,
                 lookback,
                 padding_value=0, 
                 left_pad=True, 
                 lookforward=1, 
                 simultaneous_lookforward=1,
                 out_seq_len=None,
                 keep_last = None,
                 drop_original=True):
        
        self.sequential_keys = sequential_keys
        self.padding_value = padding_value
        self.left_pad = left_pad
        self.lookback = lookback
        
        self.lookforward = lookforward
        self.simultaneous_lookforward = simultaneous_lookforward
        self.out_seq_len = out_seq_len
        
        self.keep_last = keep_last
        if keep_last is None:
            self.keep_last = lookback

        self.drop_original = drop_original

        if self.left_pad:
            self.pad_x_function = self.reverse
            self.pad_out_func = self.flip
        else:
            self.pad_x_function = self.identity
            self.pad_out_func = self.identity

        self.needed_length = self.lookback + self.lookforward + self.simultaneous_lookforward
        
    #Functions needed because AttributeError: Can't pickle local object 'SequentialCollator.__init__.<locals>.<lambda>'
    def identity(self, x):
        return x
    
    def reverse(self, x):
        return x[::-1]
    
    def flip(self, x):
        return x.flip(dims=[1])
    
    def extra_pad(self, x):
        if self.needed_length <= x.shape[1]:
            return x
        else:
            return torch.cat([x, torch.zeros((x.shape[0], self.needed_length - x.shape[1]),dtype=x.dtype)],dim=1)

    def __call__(self, batch):
        out = {}

        seq_lens = torch.tensor([len(x[self.sequential_keys[0]]) for x in batch])

        # Pad the sequences in the data using specified parameters
        for key in batch[0].keys():
            if key in self.sequential_keys:
                out[key] = self.pad_list_of_tensors([x[key] for x in batch])
            else:
                out[key] = torch.stack([torch.tensor(x[key]) for x in batch])
        
        # Pair input and output sequences based on specified parameters
        out = self.pair_input_output(out, seq_lens)

        return out
    
    # Method to pad a list of tensors and return the padded sequence as a tensor
    def pad_list_of_tensors(self, list_of_tensors):
        padded = torch.nn.utils.rnn.pad_sequence([torch.tensor(self.pad_x_function(x)) for x in list_of_tensors], batch_first=True, padding_value=self.padding_value)
        
        padded = self.extra_pad(padded)

        #Also add padding for simultaneous_lookforward
        sim_lookf_pad = self.padding_value*torch.ones((len(padded),self.simultaneous_lookforward-1))
        lookf_pad = self.padding_value*torch.ones((len(padded),self.lookforward))
        padded = torch.concat([sim_lookf_pad,padded,lookf_pad],dim=1)

        padded = self.pad_out_func(padded)

        # Change type to type of first non-empy list in list of tensors
        for x in list_of_tensors:
            if len(x)>0: break
        padded = padded.type(getattr(torch,str(type(x[0]).__name__)))

        return padded
    
    # Method to pair input and output sequences based on specified parameters
    # Now based on left_padding; TODO: reverse array if opposite
    def pair_input_output(self, data, seq_lens):
        out_seq_len = seq_lens if self.out_seq_len is None else self.out_seq_len*torch.ones_like(seq_lens)

        # decide current point t;
        # input goes from t-lookback+1 to t;
        # output goes from t+lookforward to t+lookforward+simultaneous_lookforward
        output_poss_end_ids = seq_lens #-self.lookforward+1
        output_poss_start_ids = torch.maximum(output_poss_end_ids-out_seq_len,torch.zeros_like(seq_lens))

        input_poss_start_ids = output_poss_start_ids - self.lookforward
        input_poss_end_ids = output_poss_end_ids - self.lookforward

        true_starting_point = data[self.sequential_keys[0]].shape[1] - (seq_lens+(self.simultaneous_lookforward-1))
        input_poss_start_ids += true_starting_point
        input_poss_end_ids += true_starting_point

        input_poss_start_ids = torch.minimum(input_poss_start_ids+(self.lookback-1),input_poss_end_ids-1)
        input_poss_start_ids = torch.maximum(input_poss_start_ids,(self.lookback-1)*torch.ones_like(seq_lens))

        #.int() floors the number, so max_len can't be selected (good, cause is out of bounds)
        # Generate random indices for output sequences
        rand = torch.randint(2**63 - 1, size=(len(seq_lens),))
        current_index = (rand % (input_poss_end_ids - input_poss_start_ids) + input_poss_start_ids).int()
        
        if (current_index < 0).any():
            raise ValueError("Some current index is negative")

        # subtract lookback
        input_indices = current_index.unsqueeze(1) - (torch.arange(self.lookback).flip(dims=[0])).unsqueeze(0)

        # Compute input indices based on output indices
        output_indices = input_indices + self.lookforward

        # Add simultaneous_lookforward
        output_indices = output_indices.unsqueeze(2) + torch.arange(self.simultaneous_lookforward).unsqueeze(0).unsqueeze(0)

        # Process each sequential key
        for key in self.sequential_keys:
            # Create input and output sequences based on calculated indices
            data[f"in_{key}"] = data[key][torch.arange(data[key].shape[0]).unsqueeze(-1),input_indices]
            data[f"out_{key}"] = data[key][torch.arange(data[key].shape[0]).unsqueeze(-1).unsqueeze(-1),output_indices]

            # Remove output values if index is before out_seq_len from the end
            # Option 1: keep same shape
            # self.data[f"out_{key}"][:, :-out_seq_len] = padding_value
            # Option 2: shorten array
            to_keep = -min(self.keep_last,out_seq_len.max())
            data[f"out_{key}"] = data[f"out_{key}"][:, to_keep:]
            # Shorten by number of samples reserved to this split, also removing simultaneous_lookforward

            # Optional: Squeeze out the last dimension if simultaneous_lookforward is 1
            # if simultaneous_lookforward == 1:
            #     self.data[f"out_{key}"] = self.data[f"out_{key}"].squeeze(-1)

            if self.drop_original:
                del data[key]
        
        return data
    
class RecommendationSequentialCollator(SequentialCollator):
    def __init__(self,
                 num_items,
                 primary_key="sid",
                 relevance=None,
                 normalize_relevance=False,
                 num_positives = 1,
                 num_negatives = 1,
                 mask_value = None,
                 mask_prob = 0,
                 negatives_distribution = "uniform",
                 *args,**kwargs): #out_seq_len and padding_value are set by super().__init__
        
        super().__init__(*args,**kwargs)

        self.num_items = num_items
        self.primary_key = primary_key
        self.in_key = f"in_{primary_key}"
        self.out_key = f"out_{primary_key}"

        # Set the number of positives and negatives
        self.num_positives = num_positives
        self.num_negatives = num_negatives

        if negatives_distribution not in {"uniform"} and not isinstance(negatives_distribution, torch.Tensor):
            raise NotImplementedError(f"Unsupported negatives_distribution: {negatives_distribution}")
        self.negatives_distribution = negatives_distribution
        if self.negatives_distribution == "uniform":
            self.sample_from_negative_distribution = self.uniform_negatives
        elif isinstance(negatives_distribution, torch.Tensor):
            self.sample_from_negative_distribution = self.distr_negatives

        self.relevance = relevance
        self.normalize_relevance = normalize_relevance
        # Set a relevance function based on the specified relevance type and data shape
        if relevance in {None, "fixed", "linear", "exponential"}:
            # Generate a relevance tensor based on the specified type and data shape
            self.relevance_function = self.generate_relevance_from_type
        elif isinstance(relevance, str):
            # Use an existing output key as the relevance function
            self.relevance_function = self.get_relevance_from_data
        else:
            # Raise an error for unsupported relevance types
            raise NotImplementedError(f"Unsupported relevance: {relevance}")

        if mask_prob > 0 and mask_value is None:
            mask_value = num_items+1
        self.mask_value = mask_value
        self.mask_prob = mask_prob

    def __call__(self, batch):
        out = super().__call__(batch)

        out["relevance"] = self.relevance_function(out) # Add relevance scores to out

        # Add negative samples to the out
        timesteps = out[self.out_key].shape[1]
        negatives = self.sample_negatives([x[self.primary_key] for x in batch], timesteps)

        out[self.in_key] = self.mask_input(out[self.in_key])

        out_is_padding = torch.isclose(out[self.out_key], self.padding_value*torch.ones_like(out[self.out_key]))
        not_to_use = out_is_padding
        if self.mask_value is not None:
            relevant_in = out[self.in_key][:, -out[self.out_key].shape[1]:]
            input_is_not_masking = torch.logical_not(torch.isclose(relevant_in, self.mask_value*torch.ones_like(relevant_in)))
            not_to_use = torch.logical_or(not_to_use, input_is_not_masking.unsqueeze(-1))

        all_not_to_use = not_to_use.all(-1)

        negatives[all_not_to_use] = self.padding_value # Pad negative if out (i.e. positives) is padding #.all(-1) for out because out can have multiple values, i.e. simultaneous_lookforward
        # Masked tensor are a prototype in torch, but could solve many issues with padding and not wanting to compute losses or metrics on padding
        #negatives = torch.masked.masked_tensor(negatives, out_is_padding.unsqueeze(-1))

        out[self.out_key] = torch.cat([out[self.out_key], negatives], dim=-1) #concatenate negatives to out[out_key]
        out["relevance"] = torch.cat([out["relevance"], torch.zeros_like(negatives)], dim=-1) #concatenate 0 relevance to relevance

        out["relevance"][:,:,:not_to_use.shape[-1]][not_to_use] = float("nan") # Nan relevance if out is padding or input is not masked (if applicable)
        out["relevance"][:,:,not_to_use.shape[-1]:][all_not_to_use] = float("nan") # Nan relevance if all out is padding

        return out
    
    def uniform_negatives(self, possible_negatives, n):
        return possible_negatives[torch.randint(0, len(possible_negatives), (n,))]
    
    def distr_negatives(self, possible_negatives, n):
        distr = self.negatives_distribution[possible_negatives]
        repl = True if len(possible_negatives) < n else True
        return possible_negatives[torch.multinomial(distr, n, replacement=repl)]

    # Method to sample negative items for a given set of indices
    def sample_negatives(self, original_sequences, t=1):
        if self.num_negatives == 0:
            return torch.zeros(len(original_sequences), t, self.num_negatives, dtype=torch.long)
        negatives = torch.zeros(len(original_sequences), self.num_negatives*t, dtype=torch.long)
        for i, orig_seq in enumerate(original_sequences): #TODO: parallelize this
            # Get possible negative items that are not in the original sequence
            possible_negatives = torch.tensor(list(set(range(1, self.num_items + 1)).difference(orig_seq))) #-1 is needed because index starts from 1
            # Randomly sample num_negatives negative items
            negatives[i] = self.sample_from_negative_distribution(possible_negatives, self.num_negatives*t)
            #negatives[i] = possible_negatives[torch.randperm(len(possible_negatives))[:self.num_negatives*t]]
        negatives = negatives.reshape(len(original_sequences), t, max(self.num_negatives,1))
        return negatives
    
    def mask_input(self, input):
        if self.mask_prob == 0: return input
        mask = torch.rand(input.shape) < self.mask_prob
        if self.out_seq_len is not None:
            mask[:,-self.out_seq_len:] = True #Always mask last items (for val / test purposes)
            mask[:,:-self.out_seq_len] = False #Never mask first items
        input_is_padding = torch.isclose(input, self.padding_value*torch.ones_like(input))
        mask[input_is_padding] = False
        input[mask] = self.mask_value
        return input

    # Function to generate a relevance tensor based on the specified relevance type and shape
    def generate_relevance_from_type(self, complete_data):
        relevance_type = self.relevance
        data = complete_data[self.out_key]
        shape = data.shape
        if relevance_type is None or relevance_type == "fixed":
            # Generate a tensor of ones if the relevance type is None or fixed
            app = torch.ones(shape[-1])
        else:
            # Generate a tensor with values from 1 to 0 with equal spacing
            app = torch.linspace(0, 1, shape[-1])[::-1]
            # Adjust the tensor based on the relevance type
            if relevance_type == "linear":
                pass
            elif relevance_type == "exponential":
                app = torch.exp(app)

        # Normalize the tensor
        if self.normalize_relevance:
            app /= torch.sum(app)

        # Repeat the tensor to match the shape
        app = app.repeat(*shape[:-1], 1)

        return app
    
    def get_relevance_from_data(self, complete_data):
        return complete_data[self.relevance]


def prepare_rec_datasets(data,
                         split_keys={"train": ["sid", "timestamp", "rating", "uid"],
                                     "val": ["sid", "timestamp", "rating", "uid"],
                                     "test": ["sid", "timestamp", "rating", "uid"]},
                         **dataset_params
                         ):
    """
    Prepare recommendation datasets for training and evaluation.

    Args:
        data (dict): Input dictionary containing data.
        split_keys (dict): Dictionary specifying keys for different splits.
        **dataset_params: Additional parameters for dataset preparation.

    Returns:
        dict: Dictionary containing prepared recommendation datasets for each split.
    """

    datasets = {}
    for split_name, data_keys in split_keys.items():
        split_dataset_params = deepcopy(dataset_params)
        # Select specific parameters for this split
        for key, value in split_dataset_params.items():
            if isinstance(value, dict):
                if split_name in value.keys():
                    split_dataset_params[key] = value[split_name]
        
        # Get data
        data_to_use = {}
        for key in data_keys:
            if key in data:
                data_to_use[key] = data[key]
            else: #If key is not in data, try to get it using split_name
                data_to_use[key] = data[f"{split_name}_{key}"]

        # Create the DataLoader
        datasets[split_name] = DictDataset(data_to_use, **split_dataset_params)

    return datasets

def prepare_rec_collators(data,
                         split_keys = ["train", "val", "test"],
                         original_seq_id="uid",
                         original_seq_key="sid",
                         collator_class = RecommendationSequentialCollator,
                         **collator_params):
    """
    Prepare recommendation data collators for training and evaluation.

    Args:
        data (dict): Input dictionary containing data.
        split_keys (list): List of split keys for data collators.
        original_seq_key (str): Key for original sequences in the data.
        **collator_params: Additional parameters for data collator preparation.

    Returns:
        dict: Dictionary containing prepared recommendation data collators for each split.
    """

    # Default collator parameters
    default_collator_params = {}

    # Combine default and custom collator parameters
    collator_params = dict(list(default_collator_params.items()) + list(collator_params.items()))
    
    collators = {}
    for split_name in split_keys:
        split_collator_params = deepcopy(collator_params)
        # Select specific parameters for this split
        for key, value in split_collator_params.items():
            if isinstance(value, dict):
                if split_name in value.keys():
                    split_collator_params[key] = value[split_name]

        # orig_seq_id = original_seq_id if original_seq_id in data else f"{split_name}_{original_seq_id}"
        # orig_seq_key = original_seq_key if original_seq_key in data else f"{split_name}_{original_seq_key}"
        # original_seq = {k:v for k,v in zip(data[orig_seq_id],data[orig_seq_key])}

        # Create the DataCollator
        collators[split_name] = collator_class(**split_collator_params)

    return collators

def prepare_rec_data_loaders(datasets,
                             split_keys = ["train", "val", "test"],
                             original_seq_key="sid",
                             **loader_params):
    """
    Prepare recommendation data loaders for training and evaluation.

    Args:
        datasets (dict): Dictionary containing prepared recommendation datasets.
        data (dict): Input dictionary containing data.
        split_keys (list): List of split keys for data loaders.
        original_seq_key (str): Key for original sequences in the data.
        **loader_params: Additional parameters for data loader preparation.

    Returns:
        dict: Dictionary containing prepared recommendation data loaders for each split.
    """                         
    # TODO: dict instead of list
    # I don't remember what I meant by this comment
    
    # Default loader parameters
    default_loader_params = {
        "num_workers": multiprocessing.cpu_count(),
        "pin_memory": True,
        "persistent_workers": True,
        "drop_last": {"train": False, "val": False, "test": False}, #TODO: check specifics about drop last: losing data?
        "shuffle": {"train": True, "val": False, "test": False},
    }
    # Combine default and custom loader parameters
    loader_params = dict(list(default_loader_params.items()) + list(loader_params.items()))
    
    loaders = {}
    for split_name in split_keys:
        split_loader_params = deepcopy(loader_params)
        # Select specific parameters for this split
        for key, value in split_loader_params.items():
            if isinstance(value, dict):
                if split_name in value.keys():
                    split_loader_params[key] = value[split_name]
        
        # Create the DataLoader
        loaders[split_name] = torch.utils.data.DataLoader(datasets[split_name], **split_loader_params)

    return loaders

def create_rec_model(name, seed=42, additional_module=None, **model_params):
    """
    Create a recommendation model.

    Args:
        name (str): Name of the recommendation model.
        seed (int): Random seed for weight initialization.
        **model_params: Additional parameters for model creation.

    Returns:
        torch.nn.Module: Instance of the recommendation model.
    """
    # Set a random seed for weight initialization
    pl.seed_everything(seed)
    # Get the model from the model module
    if hasattr(additional_module, name):
        model_module = additional_module
    elif hasattr(model, name):
        model_module = model
    else:
        raise NotImplementedError(f"The model {name} is not found in model submodule or additional module")

    return getattr(model_module, name)(**model_params)
