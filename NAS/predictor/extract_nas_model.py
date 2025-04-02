import os
import re
import json

def get_infos(file_path: str, is_string:bool=False):
    if is_string:
        return file_path
    else:
        try:
            with open(file_path, 'r') as file:
                content = file.read()
                return content
        except Exception as e:
            raise RuntimeError(f"Error reading the file '{file_path}': {e}")


'''def extract_metric(input_item:dict, metric_to_extract:str="validation_accuracy"):
    validation_accuracy = []
    for _, value in input_item.items():
        validation_accuracy.append(value[metric_to_extract])
    return validation_accuracy'''

def extract_metric_bench201(input_item: dict, metric_to_extract: str, total_epochs: int):
    epochs_info = input_item.get("epochs_info", {})
    extracted_metrics = {}

    is_list_format = isinstance(epochs_info.get(metric_to_extract), list)
    metric_values = []

    if is_list_format:
        metric_list = epochs_info.get(metric_to_extract, [])
        for epoch in range(1, total_epochs + 1):
            if epoch <= len(metric_list):
                metric_values.append(metric_list[epoch - 1])
            else:
                metric_values.append(-1)
    else:
        for epoch in range(1, total_epochs + 1):
            value = epochs_info.get(str(epoch), {}).get(metric_to_extract, -1)
            metric_values.append(value)

    extracted_metrics[str(total_epochs)] = metric_values
    return extracted_metrics



def extract_metric(input_item: dict, metric_to_extract: str):
    total_epochs = [4, 12, 36, 108]  # total epochs for each experiment
    half_epochs = {4: 2, 12: 6, 36: 18, 108: 54}  # Mapping half epochs
    
    extracted_metrics = {}
    
    for total_epoch in total_epochs:
        half_epoch = half_epochs[total_epoch]
        metric_values = []
        
        for epoch in range(1, total_epoch + 1):
            if epoch == half_epoch:
                metric_values.append(input_item["epochs_info"].get(str(epoch), {}).get(metric_to_extract, -1))
            elif epoch == total_epoch:
                metric_values.append(input_item["epochs_info"].get(str(epoch), {}).get(metric_to_extract, -1))
            else:
                metric_values.append(-1)
        
        extracted_metrics[str(total_epoch)] = metric_values
    
    return extracted_metrics


def extract_flops_from_text(file_path: str, is_string:bool = False):
    """
    Extracts FLOPs (floating-point operations) information from a text file named 'train_flops.txt' 
    located in the given directory path.

    Args:
        file_path (str):
            The directory path containing 'train_flops.txt'.

    Returns:
        float: The extracted FLOPs value. Returns -1 if the information is not found or if
               there's an error converting the found value to float.

    Raises:
        FileNotFoundError: If 'train_flops.txt' does not exist in the given directory.
        RuntimeError: If an error occurs while reading the file.
    """
    if not is_string:
        file_path = os.path.join(file_path, 'train_flops.txt')
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Error: The file '{file_path}' does not exist.")

    content = get_infos(file_path, is_string=is_string)
        
    # Adjusted pattern to be more flexible
    pattern = r'fwd flops of model = fwd flops per GPU \* mp_size:\s*([\d\.]+)[Tt]?'
    match = re.search(pattern, content, re.DOTALL)
    
    if match:
        try:
            extracted_value = float(match.group(1).strip())
            return extracted_value
        except ValueError:
            print(f"Error converting extracted value to float in file '{file_path}'.")
            return -1
    else:
        print(f"FLOPS information not found in file '{file_path}'. Returning -1.")
        return -1

    


def extract_architecture_metrics(file_path: str, is_string:bool=False):
    """
    Extracts depth and parameter counts from 'train_flops.txt' within the given directory.

    The function looks for lines indicating the model's depth (lines containing "depth xx")
    and a parameter count pattern "params per GPU: xxx (K/M/B)".

    Args:
        file_path (str):
            The directory path containing 'train_flops.txt'.

    Returns:
        tuple: (depth, params)
            - depth (int): The total count of recognized depth lines.
            - params (float): The number of parameters in millions. 
                              If a 'B' suffix is found, it is converted to thousands of millions.
                              If a 'K' suffix is found, it is converted to fractions of a million (K=0.001M).
                              Returns -1 if the parameters are not found.

    Raises:
        FileNotFoundError: If 'train_flops.txt' is not present.
    """
    if not is_string:
        file_path = os.path.join(file_path, 'train_flops.txt')
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Error: The file '{file_path}' does not exist.")

    content = get_infos(file_path, is_string=is_string)

    # Extract Total Depth
    depth_matches = re.findall(r"depth (\d+):", content)
    depth= len(depth_matches)

    params_pattern = r'params per GPU:\s*([\d\.]+)\s*([KMB])'
    params_match = re.search(params_pattern, content, re.DOTALL)

    if params_match:
        params_value = float(params_match.group(1).strip())
        unit = params_match.group(2).strip()

        # Convert K to M if necessary
        if unit == "K":
            params = params_value / 1000  # Convert thousands to millions
        elif unit == "M":
            params = params_value
        elif unit == "B":
            params = params_value * 1000

    else:
        params = -1
        print("Parameters per GPU not found in the file. Placeholder -1 inserted")

    return depth, params 


def extract_attention_layers(content):
    """
    Searches for various self-attention modules in a string (content of 'train_flops.txt') and 
    extracts their in_features, inferred num_heads, and LoRA (low-rank adaptation) configuration 
    if present.

    Args:
        content (str):
            The raw text content from a 'train_flops.txt' file.

    Returns:
        list: A list of dictionaries, each describing an attention layer with keys such as:
              {
                  "type": <str>,
                  "in_features": <int>,
                  "num_heads": <int>,
                  "lora_r": <int or -1 if not present>
              }
    """
    # Define patterns for various attention types
    attention_layer_patterns = [
        (
            "MultiheadAttention",
            re.compile(
                r"MultiheadAttention\([\s\S]*?in_features=(\d+)[\s\S]*?out_features=(\d+)",
                re.MULTILINE | re.IGNORECASE
            )
        ),
        (
            "BertSelfAttention",
            re.compile(
                r"(?:BertSelfAttention)[\s\S]*?\(query\): Linear\([\s\S]*?in_features=(\d+), out_features=(\d+)",
                re.MULTILINE | re.IGNORECASE
            )
        ),
        (
            "RobertaSelfAttention",
            re.compile(
                r"(?:RobertaSelfAttention)[\s\S]*?\(query\): Linear\([\s\S]*?in_features=(\d+), out_features=(\d+)",
                re.MULTILINE | re.IGNORECASE
            )
        ),
        (
            "PhiSdpaAttention",
            re.compile(
                 r"PhiSdpaAttention[\s\S]*?\(q_proj\)[\s\S]*?in_features=(\d+).*?out_features=(\d+)",
                re.MULTILINE | re.IGNORECASE
            )
        ), 
        (   "MistralSdpaAttention",
              re.compile(
                 r"MistralSdpaAttention[\s\S]*?\(q_proj\)[\s\S]*?in_features=(\d+).*?out_features=(\d+)",
                re.MULTILINE | re.IGNORECASE
            )
         )
    ]

    # Pattern specifically for lora_A
    lora_a_pattern = re.compile(
        #r"PhiSdpaAttention|MistralSdpaAttention[\s\S]*?\(lora_A\)[\s\S]*?out_features=(\d+)",
        r"lora_A[\s\S]*?out_features=(\d+)",
        re.MULTILINE | re.IGNORECASE)

    attention_layers = []

    for attention_type, pattern in attention_layer_patterns:
        matches = pattern.findall(content)
        if matches:
            for match in matches:
                # Extract in_features and out_features
                in_features, out_features = int(match[0]), int(match[1])

                # Derive num_heads (assuming head_dim = 64)
                head_dim = 64
                num_heads = in_features // head_dim
                if num_heads<=0:
                    num_heads=1


                # Extract the r value from lora_A
                lora_a_match = lora_a_pattern.search(content)
                
                r_value = int(lora_a_match.group(1)) if lora_a_match else -1

                # Add to the flattened list
                attention_layers.append({
                    "type": attention_type,
                    "in_features": in_features,
                    "num_heads": num_heads,
                    "lora_r": r_value  # Only the r value (out_features of lora_A)
                })

            # Stop further processing once matches are found
            break

    return attention_layers




def extract_dropout_layers(content):
    """
    Extracts all Dropout layers from the provided text content and retrieves their dropout probabilities.

    Args:
        content (str):
            The raw text from 'train_flops.txt' detailing model architecture.

    Returns:
        list: A list of dictionaries, each containing:
              {"p": <float> } for the dropout probability.
    """
    dropout_pattern = re.compile(
        r"Dropout\(.*?p=([\d.]+)", re.MULTILINE | re.IGNORECASE
    )

    matches = dropout_pattern.findall(content)
    #print(f"Matches found for dropout : {matches}")
    dropout_layers = []

    for match in matches:
        dropout_layers.append({
            "p": float(match)
        })

    return dropout_layers



def extract_conv_layers(content):
    """
    Extracts convolutional layers from the content of 'train_flops.txt' by matching 
    lines of the form: Conv2d(in_ch, out_ch, kernel_size=(kx, ky), stride=(sx, sy)).

    Args:
        content (str):
            The string content of 'train_flops.txt'.

    Returns:
        list: A list of dictionaries, each describing a convolutional layer with fields:
              {
                  "output_channels": <int>,
                  "kernel_size": (kx, ky),
                  "stride": (sx, sy)
              }
    """
    # Updated pattern to include stride
    conv_layer_pattern = re.compile(
        r"Conv2d\((\d+), (\d+), kernel_size=\((\d+), (\d+)\)(?:, stride=\((\d+), (\d+)\))?", 
        re.DOTALL
    )

    # Pattern to match BatchNorm2d layers
    batchnorm_pattern = re.compile(
        r"BatchNorm2d\((\d+),", re.DOTALL
    )

    layers = []

    # Extract Conv2d layers
    for match in conv_layer_pattern.finditer(content):
        groups = match.groups()
        in_channels, out_channels, kernel_h, kernel_w = map(int, groups[:4])
        
        # Handle stride - default to (1,1) if not specified
        if len(groups) > 4 and groups[4] and groups[5]:
            stride = (int(groups[4]), int(groups[5]))
        else:
            stride = (1, 1)

        layers.append({
            "in_channels": in_channels,
            "out_channels": out_channels,
            "kernel_size": (kernel_h, kernel_w),
            "stride": stride
        })

    # Extract BatchNorm2d layers
    for match in batchnorm_pattern.finditer(content):
        num_features = int(match.group(1))
        layers.append({
            "num_features": num_features
        })

    return layers

def extract_lora_layers(content):
    """
    Extracts LoRA (Low-Rank Adaptation) configurations such as lora_A, lora_B, and lora_dropout 
    from the given content string.

    Args:
        content (str):
            The raw text possibly containing references to LoRA modules.

    Returns:
        list: A list of dictionaries describing each LoRA layer, e.g.:
              {
                  "type": <"lora_A"/"lora_B"/"lora_dropout">,
                  "in_features": <int>,
                  "out_features": <int>,
                  "dropout": <float>
              }
    """
    
    # Define patterns to match LoRA configurations (lora_A, lora_B, lora_dropout)
    lora_pattern = re.compile(
        r"(lora_A|lora_B|lora_dropout):.*?Linear\(.*?in_features=(\d+),.*?out_features=(\d+).*?p=([\d.]+)?",
        re.MULTILINE | re.IGNORECASE
    )

    matches = lora_pattern.findall(content)
    lora_layers = []

    for match in matches:
        layer_type, in_features, out_features, dropout = match
        lora_layers.append({
            "type": layer_type,
            "in_features": int(in_features),
            "out_features": int(out_features),
            "dropout": float(dropout) if dropout else 0
        })

    return lora_layers

def convert_to_millions(value):
    """
    Convert a numerical value to the millions (M) scale.
    """
    return value / 1000000 


def parse_model_info(file_path: str, is_string: bool = False):
    
    content = get_infos(file_path, is_string)

    conv_layers_raw = extract_conv_layers(content)
    attention_layers = extract_attention_layers(content)
    dropout_layers = extract_dropout_layers(content)
    fc_layers = re.findall(r"Linear.*?in_features=(\d+), out_features=(\d+)", content)
    activation_functions = re.findall(r"(ReLU|GELU|Sigmoid|Tanh|Softmax|Swish|SeLU)", content)
    batch_norm_layers = re.findall(r"BatchNorm2d\(.*?eps=([\de.-]+), momentum=([\de.-]+)", content)

    model_info = {
        "conv_layers_NEW": [],
        "fc_layers_NEW": [],
        "activation_functions_NEW": [],
        "dropout_NEW": [],
        "attention_layers_NEW": attention_layers,
        "layer_norm_layers_NEW":[],
        "batch_norm_layers_NEW": [],
        "embedding_layers_NEW": []
    }

    # Populate convolution layers and batch norm layers
    for layer in conv_layers_raw:
        if "out_channels" in layer:  # Conv2d layer
            conv_layer_structured = {
                "output_channels": layer["out_channels"],
                "kernel_size": layer["kernel_size"],
                "stride": layer["stride"]
            }
            model_info["conv_layers_NEW"].append(conv_layer_structured)


    for fc in fc_layers:
        model_info["fc_layers_NEW"].append({"in_features": int(fc[0])})

    for act in activation_functions:
        model_info["activation_functions_NEW"].append({"type": act})

    for dropout in dropout_layers:
        model_info["dropout_NEW"].append({"p": dropout["p"]})

    for eps, momentum in batch_norm_layers:
        model_info["batch_norm_layers_NEW"].append({
            "eps": float(eps),
            "momentum": float(momentum)
        })

    return model_info



if __name__ == '__main__':
    with open("first_2_elements.json", "r") as file:
        data = json.load(file)
    model_info_1 = parse_model_info(file_path=data[0]["model"], is_string=True)
    #model_info_2 = parse_layers_info(file_path=data[0]["model"], is_string=True)
    print(model_info_1)
    #print(model_info_2)
    validation_accuracy = extract_metric(input_item = data[0], metric_to_extract="validation_accuracy")
    training_time = extract_metric(input_item = data[0], metric_to_extract="training_time")
    num_params = convert_to_millions(data[0]['num_params'])

    print(validation_accuracy)

