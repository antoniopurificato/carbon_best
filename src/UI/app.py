from flask import Flask, request, jsonify
from flask_cors import CORS
import subprocess
import json
import os
import threading
import time
app = Flask(__name__)
CORS(app) 

from src.utils.secondary_utils import load_json_results

# Flag to track if initialization is in progress
initialization_in_progress = False
initialization_complete = False

def find_best_models_per_config(pareto_results, weight_acc, weight_en):
    best_models = {}  
    for dataset_name, dataset_data in pareto_results.items():
        for data_perc, data_info in dataset_data.items():
            predicted_models = data_info.get("predicted", [])
            if not predicted_models:
                continue  
            
            max_score = float('-inf')
            current_best = None
            
            for model in predicted_models:
                score = (weight_acc * model["predicted_ACC"]) - (weight_en * model["predicted_EN"])
                if score > max_score:
                    max_score = score
                    current_best = model
            
            if current_best is not None:
                best_models[(dataset_name, data_perc)] = current_best
    
    return best_models

def check_dataset_type(dataset: str):
    if "tomat" in dataset.lower():
        return "text"
    elif "cif" in dataset.lower():
        return "vision"
    elif "four" in dataset.lower():
        return "rec"
    else:
        raise ValueError("Dataset not recognized.")

def obtain_config(dataset: str, tradeoff: float):
    data_type = check_dataset_type(dataset)
    pareto_results = load_json_results("src/results_csv", "pareto_results_test1")
    res = pareto_results[0]
    best_models = find_best_models_per_config(
        res, weight_acc=1-tradeoff, weight_en=tradeoff
    )
    
    best_models_list = [
        {"dataset": dataset, "data_perc": data_perc, "best_model": model_info}
        for (dataset, data_perc), model_info in best_models.items()
    ]

    for item in best_models_list:
        if item["dataset"] == dataset:
            model_name = item["best_model"]["model_name"]
            discard_percentage = item["best_model"]["data_perc"]
            batch_size = item["best_model"]["bs"]
            learning_rate = item["best_model"]["lr"]
            epochs = item["best_model"]["epoch"]
            accuracy = item["best_model"]["predicted_ACC"]
            emissions = item["best_model"]["predicted_EN"]
            terminal_string = f"python3 train_{data_type}.py --dataset {dataset} --model {model_name} --lr {learning_rate} --discard_percentage {discard_percentage} --bs {batch_size} --epochs {discard_percentage}"
            break
    time.sleep(0.02)
    return (
        model_name,
        discard_percentage,
        batch_size,
        learning_rate,
        epochs,
        accuracy,
        emissions,
        terminal_string,
    )

def initialize_data_in_background():
    global initialization_in_progress, initialization_complete
    initialization_in_progress = True
    
    try:
        # Run each subprocess one by one
        subprocess.call(["python", "-m", "src.predictor.extract_and_save_results"])
        subprocess.call(["python", "-m", "src.MOO.1_extract_results"])
        subprocess.call(["python", "-m", "src.MOO.2b_elaborate_pareto_metrics"])
        subprocess.call(["python", "-m", "src.MOO.3_get_pareto_ranking_metrics"])
        initialization_complete = True
    finally:
        initialization_in_progress = False

@app.route('/status', methods=['GET'])
def get_status():
    """Endpoint to check initialization status"""
    if initialization_complete:
        return jsonify({"status": "ready", "message": "Initialization complete"})
    elif initialization_in_progress:
        return jsonify({"status": "initializing", "message": "System is initializing, please wait..."})
    else:
        return jsonify({"status": "not_started", "message": "System not initialized"})

@app.route('/initialize', methods=['POST'])
def initialize():
    """Endpoint to start initialization if needed"""
    global initialization_in_progress, initialization_complete
    
    if initialization_complete:
        return jsonify({"status": "already_initialized", "message": "System already initialized"})
    
    if initialization_in_progress:
        return jsonify({"status": "in_progress", "message": "Initialization already in progress"})
    
    # Start initialization in a background thread
    threading.Thread(target=initialize_data_in_background).start()
    
    return jsonify({"status": "started", "message": "Initialization started"})

@app.route('/compute', methods=['POST'])
def compute():
    global initialization_complete, initialization_in_progress
    
    data = request.json
    dataset = data.get('dataset')
    tradeoff = data.get('tradeoff')
    
    # Check if initialization is needed
    if not os.path.exists('src/results_csv/pareto_results_test1.json'):
        if not initialization_in_progress and not initialization_complete:
            # Start initialization in background
            threading.Thread(target=initialize_data_in_background).start()
        
        # Return a message that the system is initializing
        return jsonify({
            "result": "System is initializing. Please wait and try again in a moment. This could take a few minutes.",
            "status": "initializing"
        })
    
    # If we made it here, either initialization is complete or the file already exists
    result = obtain_config(dataset=dataset, tradeoff=int(tradeoff)/100)

    (
    model_name,
    discard_percentage,
    batch_size,
    learning_rate,
    epochs,
    accuracy,
    emissions,
    terminal_string,
    ) = result


    if 0.0 < int(tradeoff)/100 < 1.0:

        model_name_me, discard_percentage_me, batch_size_me, learning_rate_me, epochs_me, accuracy_me, emissions_me, _, = obtain_config(dataset=dataset, tradeoff=1.0)
        

        model_name_le, discard_percentage_le, batch_size_le, learning_rate_le, epochs_le, accuracy_le, emissions_le, _, = obtain_config(dataset=dataset, tradeoff=0.0)
        
        energy_savings = emissions_le - emissions
        accuracy_diff = accuracy - accuracy_me
        
        output_str_me = (f"💡 Compared to the most energy-efficient configuration ({model_name_me}):\n"
                        f"   - Energy difference: {emissions - emissions_me:.2f} kWh\n"
                        f"   - Accuracy improvement: {accuracy - accuracy_me:.2f}%\n")
        
        output_str_le = (f"💡 Compared to the least energy-efficient configuration ({model_name_le}):\n"
                        f"   - Energy savings: {emissions_le - emissions:.2f} kWh\n"
                        f"   - Accuracy trade-off: {accuracy_le - accuracy:.2f}%\n")
        
    elif float(tradeoff) == 100.0:
        model_name_me, discard_percentage_me, batch_size_me, learning_rate_me, epochs_me, accuracy_me, emissions_me, _, = obtain_config(
            dataset=dataset, tradeoff=0.0)
        
        output_str_le = (f"🌱 You've selected the most energy-efficient configuration!\n"
                        f"Compared to the highest accuracy configuration ({model_name_me}):\n"
                        f"   - Energy savings: {emissions_me - emissions:.2f} kWh\n"
                        f"   - Accuracy trade-off: {accuracy_me - accuracy:.2f}%\n")

    elif float(tradeoff) == 0.0:
        model_name_le, discard_percentage_le, batch_size_le, learning_rate_le, epochs_le, accuracy_le, emissions_le, _, = obtain_config(dataset=dataset, tradeoff=1.0)
        
        output_str_me = (f"🎯 You've selected the highest accuracy configuration!\n"
                        f"Compared to the most energy-efficient option ({model_name_le}):\n"
                        f"   - Additional energy cost: {emissions - emissions_le:.2f} kWh\n"
                        f"   - Accuracy improvement: {accuracy - accuracy_le:.2f}%\n")


    output_text = (
    f"✔️ Model Selected: {model_name}\n"
    f"📦 Batch Size: {batch_size}\n"
    f"🧠 Learning Rate: {learning_rate}\n"
    f"⏱️ Epochs: {epochs}\n"
    f"📉 Discarded Data (%): {discard_percentage}\n"
    f"🎯 Predicted Accuracy: {accuracy:.2f}\n"
    f"🌱 Predicted Emissions: {emissions:.2f}\n\n"
    f"🔧 Command to run:\n{terminal_string}\n"
    f"{output_str_me if 'output_str_me' in locals() else ''}"
    f"{output_str_le if 'output_str_le' in locals() else ''}"
    )

    return jsonify({
        "result": output_text.strip(),
        "status": "complete"
    })
    
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)