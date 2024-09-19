# Envrinomental Impact & Best Model Selection

To install the required libraries:

``pip3 install -r requirements.txt``

To create the *vision* knowledge base use the following command:
``python3 train_vision.py --dataset [DATASETS_NAMES] --epochs [NUM_EPOCHS] --gpu_id ID_OF_THE_GPU --seed SEED``

To create the *textual* knowledge base use the following command:

``python3 train_text.py``

The [ITEM] notation is used when you can insert more than 1 value for the corresponding ITEM.