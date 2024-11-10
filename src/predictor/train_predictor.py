from src.predictor.predictor import TransformerModel
from src.predictor.data import ArchitectureDataset

if __name__ == '__main__':
    with open('features/datasets_to_features_dict.pkl', 'rb') as handle:
        datasets_to_features = pickle.load(handle)
    with open('features/models_to_features_dict.pkl', 'rb') as handle:
        models_to_features = pickle.load(handle)
    dataset = ArchitectureDataset(models_to_features, datasets_to_features)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    model = TransformerModel(input_size=1024, hidden_size=512, output_size=2)