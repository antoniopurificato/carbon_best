
import textstat
from collections import Counter
from typing import Optional
from torch.utils.data import DataLoader

# Some utils functions

def get_classes_distributions(dataset):
    classes = [label for _, label in dataset]
    counter = Counter(classes)

    return list({k: v / len(classes) for k, v in sorted(counter.items())}.values())

def calculate_mean_std_vectorized(dataset):
    loader = DataLoader(dataset, batch_size=len(dataset), num_workers=8)
    images, _ = next(iter(loader))  # Load all images at once
    images = images.view(images.size(0), images.size(1), -1)  # Flatten image
    mean = images.mean(dim=(0, 2))
    std = images.std(dim=(0, 2))
    return mean.numpy().tolist(), std.numpy().tolist()

def calculate_mean_std(dataset, batch_size=64, num_workers=2):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    mean = 0.0
    std = 0.0
    total_samples = 0

    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)  # Flatten image
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_samples += batch_samples

    mean /= total_samples
    std /= total_samples

    return mean.numpy().tolist(), std.numpy().tolist()

def calculate_readability(text):
    readability_scores = {
        "flesch_kincaid_grade": textstat.flesch_kincaid_grade(text),
        "dale_chall_readability_score": textstat.dale_chall_readability_score(text),
    }
    return readability_scores


# Extract features from different datasets
def extract_features_from_easyrec_dataset(
            train_dataset, test_dataset, dataset_name, dict_of_features
        ):
    """
    Extract features from easyrec dataset
    :param train_dataset: EasyRec dataset
    :param test_dataset: EasyRec dataset
    :param dataset_name: Name of the dataset
    :param dict_of_features: Dictionary of features
    :return: Dictionary of features
    """
    try:
        keys = None
        try:
            keys = list(train_dataset.keys())
        except:
            pass

        if keys is not None:
            train_dataset = train_dataset[keys[0]]

        # mean length of training
        len_length_list = []
        num_items_list = set()
        for sequence_length in train_dataset:
            len_length_list.append(len(sequence_length['sid']))
            for item in sequence_length['sid']:
                num_items_list.add(item)

        mean_length = np.mean(len_length_list) + 2 # considering also test set
        num_items = len(num_items_list)
        num_users = len(train_dataset)
        median_length = np.median(len_length_list) + 2 # considering also test set
        density = sum(len_length_list) / (len(len_length_list) * len(num_items_list)) * 100

        # save to dictionary 
        dict_of_features['num_users'][dataset_name] = num_users
        dict_of_features['num_items'][dataset_name] = num_items
        dict_of_features['mean_length'][dataset_name] = mean_length
        dict_of_features['median_length'][dataset_name] = median_length
        dict_of_features['density'][dataset_name] = density
        dict_of_features['num_interactions'][dataset_name] = sum(len_length_list)

    except Exception as e:
        raise ValueError(f"Error extracting features from EasyRec dataset {dataset_name}: {e}. Please contact support to add compatibility.")

    return dict_of_features

def extract_features_from_torch_dataset(train_dataset, test_dataset, dataset_name, dict_of_features):
    try:
        num_train_examples = len(train_dataset)
        num_test_examples = len(test_dataset)
        num_classes = len(train_dataset.classes)
        image_sample, _ = train_dataset[0]
        image_shape = list(image_sample.shape)
        mean, std = calculate_mean_std_vectorized(train_dataset)
        class_distribution = get_classes_distributions(train_dataset)

        dict_of_features['num_train_examples'][dataset_name] = num_train_examples
        dict_of_features['num_test_examples'][dataset_name] = num_test_examples
        dict_of_features['num_classes'][dataset_name] = num_classes
        dict_of_features['class_distribution'][dataset_name] = class_distribution
        dict_of_features['mean'][dataset_name] = mean
        dict_of_features['std'][dataset_name] = std
        dict_of_features['image_shape'][dataset_name] = image_shape

    except Exception as e:
        raise ValueError(f"Error extracting features from dataset {dataset_name}: {e}")

    return dict_of_features





def extract_features_from_huggingface_dataset(train_dataset, test_dataset, dataset_name, dict_of_features, label_column: Optional[str] = None):
    features = list(train_dataset.features)
    try: 
        # Have to do try except here because the dataset might not have a label column, and we need to check every feature singolarly
       for feature in features:
            # Find the output of the dataset 
            if ('ans' not in feature or 'lab' not in feature) and label_column is None:
               continue
            else:
                # Count how many different labels are in the dataset
                output_label_name = 'answer' if 'ans' in features else 'label' if 'lab' in features else label_column
                num_classes = len(set(train_dataset[output_label_name]))
                # get number of samples for each class 
                class_distribution_temp = {}
                for i in range(num_classes):
                    class_distribution_temp[i] = sum([1 for label in train_dataset[output_label_name] if label == i])


            # Obtain input column
            if 'tex' in feature or 'ques' in feature:
                input_label_name = 'text' if 'tex' in feature else 'question' if 'ques' in feature else None
                if input_label_name is None:
                    raise ValueError(f"Input label name not found in dataset {dataset_name}. Please contact support to add compatibility.")

                num_train_examples = train_dataset.num_rows
                num_test_examples = test_dataset.num_rows
                max_length = 0
                tot_sequence_length = 0
                flesch_kincaid_grade = 0
                dale_chall_readability_score = 0
                # Compute some statistics
                for i in range(num_train_examples):
                    current_length = len(train_dataset[input_label_name][i])
                    tot_sequence_length += current_length
                    if current_length > max_length:
                        max_length = current_length
                    readability_score = calculate_readability(train_dataset[input_label_name][i])
                    flesch_kincaid_grade += readability_score['flesch_kincaid_grade']
                    dale_chall_readability_score += readability_score['dale_chall_readability_score']
                
                # Compute the average length of the text
                mean_length = tot_sequence_length / num_train_examples
                mean_flesch_kincaid_grade = flesch_kincaid_grade / num_train_examples
                mean_dale_chall_readability_score = dale_chall_readability_score / num_train_examples
                # Compute the density of the text
                class_distribution = [i / num_train_examples for i in class_distribution_temp.values()]

    except Exception as e:
        raise ValueError(f"Error extracting features from dataset {dataset_name}: {e}. Please add the label_column or contact support to add compatibility.")
    
    # Store the features in the dictionary
    dict_of_features['num_train_examples'][dataset_name] = num_train_examples
    dict_of_features['num_test_examples'][dataset_name] = num_test_examples
    dict_of_features['num_classes'][dataset_name] = num_classes
    dict_of_features['class_distribution'][dataset_name] = class_distribution
    dict_of_features['mean_length'][dataset_name] = mean_length
    dict_of_features['mean_flesch_kincaid_grade'][dataset_name] = mean_flesch_kincaid_grade
    dict_of_features['mean_dale_chall_readability_score'][dataset_name] = mean_dale_chall_readability_score
    dict_of_features['max_length'][dataset_name] = max_length


    return dict_of_features
