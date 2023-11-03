import pytest
import torch
from ML3_MODEL_GPU_Ali_Nasser import BlackjackDataset, BlackjackModel, inference # Assuming you put your code in a module

# Test for the BlackjackDataset
def test_blackjack_dataset():
    data = {
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [5, 4, 3, 2, 1],
        'dealer_card': ['A', 'B', 'A', 'B', 'A'],
        'target': [1, 0, 1, 0, 1]
    }
    df = pd.DataFrame(data)
    dataset = BlackjackDataset(df)

    assert len(dataset) == 5
    sample, target = dataset[0]
    assert torch.is_tensor(sample)
    assert torch.is_tensor(target)

# Test the forward pass of the model
def test_blackjack_model_forward():
    model = BlackjackModel(input_size=4)
    sample = torch.randn((1, 4))
    output = model(sample)

    assert output.shape == (1, 1)

# Test the inference function
def test_inference():
    model = BlackjackModel(input_size=4)
    sample = torch.randn((1, 4))
    result = inference(model, sample)

    assert result in [0, 1]


# Run the tests

num_testing_samples = len(test_blackjack_model_forward)
print(f"Number of testing samples: {num_testing_samples}")

