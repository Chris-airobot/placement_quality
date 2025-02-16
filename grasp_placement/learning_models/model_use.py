import json
import torch
from dataset import MyStabilityDataset
from model_train import StabilityNet  # Same model definition

def load_model(model_path, input_dim, hidden_dim=64):
    """
    Load the saved model weights into a fresh StabilityNet instance.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StabilityNet(input_dim, hidden_dim=hidden_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def predict_single_sample(model, sample):
    """
    Given a single data sample, transform it into a tensor
    and feed through the model to get a prediction.
    """
    device = next(model.parameters()).device


    # Create a small dataset with 1 entry
    mini_dataset = MyStabilityDataset([sample])
    x, _ = mini_dataset[0]  # we don't have a label
    x = x.unsqueeze(0).to(device)  # shape [1, input_dim]

    with torch.no_grad():
        pred = model(x)  # shape [1, 1]
    return pred.item()


def read_inference_file(file_path):
    """
    Read a JSON file with a single data sample.
    """
    with open(file_path, "r") as f:
        data = json.load(f)
    return data


if __name__ == "__main__":
    # Suppose you know your input_dim from training
    input_dim = 21  # e.g., if your dataset had 10 features
    model_path = "/home/chris/Downloads/python/stability_model_back.pth"

    # Load the model
    model = load_model(model_path, input_dim=input_dim, hidden_dim=64)
    print("Model loaded successfully.")



    new_data_sample = read_inference_file("/home/chris/Downloads/python/inference_sample.json")

    # Do inference
    pred_score = predict_single_sample(model, new_data_sample)
    print(f"Predicted score for new sample: {pred_score:.4f}")

    



