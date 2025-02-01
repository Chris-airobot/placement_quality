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

if __name__ == "__main__":
    # Suppose you know your input_dim from training
    input_dim = 22  # e.g., if your dataset had 10 features
    model_path = "/home/chris/Downloads/python/stability_model_back.pth"

    # Load the model
    model = load_model(model_path, input_dim=input_dim, hidden_dim=64)
    print("Model loaded successfully.")


    new_data_sample = {
        "inputs": {
            "Grasp_position": [0.1, 0.2, 0.3],
            "Grasp_orientation": [0.0, 0.0, 0.0, 1.0],
            "cube_initial_position": [0.0, 0.0, 0.0],
            "cube_initial_orientation": [1.0, 0.0, 0.0, 0.0],
            "cube_target_position": [0.1, -0.1, 0.05],
            "cube_target_orientation": [0.5, 0.5, 0.5, 0.5],
            "cube_target_surface": 1
        },
        "outputs": {
            # if your dataset uses compute_label inside __getitem__, 
            # we might not need actual label. But to keep shape consistent:
            "grasp_unsuccessful": False,
            "pose_shift_position": 0.02,
            "pose_shift_orientation": 0.1,
            "position_differece":0.01,
            "orientation_differece":2.1,
            "position_successful":False,
            "orientation_successful":True
        }
    }

    # Do inference
    pred_score = predict_single_sample(model, new_data_sample)
    print(f"Predicted score for new sample: {pred_score:.4f}")
