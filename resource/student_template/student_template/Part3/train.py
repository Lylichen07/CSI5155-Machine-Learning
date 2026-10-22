from pathlib import Path

import torch
from inference import make_model

def train():
    """
    Train a model on a small subset of the dataset. This function is a placeholder and does not perform actual training.
    You should replace this with your own training code that trains your model on the dataset and saves
    the trained weights to 'weights.pt'.

    This is a useful entry point to keep the training code separate from the inference code.
    You can run this function to train your model and save the weights.
    """

    model = make_model()
    torch.save(model.cpu().state_dict(), Path(__file__).with_name("weights.pt"))

if __name__ == "__main__":
    train()
