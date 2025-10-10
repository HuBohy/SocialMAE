
import torch
import torch.nn as nn
import numpy as np
import torchvision

from tqdm import tqdm
from argparse import Namespace
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, confusion_matrix, balanced_accuracy_score

from models.social_mae import SocialMAEFT

import matplotlib.pyplot as plt

DEVICE = "mps"

def validate(
        model: nn.Module, 
        dataloader: DataLoader,
        criterion:nn.Module
        ) -> tuple[*tuple[float], float, nn.Module]:
    """
    Evaluates the model's performance on the validation set provided by the DataLoader.
    
    Parameters:
        model (nn.Module): The neural network model to evaluate.
        val_loader (DataLoader): DataLoader providing the validation dataset.
        criterion (nn.Module): Loss function used to evaluate prediction error.
        args (Namespace): Contains runtime arguments that may influence evaluation, such as device settings.

    Returns:
        tuple[*tuple[float], float, nn.Module]: Returns a tuple of floats representing accuracies, losses and the trained model.
    """
    y_true, y_pred = [], []
    model.eval()
    A_loss = []
    with torch.no_grad():
        for data_v, data_a, target in tqdm(dataloader):
            data_v = data_v.float().to(DEVICE)
            data_a = data_a.float().to(DEVICE)

            target = torch.LongTensor(target).to(DEVICE)
            output = model(data_a, data_v, mode="videoonly")
            loss = criterion(output, target)
            A_loss.append(loss.cpu().detach())

            pred = output.argmax(dim=1)
            print(pred)
            y_true.extend(target.cpu().numpy().tolist())
            y_pred.extend(pred.cpu().numpy().tolist())
    loss = np.mean(A_loss)
    

    acc = [balanced_accuracy_score(y_true, y_pred)]
    print(confusion_matrix(y_true, y_pred, labels=range(6)))

    return acc, loss, model

if __name__ == "__main__":
    device = "mps"
    model = SocialMAEFT(label_dim=6, n_frame=8)

    model = nn.DataParallel(model)
    model.to(device)
    best_model = model
    best_weight = torch.load("../../../Social-MAE/finetune/CREMA-D/videoonly/best_social_model.pth", map_location=device)
    # best_weight = torch.load("../../../Social-MAE/finetune/ndc-me/multimodal/best_social_model.pth", map_location=device)
    miss, unexpected = model.load_state_dict(best_weight, strict=False)

    criterion = nn.CrossEntropyLoss()

    videos = torchvision.io.read_image("../test.jpg")
    videos = torchvision.transforms.Resize((224))(videos)
    videos = torchvision.transforms.CenterCrop(224)(videos)

    videos: torch.Tensor = videos.unsqueeze(0).unsqueeze(-1).unsqueeze(0)
    videos = videos.repeat(1, 1, 1, 1, 1, 8)
    print(videos.shape)


    acc, loss, model = validate(best_model, zip(videos, torch.zeros(1, 1, 1024, 128), torch.ones((1, 1), dtype=torch.long)*3), criterion)
    print('Final Accuracy: {}, Loss: {:.4f}'.format(acc, loss))