import os
import json
import argparse
import numpy as np
from tqdm import tqdm
from argparse import Namespace

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataloader import MyDataset
from models.social_mae import SocialMAEFT
from sklearn.metrics import mean_absolute_error, confusion_matrix, balanced_accuracy_score

def load_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--gpus', default=None, type=int, help="Index of the GPUs to use. If None, training runs on CPU.")
    parser.add_argument('-b', '--batch-size', default=8, type=int, metavar='N', help="Batch size.")
    parser.add_argument('-w', '--num-workers', default=8, type=int, metavar='NW', help='Number of subprocesses to use for data loading. More workers can increase the loading speed (default: 8).')
    parser.add_argument("--epochs", type=int, default=20, help="Total number of training epochs to perform.")

    parser.add_argument("--data-video", type=str, help="File path to the directory containing video data.")
    parser.add_argument("--data-audio", type=str, help="File path to the directory containing audio data.")
    parser.add_argument("--dataset", type=str, help="Name of the dataset used for training or testing.")
    parser.add_argument("--n-class", type=int, help="number of classes")
    parser.add_argument("--task", type=str, default='classification', choices=["classification", "regression"], help="Objective task. Choices: classification, regression")
    parser.add_argument("--duration", default=4, type=int, help="Target duration of video clips in seconds.")
    parser.add_argument("--fps", default=2, type=int, help="Frames per second to sample in video clips.")

    parser.add_argument("--mode", type=str, default="multimodal", help="input data type", choices=["mutlimodal", "audioonly", "videoonly"])    
    
    parser.add_argument("--lr_mlp", type=int, default=1e-5, help='Initial classification head learning rate.')
    parser.add_argument('--lr_backbone', '--learning-rate', default=0.001, type=float, metavar='LR', help='Initial backbone learning rate.')
    parser.add_argument("--lr_patience", type=int, default=2, help="Number of epochs to wait before reducing the learning rate if there is no improvement in model performance.")
    parser.add_argument("--lrscheduler_start", default=10, type=int, help="Epoch number to start applying learning rate decay during fine-tuning.")
    parser.add_argument("--lrscheduler_step", default=5, type=int, help="Interval in epochs between learning rate decays during fine-tuning.")
    parser.add_argument("--lrscheduler_decay", default=0.5, type=float, help="Factor by which the learning rate is reduced at each decay step.")

    parser.add_argument("--pretrained-checkpoint", type=str, default=None, help="File path to a pretrained model checkpoint. If None, initializes a new model.")
    parser.add_argument("--saving-folder", type=str, default="./logs", help="Directory where trained models and other parameters will be saved.")
    parser.add_argument('--freeze_base', default=False, help='freeze the backbone or not', action='store_true')
    parser.add_argument("--no-training", default=False, action='store_true', help="if inference model")
    
    parser.add_argument("--contrast_loss_weight", type=float, default=0.01, help="Weighting factor for the contrastive loss component of the total loss.")
    parser.add_argument("--mae_loss_weight", type=float, default=1.0, help="Weighting factor for the mean absolute error loss component of the total loss.")
    parser.add_argument("--masking_ratio", type=float, default=0.75, help="Proportion of input data to mask or drop out during training.")
    parser.add_argument("--n-print-steps", type=int, default=1000, help="Frequency of printing training progress updates to the console.")
    parser.add_argument('-v', '--verbose', action='store_true', default=False, help="Enable verbose output for more detailed information during execution.")
    
    args = parser.parse_args()

    return args

def train_one_epoch(
        model: nn.Module, 
        dataloader: DataLoader, 
        epoch: int, 
        optimizer: torch.optim.Optimizer, 
        criterion:nn.Module, 
        args: Namespace
    ) -> None:
    """
    Executes a single training epoch for the given model using the specified data loader, optimizer, and loss function.

    Args:
        model (nn.Module): The neural network model to train.
        dataloader (DataLoader): DataLoader providing batched training data.
        epoch (int): Current epoch number, used for logging.
        optimizer (torch.optim.Optimizer): Optimizer used for parameter updates.
        criterion (nn.Module): Loss function used to evaluate prediction error.
        args (Namespace): Contains runtime arguments such as device settings.

    Returns:
        None
    """
    model.train()
    print_interval = args.n_print_steps
    for batch_idx, (data_v, data_a, target) in enumerate(tqdm(dataloader)):
        data_v = data_v.float().to(args.device)
        data_a = data_a.float().to(args.device)
        if args.task == 'classification':
            target = torch.LongTensor(target).to(args.device)            
        else:
            target = target.float().to(args.device)
        output = model(data_a, data_v, mode=args.mode)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_idx % print_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data_v), len(dataloader.dataset),
                    100. * batch_idx / len(dataloader), loss.item()))
            
def validate(
        model: nn.Module, 
        dataloader: DataLoader,
        criterion:nn.Module, 
        args: Namespace
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
            data_v = data_v.float().to(args.device)
            data_a = data_a.float().to(args.device)
            if args.task == 'classification':
                target = torch.LongTensor(target).to(args.device)            
            elif args.task == 'regression':
                target = target.float().to(args.device)
            output = model(data_a, data_v, mode=args.mode)
            loss = criterion(output, target)
            A_loss.append(loss.cpu().detach())

            if args.task == 'classification':
                pred = output.argmax(dim=1)
            elif args.task == 'regression':
                pred = output
            y_true.extend(target.cpu().numpy().tolist())
            y_pred.extend(pred.cpu().numpy().tolist())
    loss = np.mean(A_loss)
    
    if args.task == 'classification':
        acc = [balanced_accuracy_score(y_true, y_pred)]
        if args.verbose:
            print(confusion_matrix(y_true, y_pred, labels=range(args.n_class)))
    elif args.task == 'regression':
        acc = [1-mean_absolute_error(np.array(y_true)[:,i], np.array(y_pred)[:,i]) for i in range(args.n_class)]
        acc.append(1-mean_absolute_error(y_true, y_pred))

    return acc, loss, model

if __name__ == '__main__':
    args = load_args()
    if torch.cuda.is_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    setattr(args, "device", device)
    if args.gpus:
        assert torch.cuda.is_available(), f"--gpus is not supported with {args.device}."
    if args.verbose:
        print('Running on ' + str(device))
    
    exp_dir = args.saving_folder
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    if args.verbose:
        print(f"Saving hyperparams and checkpoints at {exp_dir}")

    criterion = torch.nn.CrossEntropyLoss()
    result = []

    if not args.no_training:
        with open(f'{exp_dir}/hparams.json', 'wt') as f:
            json.dump(vars(args), f, indent=4)

    sets = [MyDataset(args.data_video,  args.data_audio, partition=partition) for partition in ['train', 'val', 'test']]
    loaders = [DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers) for ds in sets]
    model = SocialMAEFT(label_dim=args.n_class, n_frame=int(args.fps * args.duration))

    model = nn.DataParallel(model)
    if args.pretrained_checkpoint:
        mdl_weights = torch.load(args.pretrained_checkpoint, map_location=args.device)
        miss, unexpected = model.load_state_dict(mdl_weights, strict=False)
        if args.verbose:
            print(f"Missed parameters: {miss}\nUnexpected parameters: {unexpected}")
            print(f"Model succesfully loaded on {args.device} from {args.pretrained_checkpoint}.")

    model.to(args.device)
    best_model = model
    best_loss = torch.inf

    if not args.no_training:
        mlp_list = ['mlp_head.0.weight', 'mlp_head.0.bias', 'mlp_head.1.weight', 'mlp_head.1.bias',
                    'mlp_head2.0.weight', 'mlp_head2.0.bias', 'mlp_head2.1.weight', 'mlp_head2.1.bias',
                    'mlp_head_a.0.weight', 'mlp_head_a.0.bias', 'mlp_head_a.1.weight', 'mlp_head_a.1.bias',
                    'mlp_head_v.0.weight', 'mlp_head_v.0.bias', 'mlp_head_v.1.weight', 'mlp_head_v.1.bias',
                    'mlp_head_concat.0.weight', 'mlp_head_concat.0.bias', 'mlp_head_concat.1.weight', 'mlp_head_concat.1.bias']
        mlp_params = list(filter(lambda kv: kv[0] in mlp_list, model.named_parameters()))
        base_params = list(filter(lambda kv: kv[0] not in mlp_list, model.named_parameters()))
        mlp_params = [i[1] for i in mlp_params]
        base_params = [i[1] for i in base_params]

        if args.freeze_base == True:
            print('Pretrained backbone parameters are frozen.')
            for param in base_params:
                param.requires_grad = False

        trainables = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam([{'params': base_params, 'lr': args.lr_backbone}, {'params': mlp_params, 'lr': args.lr_mlp}], weight_decay=5e-7, betas=(0.95, 0.999))
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, list(range(args.lrscheduler_start, 1000, args.lrscheduler_step)),gamma=args.lrscheduler_decay)
        if args.verbose:
            print('Total parameter number is : {:.3f} million'.format(sum(p.numel() for p in model.parameters()) / 1e6))
            print('Total trainable parameter number is : {:.3f} million'.format(sum(p.numel() for p in trainables) / 1e6))
            print('The learning rate scheduler starts at {:d} epoch with decay rate of {:.3f} every {:d} epoches'.format(args.lrscheduler_start, args.lrscheduler_decay, args.lrscheduler_step))
            print('Now training with {:s}, learning rate scheduler: {:s}'.format(args.dataset, str(scheduler)))

        print(f"Starting {args.task} training.")
        for epoch in range(args.epochs):
            train_one_epoch(model, loaders[0], epoch, optimizer, criterion, args)
            acc, loss, model = validate(model, loaders[1], criterion, args)
            result.append([*acc, loss, scheduler.optimizer.param_groups[-1]['lr']])
            np.savetxt(exp_dir + '/result.csv', result, delimiter=',')
            print('Validation finished.')
            if args.verbose:
                print(f"Saving current results in {exp_dir}/result.csv")
            print('Accuracy: {}, Loss: {:.4f}, Backbone LR: {}, MLP LR: {}'.format(acc, loss, scheduler.optimizer.param_groups[0]['lr'], scheduler.optimizer.param_groups[-1]['lr']))
            if loss < best_loss:
                best_model = model
                torch.save(best_model.state_dict(), "%s/best_social_model.pth" % (exp_dir))
                torch.save(optimizer.state_dict(), "%s/best_optim_state.pth" % (exp_dir))
                if args.verbose:
                    print(f"Saving checkpoint and optimizer in {exp_dir}")
            scheduler.step()

    best_weight = torch.load("%s/best_social_model.pth" % (exp_dir), map_location=args.device)
    miss, unexpected = best_model.load_state_dict(best_weight, strict=False)
    if args.verbose:
        print(f"Missed parameters: {miss}\nUnexpected parameters: {unexpected}")
        print(f"Model succesfully loaded on {args.device} from {exp_dir}.")
    acc, loss, model = validate(best_model, loaders[2], criterion, args)
    print('Final Accuracy: {}, Loss: {:.4f}'.format(acc, loss))