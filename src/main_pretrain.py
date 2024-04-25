import os
import time
import json
import argparse
import numpy as np
from tqdm import tqdm
from argparse import Namespace

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler

from dataloader import MyDataset
from models.social_mae import SocialMAE
from utilities.util import AverageMeter

def load_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--gpus', default=None, type=int, help="Index of the GPUs to use. If None, training runs on CPU.")
    parser.add_argument('-b', '--batch-size', default=8, type=int, metavar='N', help="Batch size.")
    parser.add_argument('-w', '--num-workers', default=8, type=int, metavar='NW', help='Number of subprocesses to use for data loading. More workers can increase the loading speed (default: 8).')
    parser.add_argument("--epochs", type=int, default=20, help="Total number of training epochs to perform.")

    parser.add_argument("--data-video", type=str, help="File path to the directory containing video data.")
    parser.add_argument("--data-audio", type=str, help="File path to the directory containing audio data.")
    parser.add_argument("--dataset", type=str, help="Name of the dataset used for training or testing.")
    parser.add_argument("--pretrained-checkpoint", type=str, default=None, help="File path to a pretrained model checkpoint. If None, initializes a new model.")
    
    parser.add_argument("--saving-folder", type=str, default="./logs", help="Directory where trained models and other parameters will be saved.")

    parser.add_argument("--duration", default=4, type=int, help="Target duration of video clips in seconds.")
    parser.add_argument("--fps", default=2, type=int, help="Frames per second to sample in video clips.")
    
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, metavar='LR', help='Initial learning rate for training.')
    parser.add_argument("--lr_patience", type=int, default=2, help="Number of epochs to wait before reducing the learning rate if there is no improvement in model performance.")
    parser.add_argument("--lrscheduler_start", default=10, type=int, help="Epoch number to start applying learning rate decay during fine-tuning.")
    parser.add_argument("--lrscheduler_step", default=5, type=int, help="Interval in epochs between learning rate decays during fine-tuning.")
    parser.add_argument("--lrscheduler_decay", default=0.5, type=float, help="Factor by which the learning rate is reduced at each decay step.")

    parser.add_argument("--contrast_loss_weight", type=float, default=0.01, help="Weighting factor for the contrastive loss component of the total loss.")
    parser.add_argument("--mae_loss_weight", type=float, default=1.0, help="Weighting factor for the mean absolute error loss component of the total loss.")
    parser.add_argument("--masking_ratio", type=float, default=0.75, help="Proportion of input data to mask or drop out during training.")
    parser.add_argument("--n-print-steps", type=int, default=1000, help="Frequency of printing training progress updates to the console.")
    parser.add_argument('-v', '--verbose', action='store_true', default=False, help="Enable verbose output for more detailed information during execution.")

    args = parser.parse_args()

    return args

def train(
        model: nn.Module,
        train_loader: DataLoader, 
        test_loader: DataLoader, 
        args: Namespace
    ) -> None:
    """
    Trains model using the provided training and validation data loaders.

    This function handles the training loop, including backpropagation and model evaluation on a validation set.
    Training parameters such as the number of epochs, learning rate, and others are taken from the `args` Namespace object.

    Args:
        model (nn.Module): The neural network model to train.
        train_loader (DataLoader):  DataLoader providing batched training data.
        test_loader (DataLoader):  DataLoader providing batched validation data.
        args (Namespace): Contains runtime arguments such as device settings.

    Returns:
        None
    """

    device = args.device
    if args.verbose:
        print('running on ' + str(device))
    torch.set_grad_enabled(True)

    meters = {
        'batch_time': AverageMeter(), 'per_sample_time': AverageMeter(), 'data_time': AverageMeter(), 'per_sample_data_time': AverageMeter(), 'per_sample_dnn_time': AverageMeter(),
        'loss_av_meter': AverageMeter(), 'loss_a_meter': AverageMeter(), 'loss_v_meter': AverageMeter(), 'loss_c_meter': AverageMeter()
    }
    best_epoch, best_loss = 0, np.inf
    global_step, epoch = 0, 0
    exp_dir = args.saving_folder
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    if args.verbose:
        print(f"Saving hyperparams and checkpoints at {exp_dir}")

    if not isinstance(model, nn.DataParallel):
        model = nn.DataParallel(model)
    model = model.to(device)

    trainables = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainables, args.lr, weight_decay=5e-7, betas=(0.95, 0.999))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, list(range(args.lrscheduler_start, 1000, args.lrscheduler_step)),gamma=args.lrscheduler_decay)
    if args.verbose:
        print('Total parameter number is : {:.3f} million'.format(sum(p.numel() for p in model.parameters()) / 1e6))
        print('Total trainable parameter number is : {:.3f} million'.format(sum(p.numel() for p in trainables) / 1e6))
        print('The learning rate scheduler starts at {:d} epoch with decay rate of {:.3f} every {:d} epochs'.format(args.lrscheduler_start, args.lrscheduler_decay, args.lrscheduler_step))

    print('Now training with {:s}, learning rate scheduler: {:s}'.format(args.dataset, str(scheduler)))

    epoch += 1
    scaler = GradScaler()

    with open(f'{exp_dir}/hparams.json', 'wt') as f:
        json.dump(vars(args), f, indent=4)

    result = np.zeros([args.epochs, 10])
    print("Start training.")
    while epoch < args.epochs + 1:
        begin_time = time.time()
        end_time = time.time()
        model.train()
        print('---------------')
        print(f"Current epoch={epoch}")
        if args.verbose:
            print(f'Current masking ratio is {args.masking_ratio:.3f} for both modalities.')

        for i, (v_input, a_input, _) in enumerate(tqdm(train_loader)):
            B = a_input.size(0)
            v_input = v_input.to(device, non_blocking=True)
            a_input = a_input.to(device, non_blocking=True)

            meters['data_time'].update(time.time() - end_time)
            meters['per_sample_data_time'].update((time.time() - end_time) / a_input.shape[0])
            dnn_start_time = time.time()

            with autocast():
                loss, loss_mae, loss_mae_a, loss_mae_v, loss_c, mask_a, mask_v, c_acc = model(a_input, v_input, mask_ratio_a=args.masking_ratio, mask_ratio_v=args.masking_ratio, mae_loss_weight=args.mae_loss_weight)
                loss, loss_mae, loss_mae_a, loss_mae_v, loss_c = loss.sum(), loss_mae.sum(), loss_mae_a.sum(), loss_mae_v.sum(), loss_c.sum()

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # loss_av is the main loss
            meters['loss_av_meter'].update(loss.item(), B)
            meters['loss_a_meter'].update(loss_mae_a.item(), B)
            meters['loss_v_meter'].update(loss_mae_v.item(), B)
            meters['loss_c_meter'].update(loss_c.item(), B)
            meters['batch_time'].update(time.time() - end_time)
            meters['per_sample_time'].update((time.time() - end_time)/a_input.shape[0])
            meters['per_sample_dnn_time'].update((time.time() - dnn_start_time)/a_input.shape[0])

            print_step = global_step % args.n_print_steps == 0
            early_print_step = epoch == 0 and global_step % (args.n_print_steps/10) == 0
            print_step = print_step or early_print_step

            if print_step and global_step > 0:
                print(' '+''.center(90, '-')+'\n|'+f'[Epoch {epoch}][{i}/{len(train_loader)}] - Train'.center(90, ' ')+'|\n'+' '+''.center(90, '-'), flush=True)
                print('|'+'Time Per Sample (s)'.center(20, ' ')+'|'+
                  f"Total: {meters['per_sample_time'].avg:.3f}".center(5, ' ')+'|'+
                  f"Data: {meters['per_sample_data_time'].avg:.3f}".center(20, ' ')+'|'+
                  f"DNN: {meters['per_sample_dnn_time'].avg:.3f}".center(20, ' ')+'|'+
                  f"".center(14, ' ')+'|', flush=True)
                print(
                  '|'+'Train Losses'.center(20, ' ')+'|'+
                  f"Total: {meters['loss_av_meter'].val:.3f}".center(5, ' ')+'|'+
                  f"Audio MAE: {meters['loss_a_meter'].val:.3f}".center(20, ' ')+'|'+
                  f"Visual MAE: {meters['loss_v_meter'].val:.3f}".center(20, ' ')+'|'+
                  f"Contr.: {meters['loss_c_meter'].val:.3f}".center(14, ' ')+'|', flush=True)
                print(' '+''.center(90, '-'), flush=True)
                if np.isnan(meters['loss_av_meter'].avg):
                    print("Training diverged...")
                    return

            end_time = time.time()
            global_step += 1

        print('Start validation.')
        eval_loss_av, eval_loss_mae, eval_loss_mae_a, eval_loss_mae_v, eval_loss_c, eval_c_acc = validate(model, test_loader, args)


        print(' '+''.center(87, '-')+'\n|'+f'[Epoch {epoch}] - Results'.center(87, ' ')+'|\n'+' '+''.center(87, '-'), flush=True)
        print('|'+'Loss Type'.center(20, ' ')+'|'+'Audio MAE'.center(12, ' ')+'|'+'Visual MAE'.center(13, ' ')+'|'+'Total MAE'.center(10, ' ')+'|'+'Contrast.'.center(15, ' ')+'|'+'Total'.center(12, ' ')+'|')
        print(
            '|'+'Train'.center(20, ' ')+'|'+
            f"{meters['loss_a_meter'].val:.3f}".center(12, ' ')+'|'+
            f"{meters['loss_v_meter'].val:.3f}".center(13, ' ')+'|'+
            f"".center(10, ' ')+'|'+
            f"{meters['loss_c_meter'].val:.3f}".center(15, ' ')+'|'+
            f"{meters['loss_av_meter'].val:.3f}".center(12, ' ')+'|', flush=True)
        print(
            '|'+'Validation'.center(20, ' ')+'|'+
            f'{eval_loss_mae_a:.3f}'.center(12, ' ')+'|'+
            f'{eval_loss_mae_v:.3f}'.center(13, ' ')+'|'+
            f'{eval_loss_mae:.3f}'.center(10, ' ')+'|'+
            f'{eval_loss_c:.3f}'.center(15, ' ')+'|'+
            f'{eval_loss_av:.3f}'.center(12, ' ')+'|', flush=True)
        print(' '+''.center(87, '-'), flush=True)

        result[epoch-1, :] = [meters['loss_a_meter'].avg, meters['loss_v_meter'].avg, meters['loss_c_meter'].avg, meters['loss_av_meter'].avg, eval_loss_mae_a, eval_loss_mae_v, eval_loss_c, eval_loss_av, eval_c_acc, optimizer.param_groups[0]['lr']]
        np.savetxt(exp_dir + '/result.csv', result, delimiter=',')
        if args.verbose:
            print(f"Saving current results in {exp_dir}/result.csv")

        if eval_loss_av < best_loss:
            best_loss = eval_loss_av
            best_epoch = epoch

        if best_epoch == epoch:
            torch.save(model.state_dict(), "%s/best_model.pth" % (exp_dir))
            torch.save(optimizer.state_dict(), "%s/best_optim_state.pth" % (exp_dir))
            if args.verbose:
                print(f"Saving checkpoint and optimizer in {exp_dir}")

        scheduler.step()

        finish_time = time.time()
        if args.verbose:
           print(f'[Epoch {epoch}] lr: {optimizer.param_groups[0]['lr']}; training time: {finish_time-begin_time:.3f}')

        epoch += 1

        for average_meter in meters.values():
            average_meter.reset()

def validate(
        model: nn.Module, 
        val_loader: DataLoader, 
        args: Namespace
    ) -> tuple[float, float, float, float, float]:
    """
    Evaluates the model's performance on the validation set provided by the DataLoader.

    Parameters:
        model (nn.Module): The neural network model to evaluate.
        val_loader (DataLoader): DataLoader providing the validation dataset.
        args (Namespace): Contains runtime arguments that may influence evaluation, such as device settings.

    Returns:
        tuple[float, float, float, float, float]: Returns a tuple of floats representing losses (total, reconstruction (AV, A, V) loss, contrastive loss and contrastive accuracy).
    """
    device = args.device
    batch_time = AverageMeter()
    model = model.to(device)
    model.eval()

    end = time.time()
    A_loss, A_loss_mae, A_loss_mae_a, A_loss_mae_v, A_loss_c, A_c_acc = [], [], [], [], [], []
    with torch.no_grad():
        for i, (v_input, a_input, _) in enumerate(tqdm(val_loader)):
            v_input = v_input.to(device)
            a_input = a_input.to(device)
            with autocast():
                loss, loss_mae, loss_mae_a, loss_mae_v, loss_c, mask_a, mask_v, c_acc = model(a_input, v_input, mask_ratio_a=args.masking_ratio, mask_ratio_v=args.masking_ratio)
                loss, loss_mae, loss_mae_a, loss_mae_v, loss_c, c_acc = loss.sum(), loss_mae.sum(), loss_mae_a.sum(), loss_mae_v.sum(), loss_c.sum(), c_acc.mean()
            A_loss.append(loss.to('cpu').detach())
            A_loss_mae.append(loss_mae.to('cpu').detach())
            A_loss_mae_a.append(loss_mae_a.to('cpu').detach())
            A_loss_mae_v.append(loss_mae_v.to('cpu').detach())
            A_loss_c.append(loss_c.to('cpu').detach())
            A_c_acc.append(c_acc.to('cpu').detach())
            batch_time.update(time.time() - end)
            end = time.time()
            if i > 0:
                break

        loss = np.mean(A_loss)
        loss_mae = np.mean(A_loss_mae)
        loss_mae_a = np.mean(A_loss_mae_a)
        loss_mae_v = np.mean(A_loss_mae_v)
        loss_c = np.mean(A_loss_c)
        c_acc = np.mean(A_c_acc)

    return loss, loss_mae, loss_mae_a, loss_mae_v, loss_c, c_acc

if __name__ == "__main__":
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
    
    dev_ds = MyDataset(args.data_video, args.data_audio, 'dev', target_duration=args.duration, target_fps=args.fps)
    train_ds, validation_ds = random_split(dev_ds, [0.7, 0.3])

    train_dl = DataLoader(
        train_ds, args.batch_size, drop_last=True, num_workers=args.num_workers, shuffle=True, pin_memory=True
    )
    validation_dl = DataLoader(
        validation_ds, args.batch_size, drop_last=True, num_workers=args.num_workers, shuffle=True, pin_memory=True
    )

    n_frames = int(args.fps * args.duration)
    model = SocialMAE(n_frame=n_frames, stride=2, tr_pos=True)
    model = nn.DataParallel(model)
    if args.pretrained_checkpoint:
        mdl_weights = torch.load(args.pretrained_checkpoint, map_location=args.device)
    
        del mdl_weights['module.patch_embed_v.proj.weight'], mdl_weights['module.patch_embed_v.proj.bias']
        del mdl_weights['module.pos_embed_v']
        del mdl_weights['module.decoder_pos_embed_v'], mdl_weights['module.decoder_pred_v.weight'], mdl_weights['module.decoder_pred_v.bias']
        
        miss, unexpected = model.load_state_dict(mdl_weights, strict=False)
        if args.verbose:
            print(f"Missed parameters: {miss}\nUnexpected parameters: {unexpected}")
            print(f"Model succesfully loaded on {args.device} from {os.path.abspath(args.pretrained_checkpoint)}.")

    train(model, train_dl, validation_dl, args)
