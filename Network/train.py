import math
import os
import time

import numpy as np
import torch
from torch._C import dtype
import torch.nn as nn

from scipy.optimize import curve_fit
from scipy import asarray as ar,exp
import matplotlib.pyplot as plt
import tqdm
import PIL.Image as Image

from Network.model import get_model
from Network.dataloader import EchoSet
from Network.test import test

        
def train(  dataset_path,
            num_epochs=20,
            device="cuda",
            batch_size=2,
            seed=0,
            run_test=False,
            lr = 1e-5,
            modelname=None,
            latent_dim=1024,
            lr_step_period=None,
            ds_max_length = 128,
            ds_min_spacing = 10,
            DTmode = 'repeat',
            SDmode = 'reg',
            num_hidden_layers = 16,
            intermediate_size = 8192,
            rm_branch = None, # SD / EF / None
            use_conv = False,
            attention_heads = 16
            ):
    
    # Seed RNGs
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    dataroot = dataset_path  
    if not os.path.exists(dataroot):
        raise ValueError(dataroot+" does not exist.")  
        
    # Set default output directory
    output = os.path.join(".", "output", modelname)
    os.makedirs(output, exist_ok=True)
    
    # Set device for computations
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if type(device) == type(list()):
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in device)
        device = "cuda"
    device = torch.device(device)
    print("Using device:", device)

    # Set up model
    model = get_model(latent_dim, img_per_video=ds_max_length, SDmode=SDmode, num_hidden_layers=num_hidden_layers, intermediate_size=intermediate_size, rm_branch=rm_branch, use_conv=use_conv,attention_heads=attention_heads)
    
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(modelname, model.__class__.__name__, "contains", pytorch_total_params, "parameters.")
    model = nn.DataParallel(model)
    model.to(device)
    
    # Set up optimizer
    optim = torch.optim.AdamW(model.parameters(), lr=lr)
    if lr_step_period is None:
        lr_step_period = math.inf
    scheduler = torch.optim.lr_scheduler.StepLR(optim, lr_step_period)
    
    # Set up datasets and dataloaders
    train_set = EchoSet(  root=dataroot,
                            split="train",
                            min_spacing=ds_min_spacing,
                            max_length=ds_max_length,
                            fixed_length=ds_max_length,
                            pad=8,
                            random_clip=False,
                            dataset_mode=DTmode,
                            SDmode = SDmode)
    val_set   = EchoSet(  root=dataroot,
                            split="val",
                            min_spacing=ds_min_spacing,
                            max_length=ds_max_length,
                            fixed_length=ds_max_length,
                            pad=8,
                            random_clip=False,
                            dataset_mode=DTmode,
                            SDmode = SDmode)
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, num_workers=0, shuffle=True, pin_memory=(device.type == "cuda"), drop_last=True)
    val_dataloader   = torch.utils.data.DataLoader(val_set,   batch_size=batch_size, num_workers=0, shuffle=True, pin_memory=(device.type == "cuda"), drop_last=True)
    dataloaders = {'train': train_dataloader, 'val': val_dataloader}
    
    # Compute EF imbalance
    
    ejections = np.round(train_set.ejection)
    unique, count = np.unique(ejections, return_counts=True)
    n = len(unique)                          #the number of data
    mean = sum(unique*count)/n                   #note this correction
    sigma = sum(count*(unique-mean)**2)/n        #note this correction   
    
    x = np.ones(101)
    x[unique.astype(np.int)] = count
    x = np.log(x)
    x = x/x.max()
    mult = 1-x
    mult = torch.tensor(mult, dtype=torch.float)
    mult = torch.nn.functional.pad(mult, (4,4), mode='constant', value=1)
    mult = smooth(mult, 5, 2).numpy()[4:-4]
    model.mult = mult
    
    
    # Run training and testing loops
    with open(os.path.join(output, "log.csv"), "a") as f:
        epoch_resume = 0
        bestLoss = float("inf")
        try:
            # Attempt to load checkpoint
            checkpoint = torch.load(os.path.join(output, "checkpoint.pt"))
            model.load_state_dict(checkpoint['state_dict'])
            optim.load_state_dict(checkpoint['opt_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_dict'])
            epoch_resume = checkpoint["epoch"] + 1
            bestLoss = checkpoint["best_loss"]
            f.write("Resuming from epoch {}\n".format(epoch_resume))
        except FileNotFoundError:
            f.write("Starting run from scratch\n")

        for epoch in range(epoch_resume, num_epochs):
            print("Epoch {} / {}".format(epoch, num_epochs), flush=True)
            for phase in ['train', 'val']:
                start_time = time.time()
                print("Running on", phase)
                loss, _ = run_epoch(model, dataloaders[phase], phase == "train", optim, device, epoch)

                f.write("{},{},{},{},{},{},{},{},{},{},{}\n".format(epoch, phase, loss, 0, 0, 0, time.time() - start_time, 0, 0, 0, batch_size))
                f.flush()
            scheduler.step()

            # Save checkpoint
            save = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'best_loss': bestLoss,
                'loss': loss,
                'opt_dict': optim.state_dict(),
                'scheduler_dict': scheduler.state_dict(),
            }
            torch.save(save, os.path.join(output, "checkpoint.pt"))
            if loss < bestLoss:
                torch.save(save, os.path.join(output, "best.pt"))
                bestLoss = loss
                
            test(dataset_path=dataset_path,
                 SDmode=SDmode, 
                 DTmode=DTmode, 
                 use_full_videos=False, 
                 latent_dim=latent_dim, 
                 fixed_length=ds_max_length, 
                 num_hidden_layers=num_hidden_layers, 
                 intermediate_size=intermediate_size, 
                 rm_branch = rm_branch, 
                 use_conv = use_conv,
                 attention_heads=16,
                 model_path = output)
    
    if run_test:
        # Run on validation and test
        for split in ["val", "test"]:
            print("Evaluating on", split)

            dataset = EchoSet(
                root=dataroot,
                split=split,
                min_spacing=ds_min_spacing,
                max_length=ds_max_length,
                fixed_length=ds_max_length,
                pad=8,
                random_clip=False,
                dataset_mode=DTmode,
                SDmode = SDmode)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=0, shuffle=False, pin_memory=(device.type == "cuda"))
            
            _, loss_hist = run_epoch(model, dataloader, False, None, device, output=output)

            with open(os.path.join(output, "{}_dice.csv".format(split)), "w") as g:
                g.write("Filename, Overall, Large, Small\n")
                for (filename, loss) in zip(dataset.fnames, loss_hist):
                    g.write("{},{}\n".format(filename, loss))

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
def run_epoch(model, dataloader, train, optim, device, epoch=0, output=None):
    """Run one epoch of training/evaluation for segmentation.

    Args:
        model (torch.nn.Module): Model to train/evaulate.
        dataloder (torch.utils.data.DataLoader): Dataloader for dataset.
        train (bool): Whether or not to train model.
        optim (torch.optim.Optimizer): Optimizer
        device (torch.device): Device to run on
    """
    
    total = 0.
    n = 0
    loss_hist = []
    
    DTmode = dataloader.dataset.mode
    SDmode = dataloader.dataset.SDmode 

    model.train(train)
    print("Train:", train, DTmode, SDmode)
    if train:
        print("Learning rate:", get_lr(optim))
    
    
    # Classification
    if   SDmode == 'cla':
        weighting = torch.tensor([1., 5., 5.]).to(device)
        loss_fct1 = nn.CrossEntropyLoss(weight=weighting, reduction='mean')
    elif SDmode == 'reg':
        loss_fct1 = nn.MSELoss(reduction='mean')
    else:
        raise ValueError("Wrong SDmode:", SDmode)
    
    loss_fct2 = nn.MSELoss(reduction='none')
    loss_fct3 = nn.L1Loss(reduction='none')

    with torch.set_grad_enabled(train):
        with tqdm.tqdm(total=len(dataloader)) as pbar:
            for (filename, video, label, ejection, repeat, fps) in dataloader:

                nB, nF, nC, nH, nW = video.shape
                
                # Merge batch and frames dimension
                video = video.view(nB*nF,nC,nH,nW)
                video = video.to(device, dtype=torch.float32)
                
                ef_label = ejection/100.0
                ef_label = ef_label.to(device, dtype=torch.float32)

                class_vec, ef_pred = model(video, nB, nF)
                
                if class_vec is not None:
                    if  SDmode == 'cla' and DTmode == 'repeat':
                        label = label.to(device, dtype=torch.long)
                        loss1 = loss_fct1(class_vec.view(-1, 3), label.view(-1))
                        
                    elif SDmode == 'cla' and DTmode == 'sample':
                        label = label.to(device, dtype=torch.long)
                        active_loss = repeat.view(-1) == 1
                        active_logits = class_vec.view(-1, 3)
                        active_labels = torch.where( active_loss.to(device), label.view(-1), torch.tensor(loss_fct1.ignore_index).type_as(label))
                        loss1 = loss_fct1(active_logits, active_labels)

                    elif SDmode == 'reg' and DTmode == 'repeat':
                        label = label.to(device, dtype=torch.float)
                        loss1 = loss_fct1(class_vec.view(nB, nF), label.view(nB, nF))
                        
                    elif SDmode == 'reg' and DTmode == 'sample':
                        label = label.to(device, dtype=torch.float)
                        attention = repeat.view(-1) == 1
                        class_vec_attention = class_vec.view(-1)[attention]
                        label_attention = label.view(-1)[attention]
                        loss1 = loss_fct1(class_vec_attention, label_attention)
                    
                else:
                    loss1 = torch.tensor(0)
                    
                if ef_pred is not None: 
                    loss2 = loss_fct2(ef_pred, ef_label).mean()*5
                    loss = loss1+loss2

                else:
                    loss = loss1
                    loss2 = torch.tensor(0)
                
                # Take gradient step if training
                if train:
                    optim.zero_grad()
                    loss.backward()
                    optim.step()

                # Accumulate losses and compute baselines
                total += loss.item()
                n += 1
                loss_hist.append(loss.item())
                
                mavg = np.mean(loss_hist[max(-len(loss_hist),-10):])

                # Show info on process bar
                pbar.set_postfix_str("{:.4f} / {:.4f} {:.4f} / {:.4f}".format(total / n, loss1.item(), loss2.item(), mavg))
                pbar.update()
                                
    loss_hist = np.array(loss_hist)
        
    return (total / 1 ), loss_hist

def smooth(vec, window=5, rep=1):
    weight = torch.ones((1,1, window))/window
    for _ in range(rep):
        pad = int((window-1)/2)
        vec = vec.unsqueeze(0).unsqueeze(0)
        vec = torch.nn.functional.conv1d(vec, weight, bias=None, stride=1, padding=pad, dilation=1, groups=1).squeeze()
    return vec