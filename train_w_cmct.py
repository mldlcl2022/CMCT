import argparse
from utils.seed import seed
import torch.distributed as dist
import os
import torch
from utils.dataset import CMCTDataset
from torch.utils.data import DistributedSampler, DataLoader
from models.cmct import CMCT
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import time
from utils.lr_scheduler import adjust_learning_rate
from utils.evaluate import evaluate_cbm

import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
# experiment argument #
parser.add_argument('--exp_name', type= str, required= True)
parser.add_argument('--seed', default= 7, type= int)
# data argument #
parser.add_argument('--data_name', type= str, required= True)
parser.add_argument('--csv_dir', type= str, required= True)
parser.add_argument('--root_dir', type= str, required= True)
parser.add_argument('--batch_size', default= 64, type= int)
# model argument #
parser.add_argument('--model_name', type= str, required= True)
parser.add_argument('--pretrained', default= 'True', type= str)
# train argument #
parser.add_argument('--epochs', default= 30, type= int)

def main():
    # argparse #
    args = parser.parse_args()
    
    # seed #
    seed(args.seed)
    
    # ddp #
    dist.init_process_group(backend= 'nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
    
    # data #
    dataloaders = {}
    for split in ['train','valid','test']:
        dataset = CMCTDataset(args.data_name, split, args.csv_dir, args.root_dir)
        
        if split == 'train':
            sampler = DistributedSampler(dataset, shuffle= True)
            dataloaders[split] = DataLoader(
                dataset, batch_size= args.batch_size, sampler= sampler, num_workers= 32
            )
            num_concepts = dataset[0][1].shape[0]
            num_classes = dataset[0][2].shape[0]
        else:
            dataloaders[split] = DataLoader(
                dataset, batch_size= args.batch_size, shuffle= False, num_workers= 32
            )
    
    # model #
    if args.pretrained == 'True':
        model = CMCT(args.model_name, True, num_concepts, num_classes)
    else:
        model = CMCT(args.model_name, False, num_concepts, num_classes)
    model = model.to(device, memory_format= torch.channels_last)
    model = DDP(
        model,
        device_ids= [local_rank],
        output_device= local_rank,
        gradient_as_bucket_view= False
    )
    
    # train setting #
    c_criterion = nn.SmoothL1Loss(beta= 1.0)
    t_criterion = nn.BCEWithLogitsLoss()
    
    optimizer = optim.AdamW(model.parameters(), lr= 5e-4, weight_decay= 1e-3)
    lr_config = {
        'lr': 5e-4,
        'min_lr': 1e-5,
        'warmup_epochs': 20,
        'epochs': args.epochs
    }
    
    # result setting #
    if dist.get_rank() == 0:
        result_path = f'./results/{args.exp_name}/{args.data_name}/{args.model_name}/seed{args.seed}'
        os.makedirs(result_path, exist_ok= True)
        best_dict = os.path.join(result_path, 'best_dict.pth')
        writer = SummaryWriter(log_dir= os.path.join(result_path, 'tensorboard'))
    best_epoch = 0
    best_f1 = float('-inf')
    
    # loss setting #
    world_size = dist.get_world_size()
    USE_DYNAMIC_WEIGHT = True
    EMA_ALPHA = 0.9
    POWER_P = 1.5
    EPS = 1e-8
    LAMBDA_MIN, LAMBDA_MAX = 0.1, 0.9
    SYNC_EMA_ACROSS_DDP = True
    NORMALIZE_BY_EMA = False
    
    # train #
    for epoch in range(args.epochs):
        s = time.time()
        dataloaders['train'].sampler.set_epoch(epoch)
        current_lr = adjust_learning_rate(optimizer, epoch, lr_config)
        
        # train session #
        model.train()
        train_concept_loss, train_task_loss, train_total_loss = 0.0, 0.0, 0.0
        
        ema_c, ema_t = None, None
        lambda_c_sum, lambda_t_sum, lambda_steps= 0.0, 0.0, 0
        
        for img, c_gt, t_gt in dataloaders['train']:
            img = img.to(device, memory_format= torch.channels_last, non_blocking= True)
            c_gt, t_gt = c_gt.to(device), t_gt.to(device)
            
            c_pred, t_pred = model(img)
            
            c_loss = c_criterion(c_pred, c_gt)
            t_loss = t_criterion(t_pred, t_gt)
            
            # '''loss default'''
            # loss = c_loss + t_loss
            
            '''dynamic loss'''
            if USE_DYNAMIC_WEIGHT:
                with torch.no_grad():
                    if SYNC_EMA_ACROSS_DDP:
                        pair = torch.tensor([c_loss.detach().item(), t_loss.detach().item()],
                                            device=device, dtype=torch.float32)
                        dist.all_reduce(pair, op=dist.ReduceOp.SUM)
                        pair /= world_size
                        c_val, t_val = pair[0].item(), pair[1].item()
                    else:
                        c_val, t_val = c_loss.detach().item(), t_loss.detach().item()
                
                if ema_c is None:
                    ema_c, ema_t = c_val, t_val
                else:
                    ema_c = EMA_ALPHA * ema_c + (1 - EMA_ALPHA) * c_val
                    ema_t = EMA_ALPHA * ema_t + (1 - EMA_ALPHA) * t_val
                
                wc = (ema_c + EPS) ** POWER_P
                wt = (ema_t + EPS) ** POWER_P
                S = wc + wt
                lambda_c = wc / S
                lambda_t = wt / S
                
                lambda_c = float(max(LAMBDA_MIN, min(LAMBDA_MAX, lambda_c)))
                lambda_t = 1.0 - lambda_c
                
                if NORMALIZE_BY_EMA:
                    loss = lambda_c * (c_loss / (ema_c + EPS)) + lambda_t * (t_loss / (ema_t + EPS))
                else:
                    loss = lambda_c * c_loss + lambda_t * t_loss
                
                lambda_c_sum += lambda_c
                lambda_t_sum += lambda_t
                lambda_steps += 1
            else:
                loss = c_loss + t_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_concept_loss += c_loss.item()
            train_task_loss += t_loss.item()
            train_total_loss += (c_loss.item() + t_loss.item())
        train_concept_loss /= len(dataloaders['train'])
        train_task_loss /= len(dataloaders['train'])
        train_total_loss /= len(dataloaders['train'])
        
        # dynamic loss weight's lambda setting #
        if USE_DYNAMIC_WEIGHT:
            lam_tensor = torch.tensor(
                [lambda_c_sum, lambda_t_sum, float(lambda_steps)],
                device=device, dtype=torch.float64
            )
            dist.all_reduce(lam_tensor, op=dist.ReduceOp.SUM)
            
            lambda_c_mean = (lam_tensor[0].item()) / (lam_tensor[2].item())
            lambda_t_mean = (lam_tensor[1].item()) / (lam_tensor[2].item())
        else:
            lambda_c_mean, lambda_t_mean = 0.5, 0.5
        
        # valid session #
        if dist.get_rank() == 0:
            model.eval()
            valid_concept_loss, valid_task_loss, valid_total_loss = 0.0, 0.0, 0.0
            ema_c, ema_t = None, None
            
            with torch.no_grad():
                for img, c_gt, t_gt in dataloaders['valid']:
                    img = img.to(device, memory_format= torch.channels_last, non_blocking= True)
                    c_gt, t_gt = c_gt.to(device), t_gt.to(device)
                    
                    c_pred, t_pred = model(img)
                    
                    c_loss = c_criterion(c_pred, c_gt)
                    t_loss = t_criterion(t_pred, t_gt)
                    
                    '''loss default'''
                    loss = c_loss + t_loss
                    
                    valid_concept_loss += c_loss.item()
                    valid_task_loss += t_loss.item()
                    valid_total_loss += loss.item()
            valid_concept_loss /= len(dataloaders['valid'])
            valid_task_loss /= len(dataloaders['valid'])
            valid_total_loss /= len(dataloaders['valid'])
            
            trainscore = evaluate_cbm(model, dataloaders['train'], device, num_classes)
            validscore = evaluate_cbm(model, dataloaders['valid'], device, num_classes)
            
            # save learning state #
            writer.add_scalars('Loss(concept)', {'train': train_concept_loss, 'valid': valid_concept_loss}, epoch)
            writer.add_scalars('Loss(task)', {'train': train_task_loss, 'valid': valid_task_loss}, epoch)
            writer.add_scalars('Loss(total)', {'train': train_total_loss, 'valid': valid_total_loss}, epoch)
            
            if USE_DYNAMIC_WEIGHT:
                writer.add_scalars('Lambda', {'concept': lambda_c_mean, 'task': lambda_t_mean}, epoch)
            
            writer.add_scalars('AUC', {'train': trainscore["auc_avg"], 'valid': validscore["auc_avg"]}, epoch)
            writer.add_scalars('F1', {'train': trainscore["f1_avg"], 'valid': validscore["f1_avg"]}, epoch)
            
            # best state #
            if best_f1 < validscore['f1_avg']:
                best_epoch = epoch + 1
                best_f1 = validscore['f1_avg']
                torch.save(model.module.state_dict(), best_dict)
            
            # print state #
            e = time.time()
            t = e - s
            print(
                f'Epoch [{epoch+1:>3}/{args.epochs}] {t//60:2.0f}m {t%60:2.0f}s - '
                f'Loss - Concept: {train_concept_loss:.4f} Task: {train_task_loss:.4f} Total: {train_total_loss:.4f}, '
                f'LR: {current_lr:.6f}', flush= True
            )
    
    # result #
    if dist.get_rank() == 0:
        final_dict = torch.load(best_dict, map_location= device)
        if isinstance(model, DDP):
            model.module.load_state_dict(final_dict)
        else:
            model.load_state_dict(final_dict)
        
        testscore = evaluate_cbm(model, dataloaders['test'], device, num_classes)
        result_txt = os.path.join(result_path, 'result.txt')
        with open(result_txt, 'w') as f:
            f.write(f'best epoch: {best_epoch}\n')
            for k,v in testscore.items():
                f.write(f'{k}: {v}\n')
        
        writer.close()
        print('Exp complete :D')
    
    dist.destroy_process_group()

if __name__ == '__main__':
    main()