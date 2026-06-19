import argparse
from utils.seed import seed
import torch
from utils.dataset import CMCTDataset
from torch.utils.data import DataLoader
from methods.cmct import CMCT
import torch.nn as nn
from utils.loss_function import DynamicLoss
import torch.optim as optim
import os
from torch.utils.tensorboard import SummaryWriter
import time
from utils.lr_scheduler import adjust_learning_rate
from utils.metric import evaluate_cmct

# import warnings
# warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
# experiment argument
parser.add_argument('--exp_name', type= str, required= True)
parser.add_argument('--seed', default= 7, type= int)
parser.add_argument('--gpu', default= 'cuda:0', type= str)
# data argument
parser.add_argument(
    '--data_name', type= str, required= True,
    choices= ['cpsc2018','ptbxl']
)
parser.add_argument(
    '--csv_dir', default= './datasets/splits', type= str
)
parser.add_argument('--root_dir', type= str, required= True)
parser.add_argument('--batch_size', default= 64, type= int)
# method argument
parser.add_argument(
    '--model_name', default= 'convnext', type= str,
    choices= ['convnext','inception','resnet','efficientnet','vit']
)
# train argument
parser.add_argument('--epochs', default= 100, type= int)

def main():
    # argparse
    args = parser.parse_args()
    
    # seed
    seed(args.seed)
    
    # device
    device = torch.device(args.gpu if torch.cuda.is_available() else 'cpu')
    print(f'device: \'{device}\'\n')
    
    # data
    dataloaders = {}
    for split in ['train','valid','test']:
        dataset = CMCTDataset(args.data_name, split, args.csv_dir, args.root_dir)
        
        if split == 'train':
            dataloaders[split] = DataLoader(
                dataset, batch_size= args.batch_size, shuffle= True, num_workers= 32
            )
            num_concepts = dataset[0][1].shape[0]
            num_labels = dataset[0][2].shape[0]
        else:
            dataloaders[split] = DataLoader(
                dataset, batch_size= args.batch_size, shuffle= False, num_workers= 32
            )
    
    # model
    model = CMCT(args.model_name, num_concepts, num_labels)
    model.to(device)
    
    # train setting
    c_criterion = nn.SmoothL1Loss(beta= 1.0)
    t_criterion = nn.BCEWithLogitsLoss()
    
    loss_fn = DynamicLoss(
        ema_alpha= 0.9,
        power_p= 1.5,
        eps= 1e-8,
        lambda_min= 0.1,
        lambda_max= 0.9
    )
    
    optimizer = optim.AdamW(model.parameters(), lr= 5e-4, weight_decay= 1e-3)
    lr_config = {
        'lr': 5e-4,
        'min_lr': 1e-5,
        'warmup_epochs': 20,
        'epochs': args.epochs
    }
    
    # result save setting
    result_path = f'./results/{args.exp_name}/{args.data_name}_{args.model_name}_seed{args.seed}'
    os.makedirs(result_path, exist_ok= True)
    
    best_epoch = 0
    best_f1 = float('-inf')
    best_dict = os.path.join(result_path, 'best_dict.pth')
    writer = SummaryWriter(log_dir= os.path.join(result_path, 'tensorboard'))
    
    # train
    for epoch in range(args.epochs):
        s = time.time()
        
        current_lr = adjust_learning_rate(optimizer, epoch, lr_config)
        loss_fn.reset()
        
        # train session
        model.train()
        train_concept_loss, train_task_loss, train_total_loss = 0.0, 0.0, 0.0
        
        for img, c_gt, t_gt in dataloaders['train']:
            img = img.to(device, memory_format= torch.channels_last, non_blocking= True)
            c_gt, t_gt = c_gt.to(device), t_gt.to(device)
            
            c_pred , t_pred = model(img)
            
            c_loss = c_criterion(c_pred, c_gt)
            t_loss = t_criterion(t_pred, t_gt)
            
            loss, lambda_c, lambda_t = loss_fn(c_loss, t_loss)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_concept_loss += c_loss.item()
            train_task_loss += t_loss.item()
            train_total_loss += (c_loss.item() + t_loss.item())
        train_concept_loss /= len(dataloaders['train'])
        train_task_loss /= len(dataloaders['train'])
        train_total_loss /= len(dataloaders['train'])
        
        # valid session
        model.eval()
        
        trainscore = evaluate_cmct(model, device, dataloaders['train'], num_labels)
        validscore = evaluate_cmct(model, device, dataloaders['valid'], num_labels)
        
        # save learning rate
        writer.add_scalars('Train Loss', {'concept_loss': train_concept_loss, 'task_loss': train_task_loss, 'total_loss': train_total_loss}, epoch)
        writer.add_scalars('F1', {'train': trainscore['f1_avg'], 'valid': validscore['f1_avg']}, epoch)
        writer.add_scalars('Accuracy', {'train': trainscore['accuracy_avg'], 'valid': validscore['accuracy_avg']}, epoch)
        
        # best state
        if best_f1 < validscore['f1_avg']:
            best_epoch = epoch + 1
            best_f1 = validscore['f1_avg']
            torch.save(model.state_dict(), best_dict)
        
        # print state
        e = time.time()
        t = e - s
        print(
            f'Epoch [{epoch+1}/{args.epochs}] {t//60:2.0f}m {t%60:2.0f}s - '
            f'Train Loss - Concept: {train_concept_loss:.4f} Task: {train_task_loss:.4f} Total: {train_total_loss:.4f}, '
            f'LR: {current_lr:.6f}', flush= True
        )
    
    # save result
    final_dict = torch.load(best_dict, map_location= device)
    model.load_state_dict(final_dict)
    
    testscore = evaluate_cmct(model, device, dataloaders['test'], num_labels)
    result_txt = os.path.join(result_path, 'result.txt')
    with open(result_txt, 'w') as f:
        f.write(f'best_epoch: {best_epoch}\n')
        for k, v in testscore.items():
            if any(metric in k for metric in ['f1', 'precision', 'recall', 'accuracy']):
                f.write(f'{k}: {v}\n')
    
    writer.close()
    print(f'Exp complete. Results saved to: {result_path}')

if __name__ == '__main__':
    main()