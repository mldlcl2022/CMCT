import argparse
from utils.seed import seed
import torch
from utils.dataset import ImageDataset
from torch.utils.data import DataLoader
from methods.baseline import ImageModel
import torch.nn as nn
import torch.optim as optim
import os
from torch.utils.tensorboard import SummaryWriter
import time
from utils.lr_scheduler import adjust_learning_rate
from utils.metric import evaluate_task

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
        dataset = ImageDataset(args.data_name, split, args.csv_dir, args.root_dir)
        
        if split == 'train':
            dataloaders[split] = DataLoader(
                dataset, batch_size= args.batch_size, shuffle= True, num_workers= 32
            )
            num_labels = dataset[0][1].shape[0]
        else:
            dataloaders[split] = DataLoader(
                dataset, batch_size= args.batch_size, shuffle= False, num_workers= 32
            )
    
    # model
    model = ImageModel(args.model_name, num_labels)
    model.to(device)
    
    # train setting
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr= 5e-4, weight_decay= 1e-3)
    lr_config = {
        'lr': 5e-4,
        'min_lr': 1e-5,
        'warmup_epochs': 20,
        'epochs': args.epochs
    }
    
    # result setting
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
        
        # train session
        model.train()
        train_loss = 0.0
        
        for img, t_gt in dataloaders['train']:
            img = img.to(device, memory_format= torch.channels_last, non_blocking= True)
            t_gt = t_gt.to(device)
            
            t_pred = model(img)
            
            loss = criterion(t_pred, t_gt)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        train_loss /= len(dataloaders['train'])
        
        # valid session
        model.eval()
        
        trainscore = evaluate_task(model, device, dataloaders['train'], num_labels)
        validscore = evaluate_task(model, device, dataloaders['valid'], num_labels)
        
        # save learning rate
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
            f'Train Loss - {train_loss:.4f}, LR: {current_lr:.6f}', flush= True
        )
    
    # save result
    final_dict = torch.load(best_dict, map_location= device)
    model.load_state_dict(final_dict)
    
    testscore = evaluate_task(model, device, dataloaders['test'], num_labels)
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