import torch
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

def evaluate_cmct(model, device, loader, num_labels):
    c_trues, c_preds = [], []
    t_trues, t_preds = [], []
    
    model.to(device)
    model.eval()
    with torch.no_grad():
        for img, c_gt, t_gt in loader:
            img = img.to(device, memory_format= torch.channels_last, non_blocking= True)
            c_gt, t_gt = c_gt.to(device), t_gt.to(device)
            
            c_pred, t_pred = model(img)
            
            # concept
            c_trues.append(c_gt.cpu().numpy())
            c_preds.append(c_pred.cpu().numpy())
            
            # task
            t_trues.append(t_gt.cpu().numpy())
            t_preds.append(t_pred.cpu().numpy())
    
    c_trues = np.vstack(c_trues)
    c_preds = np.vstack(c_preds)
    
    t_trues = np.vstack(t_trues)
    t_preds = np.vstack(t_preds)
    
    # concept evaluation
    mses = mean_squared_error(c_trues, c_preds, multioutput= 'raw_values')
    rmses = np.sqrt(mses)
    r2s = r2_score(c_trues, c_preds, multioutput= 'raw_values')
    maes = mean_absolute_error(c_trues, c_preds, multioutput= 'raw_values')
    
    # task evaluation
    f1s, prs, res, accs = [], [], [], []
    for i in range(num_labels):
        y = t_trues[:, i]
        p = t_preds[:, i]
        y_hat = (p > 0.5).astype(int)
        
        f1s.append(f1_score(y, y_hat, average= 'binary', zero_division= 0))
        prs.append(precision_score(y, y_hat, average= 'binary', zero_division= 0))
        res.append(recall_score(y, y_hat, average= 'binary', zero_division= 0))
        accs.append(accuracy_score(y, y_hat))
    
    return {
        'f1': f1s, 'f1_avg': float(np.mean(f1s)),
        'precisions': prs, 'precision_avg': float(np.mean(prs)),
        'recalls': res, 'recall_avg': float(np.mean(res)),
        'accuracys': accs, 'accuracy_avg': float(np.mean(accs)),
        'mses': mses, 'mse_avg': float(np.mean(mses)),
        'rmses': rmses, 'rmse_avg': float(np.mean(rmses)),
        'r2s': r2s, 'r2_avg': float(np.mean(r2s)),
        'maes': maes, 'mae_avg': float(np.mean(maes))
    }

def evaluate_task(model, device, loader, num_labels):
    t_trues, t_preds = [], []
    
    model.to(device)
    model.eval()
    with torch.no_grad():
        for img, t_gt in loader:
            img = img.to(device, memory_format= torch.channels_last, non_blocking= True)
            t_gt = t_gt.to(device)
            
            t_pred = model(img)
            
            t_trues.append(t_gt.cpu().numpy())
            t_preds.append(torch.sigmoid(t_pred).cpu().numpy())
    
    t_trues = np.vstack(t_trues)
    t_preds = np.vstack(t_preds)
    
    f1s, prs, res, accs = [], [], [], []
    for i in range(num_labels):
        y = t_trues[:, i]
        p = t_preds[:, i]
        y_hat = (p > 0.5).astype(int)
        
        f1s.append(f1_score(y, y_hat, average= 'binary', zero_division= 0))
        prs.append(precision_score(y, y_hat, average= 'binary', zero_division= 0))
        res.append(recall_score(y, y_hat, average= 'binary', zero_division= 0))
        accs.append(accuracy_score(y, y_hat))
    
    return {
        'f1': f1s, 'f1_avg': float(np.mean(f1s)),
        'precisions': prs, 'precision_avg': float(np.mean(prs)),
        'recalls': res, 'recall_avg': float(np.mean(res)),
        'accuracys': accs, 'accuracy_avg': float(np.mean(accs))
    }