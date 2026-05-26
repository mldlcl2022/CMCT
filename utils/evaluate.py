import torch
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def evaluate_task(model, loader, device, num_classes):
    trues, preds = [], []
    
    if isinstance(model, torch.nn.DataParallel) or isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module
    else:
        model = model
        
    model.eval()
    with torch.no_grad():
        for data, gt in loader:
            data, gt = data.to(device), gt.to(device)
            
            pred = model(data)
            
            trues.append(gt.cpu().numpy())
            preds.append(torch.sigmoid(pred).cpu().numpy())
    
    trues = np.vstack(trues)
    preds = np.vstack(preds)
    
    f1s, prs, res, aucs, accs = [], [], [], [], []
    for i in range(num_classes):
        y = trues[:,i]
        p = preds[:,i]
        y_hat = (p > 0.5).astype(int)
        
        f1s.append(f1_score(y, y_hat, average= 'binary', zero_division= 0))
        prs.append(precision_score(y, y_hat, average= 'binary', zero_division= 0))
        res.append(recall_score(y, y_hat, average= 'binary', zero_division= 0))
        if np.any(y == 0) and np.any(y == 1):
            aucs.append(roc_auc_score(y, p))
        else:
            aucs.append(np.nan)
        accs.append(accuracy_score(y, y_hat))
    
    return {
        'f1': f1s, 'f1_avg': float(np.mean(f1s)),
        'precisions': prs, 'precision_avg': float(np.mean(prs)),
        'recalls': res, 'recall_avg': float(np.mean(res)),
        'aucs': aucs, 'auc_avg': float(np.nanmean(aucs)),
        'accs': accs, 'acc_avg': float(np.mean(accs))
    }

def evaluate_cbm(model, loader, device, num_classes):
    c_trues, c_preds, t_trues, t_preds = [], [], [], []
    
    if isinstance(model, torch.nn.DataParallel) or isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module
    else:
        model = model
    
    model.eval()
    with torch.no_grad():
        for data, c_gt, t_gt in loader:
            data, c_gt, t_gt = data.to(device), c_gt.to(device), t_gt.to(device)
            
            c_pred, t_pred = model(data)
            
            # concept #
            c_trues.append(c_gt.cpu().numpy())
            c_preds.append(c_pred.cpu().numpy())
            
            # task #
            t_trues.append(t_gt.cpu().numpy())
            t_preds.append(torch.sigmoid(t_pred).cpu().numpy())
    
    c_trues = np.vstack(c_trues);c_preds = np.vstack(c_preds)
    t_trues = np.vstack(t_trues);t_preds = np.vstack(t_preds)
    
    # concept evaluate #
    mses = mean_squared_error(c_trues, c_preds, multioutput= 'raw_values')
    rmses = np.sqrt(mses)
    r2s = r2_score(c_trues, c_preds, multioutput= 'raw_values')
    maes = mean_absolute_error(c_trues, c_preds, multioutput= 'raw_values')
    
    # task evaluate #
    f1s, prs, res, aucs, accs = [], [], [], [], []
    for i in range(num_classes):
        y = t_trues[:,i]
        p = t_preds[:,i]
        y_hat = (p > 0.5).astype(int)
        
        f1s.append(f1_score(y, y_hat, average= 'binary', zero_division= 0))
        prs.append(precision_score(y, y_hat, average= 'binary', zero_division= 0))
        res.append(recall_score(y, y_hat, average= 'binary', zero_division= 0))
        if np.any(y == 0) and np.any(y == 1):
            aucs.append(roc_auc_score(y, p))
        else:
            aucs.append(np.nan)
        accs.append(accuracy_score(y, y_hat))
    
    return {
        'f1': f1s, 'f1_avg': float(np.mean(f1s)),
        'precisions': prs, 'precision_avg': float(np.mean(prs)),
        'recalls': res, 'recall_avg': float(np.mean(res)),
        'aucs': aucs, 'auc_avg': float(np.mean(aucs)),
        'accs': accs, 'acc_avg': float(np.mean(accs)),
        'rmses': rmses, 'rmse_avg': float(np.mean(rmses)),
        'r2s': r2s, 'r2_avg': float(np.mean(r2s)),
        'mses': mses, 'mse_avg': float(np.mean(mses)),
        'maes': maes, 'mae_avg': float(np.mean(maes))
    }