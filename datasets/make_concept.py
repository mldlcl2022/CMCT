import argparse
import pandas as pd
import os
from tqdm import tqdm
import wfdb
import neurokit2 as nk
import numpy as np
import copy
from sklearn.preprocessing import StandardScaler
import joblib

import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--data_name', type= str, required= True)
parser.add_argument('--root_dir', type= str, required= True)

def main():
    args = parser.parse_args()
    
    total = {}
    splits = ['train','valid','test']
    for split in splits:
        df = pd.read_csv(os.path.join('./raw', args.data_name, f'{split}.csv'))
        
        if args.data_name == 'cpsc': label_start_col = 7
        elif args.data_name == 'ptbxl_super': label_start_col = 5
        
        label_cols = df.columns[label_start_col:]
        other_labels = [col for col in label_cols if col != 'NORM']
        df_filtered = df[~((df['NORM'] == 1) & (df[other_labels].sum(axis= 1) > 0))]
        
        total[split] = df_filtered
    df_total = pd.concat([total['train'], total['valid'], total['test']], ignore_index= True)
    
    fs = 500
    ecg_features = []
    for idx, filepath in enumerate(tqdm(df_total.filepath, desc= 'extract concept', unit= 'record')):
        '''load signal data'''
        if args.data_name == 'cpsc': sig_path = os.path.join(args.root_dir, 'cpsc', filepath)
        elif args.data_name == 'ptbxl_super': sig_path = os.path.join(args.root_dir, 'ptbxl', filepath.replace('records500/',''))
        sig, _ = wfdb.rdsamp(sig_path)
        
        '''ECG feature's initial settings'''
        rr_interval, heartrate, pr_segment, st_segment, pr_interval = np.nan, np.nan, np.nan, np.nan, np.nan
        qrs_interval, st_interval, qt_interval, qtc, p_wave_duration = np.nan, np.nan, np.nan, np.nan, np.nan
        hrv_rmssd, hrv_sdsd, hrv_sdnn, hrv_cvnn, hrv_iqrnn = np.nan, np.nan, np.nan, np.nan, np.nan
        hrv_sd1, hrv_sd2, hrv_sd1sd2, hrv_cvsd, hrv_madnn = np.nan, np.nan, np.nan, np.nan, np.nan
        
        '''extract ECG features'''
        _, rpeaks_raw = nk.ecg_peaks(sig[:,1], fs)
        clean_sig = nk.ecg_clean(sig[:,1], fs, method= 'neurokit')
        _, rpeaks_clean = nk.ecg_peaks(clean_sig, fs)
        ## Peak-Related Concept features ##
        ### RR interval
        if len(rpeaks_raw['ECG_R_Peaks']) < len(rpeaks_clean['ECG_R_Peaks']):
            rpeaks = rpeaks_clean
        else:
            rpeaks = rpeaks_raw
        r_locs = rpeaks['ECG_R_Peaks']
        rr_interval = np.median(np.diff(r_locs)/fs)
        ### Heart Rate
        heartrate = 60 / rr_interval
        ### keys
        _, waves = nk.ecg_delineate(clean_sig, r_locs, fs, method= 'dwt')
        qrs_on  = np.array(waves.get('ECG_QRS_Onsets',  waves.get('ECG_R_Onsets')),  dtype= float)
        qrs_off = np.array(waves.get('ECG_QRS_Offsets', waves.get('ECG_R_Offsets')), dtype= float)
        p_on  = np.array(waves.get('ECG_P_Onsets'),  dtype= float) if 'ECG_P_Onsets'  in waves else np.full_like(qrs_on, np.nan)
        p_off = np.array(waves.get('ECG_P_Offsets'), dtype= float) if 'ECG_P_Offsets' in waves else np.full_like(qrs_on, np.nan)
        t_on  = np.array(waves.get('ECG_T_Onsets'),  dtype= float) if 'ECG_T_Onsets'  in waves else np.full_like(qrs_on, np.nan)
        t_off = np.array(waves.get('ECG_T_Offsets'), dtype= float) if 'ECG_T_Offsets' in waves else np.full_like(qrs_on, np.nan)
        ### PR segment
        prs_valid = np.isfinite(p_off) & np.isfinite(qrs_on) & (qrs_on > p_off)
        if np.any(prs_valid):
            pr_segment = float(np.median((qrs_on[prs_valid] - p_off[prs_valid]) / fs * 1000.0))
        ### ST segment
        sts_valid = np.isfinite(qrs_off) & np.isfinite(t_on) & (t_on > qrs_off)
        if np.any(sts_valid):
            st_segment = float(np.median((t_on[sts_valid] - qrs_off[sts_valid]) / fs * 1000.0))
        ### PR interval
        pri_valid = np.isfinite(p_on) & np.isfinite(qrs_on) & (qrs_on > p_on)
        if np.any(pri_valid):
            pr_interval = float(np.median((qrs_on[pri_valid] - p_on[pri_valid]) / fs * 1000.0))
        ### QRS interval
        qrsi_valid = np.isfinite(qrs_on) & np.isfinite(qrs_off) & (qrs_off > qrs_on)
        if np.any(qrsi_valid):
            qrs_interval = float(np.median((qrs_off[qrsi_valid] - qrs_on[qrsi_valid]) / fs * 1000.0))
        ### ST interval
        sti_valid = np.isfinite(qrs_off) & np.isfinite(t_off) & (t_off > qrs_off)
        if np.any(sti_valid):
            st_interval = float(np.median((t_off[sti_valid] - qrs_off[sti_valid]) / fs * 1000.0))
        ### QT interval
        qti_valid = np.isfinite(qrs_on) & np.isfinite(t_off) & (t_off > qrs_on)
        if np.any(qti_valid):
            qt_interval = float(np.median((t_off[qti_valid] - qrs_on[qti_valid]) / fs * 1000.0))
        ### QTc
        if np.isfinite(qt_interval) and np.isfinite(rr_interval) and (rr_interval > 0):
            qtc = float(qt_interval / np.sqrt(rr_interval))
        ### P wave duration
        pwd_valid = np.isfinite(p_on) & np.isfinite(p_off) & (p_off > p_on)
        if np.any(pwd_valid):
            p_wave_duration = float(np.median((p_off[pwd_valid] - p_on[pwd_valid]) / fs * 1000.0))
        ## HRV-Related Concept features ##
        try:
            hrv_features = nk.hrv(rpeaks, fs, show= False)
            ### HRV_RMSSD
            hrv_rmssd_valid = float(hrv_features.loc[0, 'HRV_RMSSD'])
            hrv_rmssd = hrv_rmssd_valid if np.isfinite(hrv_rmssd_valid) else np.nan
            ### HRV_SDSD
            hrv_sdsd_valid = float(hrv_features.loc[0, 'HRV_SDSD'])
            hrv_sdsd = hrv_sdsd_valid if np.isfinite(hrv_sdsd_valid) else np.nan
            ### HRV_SDNN
            hrv_sdnn_valid = float(hrv_features.loc[0, 'HRV_SDNN'])
            hrv_sdnn = hrv_sdnn_valid if np.isfinite(hrv_sdnn_valid) else np.nan
            ### HRV_CVNN
            hrv_cvnn_valid = float(hrv_features.loc[0, 'HRV_CVNN'])
            hrv_cvnn = hrv_cvnn_valid if np.isfinite(hrv_cvnn_valid) else np.nan
            ### HRV_IQRNN
            hrv_iqrnn_valid = float(hrv_features.loc[0, 'HRV_IQRNN'])
            hrv_iqrnn = hrv_iqrnn_valid if np.isfinite(hrv_iqrnn_valid) else np.nan
            ### HRV_SD1
            hrv_sd1_valid = float(hrv_features.loc[0, 'HRV_SD1'])
            hrv_sd1 = hrv_sd1_valid if np.isfinite(hrv_sd1_valid) else np.nan
            ### HRV_SD2
            hrv_sd2_valid = float(hrv_features.loc[0, 'HRV_SD2'])
            hrv_sd2 = hrv_sd2_valid if np.isfinite(hrv_sd2_valid) else np.nan
            ### HRV_SD1SD2
            hrv_sd1sd2_valid = float(hrv_features.loc[0, 'HRV_SD1SD2'])
            hrv_sd1sd2 = hrv_sd1sd2_valid if np.isfinite(hrv_sd1sd2_valid) else np.nan
            ### HRV_CVSD
            hrv_cvsd_valid = float(hrv_features.loc[0, 'HRV_CVSD'])
            hrv_cvsd = hrv_cvsd_valid if np.isfinite(hrv_cvsd_valid) else np.nan
            ### MADNN
            hrv_madnn_valid = float(hrv_features.loc[0, 'HRV_MadNN'])
            hrv_madnn = hrv_madnn_valid if np.isfinite(hrv_madnn_valid) else np.nan
        except:
            pass
        
        ecg_features.append({
            'filepath': filepath,
            'rr_interval': rr_interval * 1000.0,
            'heartrate': heartrate,
            'pr_segment': pr_segment,
            'st_segment': st_segment,
            'pr_interval': pr_interval,
            'qrs_interval': qrs_interval,
            'st_interval': st_interval,
            'qt_interval': qt_interval,
            'qtc': qtc,
            'p_wave_duration': p_wave_duration,
            'hrv_rmssd': hrv_rmssd,
            'hrv_sdsd': hrv_sdsd,
            'hrv_sdnn': hrv_sdnn,
            'hrv_cvnn': hrv_cvnn,
            'hrv_iqrnn': hrv_iqrnn,
            'hrv_sd1': hrv_sd1,
            'hrv_sd2': hrv_sd2,
            'hrv_sd1sd2': hrv_sd1sd2,
            'hrv_cvsd': hrv_cvsd,
            'hrv_madnn': hrv_madnn
        })
    df_new_total = pd.merge(df_total, pd.DataFrame(ecg_features), on= 'filepath')
    
    new_total = copy.deepcopy(df_new_total)
    ecg_feature_cols = new_total.columns[-20:]
    scaler = StandardScaler()
    
    new_total.dropna(subset= ecg_feature_cols, how= 'any', inplace= True)
    
    n_total = len(new_total)
    n_train = int(n_total * 0.8)
    n_valid = int(n_total * 0.1)
    new_train = new_total.iloc[:n_train].reset_index(drop= True)
    new_valid = new_total.iloc[n_train:n_train+n_valid].reset_index(drop= True)
    new_test = new_total.iloc[n_train+n_valid:].reset_index(drop= True)
    
    new_train[ecg_feature_cols] = scaler.fit_transform(new_train[ecg_feature_cols].values)
    new_valid[ecg_feature_cols] = scaler.transform(new_valid[ecg_feature_cols].values)
    new_test[ecg_feature_cols] = scaler.transform(new_test[ecg_feature_cols].values)
    
    save_path = f'./preprocessed/{args.data_name}'
    os.makedirs(save_path, exist_ok= True)
    new_train.to_csv(os.path.join(save_path, 'train.csv'), index= False)
    new_valid.to_csv(os.path.join(save_path, 'valid.csv'), index= False)
    new_test.to_csv(os.path.join(save_path, 'test.csv'), index= False)
    joblib.dump(scaler, os.path.join(save_path, 'scaler.joblib'))
    print('complete')

if __name__ == '__main__':
    main()