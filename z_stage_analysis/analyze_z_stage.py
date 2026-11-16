"""THz-TDS Z-stage optical displacement analysis; run with config.yaml.

Uses the supplied THZDataset class with align=False. Never aligns the data
before estimating the physical delay. Results are apparent optical displacement,
not independently calibrated absolute stage position.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import correlate, correlation_lags
from thz_tds.dataset import THZDataset

C_UM_PER_PS = 299.792458

def cfg_get(d, key, default=None):
    return d.get(key, default)

def stage_to_um(value, cfg):
    unit = cfg['stage_unit'].lower()
    if unit == 'mm': return float(value)*1000
    if unit in ('um', 'µm'): return float(value)
    if unit == 'native': return float(value)/float(cfg['stage_units_per_mm'])*1000
    raise ValueError('stage_unit must be mm, um or native')

def time_to_ps(t, cfg):
    unit = cfg['time_unit'].lower()
    factor = {'ps':1, 'fs':1e-3, 's':1e12}.get(unit)
    if factor is None: raise ValueError('time_unit must be ps, fs, or s')
    return t*factor

def baseline_correct(y, pre):
    y = np.asarray(y, dtype=float)
    n = int(pre.get('baseline_samples', 30))
    if n > 0:
        if n >= y.shape[-1]: raise ValueError('baseline_samples must be smaller than trace length')
        if pre.get('subtract_linear_baseline', False):
            x = np.arange(y.shape[-1]); left = np.median(y[..., :n], axis=-1, keepdims=True)
            right = np.median(y[..., -n:], axis=-1, keepdims=True)
            y = y - (left + (right-left)*x/(len(x)-1))
        else:
            y = y - np.median(y[..., :n], axis=-1, keepdims=True)
    return y * int(pre.get('polarity',1))

def parabolic(a,b,c):
    denom = a-2*b+c
    return 0. if abs(denom) < 1e-25 else float(np.clip(0.5*(a-c)/denom,-1,1))

def estimate_delay(t, ref, signal, max_delay_ps, window_ps, anchor_ps=None):
    """Positive delay means signal arrives later than reference.

    Full-record cross-correlation finds coarse lag (including 20 ps shifts).
    A local, waveform-centred correlation refines it. Sub-sample parabolic
    interpolation gives a delay estimate, NOT guaranteed sub-sample accuracy.
    """
    dt = float(np.median(np.diff(t)))
    a = np.asarray(ref, float); b = np.asarray(signal, float)
    a = a - a.mean(); b = b - b.mean()
    corr = correlate(b, a, mode='full', method='fft')
    lags = correlation_lags(len(b),len(a),mode='full')
    mask = np.abs(lags*dt) <= max_delay_ps
    if not np.any(mask): return np.nan, np.nan
    idxs = np.flatnonzero(mask)
    k = idxs[np.argmax(corr[mask])]
    coarse = int(lags[k]); lag = float(coarse)
    # Refine using only the same pulse feature, after coarse registration.
    if window_ps is not None and window_ps > 0:
        center = int(np.argmax(np.abs(a))) if anchor_ps is None else int(np.argmin(abs(t-anchor_ps)))
        half = max(6,int(round(window_ps/(2*dt))))
        lo=max(0,center-half); hi=min(len(a),center+half+1)
        aa=a[lo:hi]; xx=np.arange(lo,hi)+coarse
        ok=(xx>=0)&(xx<len(b))
        if ok.sum()>=10:
            aa=aa[ok]; bb=b[xx[ok]]
            # small-lag correlation in the local window
            local = correlate(bb-bb.mean(),aa-aa.mean(),mode='full',method='fft')
            local_lags=correlation_lags(len(bb),len(aa),mode='full')
            small=np.abs(local_lags)<=max(2,int(round(min(window_ps/4,0.5)/dt)))
            if np.any(small):
                loc_idx=np.flatnonzero(small)[np.argmax(local[small])]
                if 0<loc_idx<len(local)-1:
                    lag=coarse+float(local_lags[loc_idx])+parabolic(local[loc_idx-1],local[loc_idx],local[loc_idx+1])
                else: lag=coarse+float(local_lags[loc_idx])
    else:
        if 0<k<len(corr)-1: lag += parabolic(corr[k-1],corr[k],corr[k+1])
    # waveform agreement at estimated shift, only where both signals overlap
    # For b(t) = a(t-delay), to compare with a(t), sample b(t+delay)
    shifted=np.interp(t+lag*dt,t,b,left=np.nan,right=np.nan)
    good=np.isfinite(shifted)
    if good.sum()<10: return lag*dt, np.nan
    aa=a[good]; bb=shifted[good]
    denom=np.linalg.norm(aa-aa.mean())*np.linalg.norm(bb-bb.mean())
    quality=float(np.dot(aa-aa.mean(),bb-bb.mean())/denom) if denom>0 else np.nan
    return lag*dt, quality

def save_with_zoom(fig, ax, path, zoom, dpi):
    """Save full figure; if zoom=[t_min, t_max] (ps), also save <name>_zoom.png
    with the x-axis limited to that window and y rescaled to the visible data."""
    fig.tight_layout(); fig.savefig(path,dpi=dpi)
    if zoom:
        lo,hi=float(zoom[0]),float(zoom[1])
        ys=[l.get_ydata()[(l.get_xdata()>=lo)&(l.get_xdata()<=hi)] for l in ax.get_lines()]
        ys=np.concatenate([y for y in ys if len(y)]) if any(len(y) for y in ys) else np.array([])
        ax.set_xlim(lo,hi)
        if len(ys):
            pad=0.05*(ys.max()-ys.min() or 1); ax.set_ylim(ys.min()-pad,ys.max()+pad)
        fig.tight_layout(); fig.savefig(path.with_name(path.stem+'_zoom'+path.suffix),dpi=dpi)
    plt.close(fig)

def plot_save(path, x,y,xlabel,ylabel,title,zero=True,zticks=None):
    fig,ax=plt.subplots(figsize=(10,4.5))
    ax.plot(x,y,'o-',markersize=4)
    if zticks is not None:  # label measurement order with commanded Z (Z repeats at returns)
        ax.set_xticks(list(x)); ax.set_xticklabels([f'{v:.4f}' for v in zticks],rotation=45,ha='right',fontsize=8)
    if zero: ax.axhline(0,color='gray',linestyle='--',linewidth=1)
    ax.set(xlabel=xlabel,ylabel=ylabel,title=title); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(path,dpi=170); plt.close(fig)

def main(config_path):
    with open(config_path,encoding='utf-8') as f: cfg=yaml.safe_load(f)
    base=Path(cfg['data_dir']).expanduser()
    if not base.is_absolute(): base=(config_path.parent/base).resolve()
    if not base.exists(): raise FileNotFoundError(f'Measurement folder not found: {base}')
    out=Path(cfg['output_dir'])
    if not out.is_absolute(): out=(config_path.parent/out).resolve()
    out=out/'cross_correlation'  # separate from analyze_gaussian_jitter.py outputs
    out.mkdir(parents=True,exist_ok=True)
    manifest=pd.read_csv(base/'manifest.csv')
    required={'cycle','step','z_command','previous_z_command','approach_direction','folder'}
    if not required.issubset(manifest.columns): raise ValueError(f'manifest.csv missing columns: {required-set(manifest.columns)}')
    manifest=manifest.reset_index(drop=True)
    angle=float(cfg['incidence_angle_deg']); cosine=np.cos(np.deg2rad(angle))
    if cosine <= 0.05: raise ValueError('incidence_angle_deg invalid for this reflection model')
    if cfg['analysis'].get('drift_correction','none')!='none':
        raise ValueError('Only drift_correction: none is implemented; do not silently subtract stage motion.')
    pre=cfg['preprocessing']; opts=cfg['analysis']
    max_traces=opts.get('max_traces_per_position')
    correction=float(cfg.get('correction_factor',0.0414))
    data=[]
    for idx,row in manifest.iterrows():
        folder=base/str(row.folder)
        if not folder.is_dir():
            warnings.warn(f'Missing folder {folder}; skipping');continue
        ds=THZDataset(folder,correction_factor=correction,align=False,fit_method=None,max_traces=max_traces)
        t=time_to_ps(ds.x_ps,cfg)
        if np.any(np.diff(t)<=0): raise ValueError(f'Non-increasing time axis: {folder}')
        raw=baseline_correct(ds.all_y_raw,pre)
        # Dataset interpolates onto a common axis if traces differ. Warn in README.
        avg=np.mean(raw,axis=0)
        data.append({'row':row,'idx':idx,'t':t,'raw':raw,'avg':avg,'folder':folder})
        print(f'Loaded {folder.name}: {len(raw)} traces, {len(t)} samples')
    if not data: raise RuntimeError('No valid measurement folders')
    first=data[0]; t0=first['t']; baseline=first['avg']
    for d in data:
        if len(d['t'])!=len(t0) or np.max(abs(d['t']-t0))>float(opts.get('max_time_grid_error_ps',1e-5)):
            raise ValueError(f'Time grids differ across folders: {d["folder"]}; check export/time-axis calibration')
    refz=float(cfg['z_reference']); factor=C_UM_PER_PS/(2*cosine)
    maxdelay=float(opts['max_delay_ps']); window=float(opts['local_window_ps'])
    # Determine sign from measured excursions only (not reference returns).
    preliminary=[]
    for d in data:
        lag,q=estimate_delay(t0,baseline,d['avg'],maxdelay,window)
        expected=stage_to_um(float(d['row'].z_command)-refz,cfg)
        preliminary.append((lag,q,expected))
    mode=opts.get('sign_from_stage','auto')
    if mode=='auto':
        eligible=[(lag,exp) for lag,_,exp in preliminary if np.isfinite(lag) and abs(exp)>20 and abs(lag)>0.03]
        if not eligible: raise ValueError('Cannot determine delay sign: no usable nonzero displacement. Set sign_from_stage explicitly after inspection.')
        sign=1 if np.median([np.sign(lag*exp) for lag,exp in eligible])>0 else -1
    elif mode=='positive': sign=1
    elif mode=='negative': sign=-1
    else: raise ValueError('sign_from_stage must be auto, positive or negative')
    results=[]; individual=[]
    minq=float(opts.get('min_correlation',0.7)); minamp=float(opts.get('min_peak_amplitude_ratio',0.05))
    refamp=max(abs(baseline))
    for d,(delay,q,expected) in zip(data,preliminary):
        row=d['row']; amp=float(max(abs(d['avg']))/refamp)
        quality_ok=bool(np.isfinite(q) and q>=minq and amp>=minamp)
        displacement=sign*factor*delay
        is_return=bool(abs(expected)<1e-8)
        results.append({'measurement_index':d['idx'],'cycle':row.cycle,'step':row.step,
            'folder':row.folder,'z_command':row.z_command,'previous_z_command':row.previous_z_command,
            'approach_direction':row.approach_direction,'is_reference_return':is_return,
            'expected_um':expected,'delay_ps':delay,'measured_um':displacement,
            'error_um':displacement-expected,'correlation':q,'amplitude_ratio':amp,
            'quality_ok':quality_ok,'trace_count':len(d['raw'])})
        if opts.get('analyze_individual_traces',True):
            for j,trace in enumerate(d['raw'],1):
                # Compare each trace against the SAME baseline template: preserve jitter and motion.
                tr_delay,tr_q=estimate_delay(t0,baseline,trace,maxdelay,window)
                tr_amp=float(max(abs(trace))/refamp)
                ok=bool(np.isfinite(tr_q) and tr_q>=minq and tr_amp>=minamp)
                individual.append({'measurement_index':d['idx'],'cycle':row.cycle,'step':row.step,
                    'trace_number':j,'folder':row.folder,'delay_ps':tr_delay,
                    'measured_um':sign*factor*tr_delay,'error_um':sign*factor*tr_delay-expected,
                    'correlation':tr_q,'amplitude_ratio':tr_amp,'quality_ok':ok})
    df=pd.DataFrame(results); df.to_csv(out/'position_results.csv',index=False)
    if individual:
        traces=pd.DataFrame(individual)
        traces.to_csv(out/'trace_results.csv',index=False)
        grp=traces[traces.quality_ok].groupby('measurement_index').agg(
            trace_mean_um=('measured_um','mean'),trace_std_um=('measured_um','std'),
            valid_trace_count=('measured_um','count')).reset_index()
        df=df.merge(grp,on='measurement_index',how='left')
        df.to_csv(out/'position_results.csv',index=False)
    ref=df[df.is_reference_return & df.quality_ok]
    # Return scatter is not a calibrated absolute stage accuracy.
    summary={'positions':len(df),'quality_pass_positions':int(df.quality_ok.sum()),
        'delay_to_displacement_sign':int(sign),'incidence_angle_deg':angle,
        'reference_coordinate':refz,'reference_return_count':len(ref),
        'return_mean_um':float(ref.measured_um.mean()) if len(ref) else None,
        'return_std_um':float(ref.measured_um.std(ddof=1)) if len(ref)>1 else None,
        'return_max_abs_um':float(ref.measured_um.abs().max()) if len(ref) else None,
        'return_range_um':float(ref.measured_um.max()-ref.measured_um.min()) if len(ref) else None,
        'notes':['Apparent optical displacement includes ECOPS timing drift and optical-path changes.',
                 'Absolute stage accuracy requires independent displacement calibration.',
                 'Return statistics are relative to initial reference; reference itself is noisy.',
                 'Low-correlation excursion data may indicate defocus or pulse leaving time window.']}
    with open(out/'summary.yaml','w',encoding='utf-8') as f: yaml.safe_dump(summary,f,sort_keys=False,allow_unicode=True)
    plot_save(out/'01_displacement.png',df.expected_um,df.measured_um,'Commanded displacement (µm)',
              'Measured optical displacement (µm)','Displacement: commanded vs measured',False)
    fig,ax=plt.subplots(figsize=(7,6)); ax.scatter(df.expected_um,df.measured_um,c=np.where(df.quality_ok,1,0),cmap='coolwarm')
    v=np.r_[df.expected_um,df.measured_um]; ax.plot([min(v),max(v)],[min(v),max(v)],'k--',lw=1)
    ax.set(xlabel='Commanded displacement (µm)',ylabel='Measured displacement (µm)',title='Identity-line comparison');ax.grid(alpha=.25)
    fig.tight_layout();fig.savefig(out/'02_identity.png',dpi=170);plt.close(fig)
    zlab=f'Commanded Z ({cfg["stage_unit"]}), in measurement order'
    plot_save(out/'03_position_error.png',df.measurement_index,df.error_um,zlab,'Apparent error (µm)','Error across sequence',zticks=df.z_command)
    if len(ref): plot_save(out/'04_reference_returns.png',ref.measurement_index,ref.measured_um,zlab,'Apparent return displacement (µm)','Reference return repeatability',zticks=ref.z_command)
    if individual:
        fig,ax=plt.subplots(figsize=(10,4.5))
        for idx in df.measurement_index:
            if idx==0 or idx in set(ref.measurement_index):
                ss=traces[(traces.measurement_index==idx)&traces.quality_ok]
                if len(ss): ax.plot(ss.trace_number,ss.measured_um,label=f'Z = {df.loc[df.measurement_index==idx,"z_command"].iloc[0]:.4f} ({idx})',alpha=.65)
        ax.set(xlabel='Trace number within position',ylabel='Apparent displacement (µm)',title='Baseline and return traces: timing noise / drift')
        ax.grid(alpha=.25);ax.legend(fontsize=8);fig.tight_layout();fig.savefig(out/'05_trace_stability.png',dpi=170);plt.close(fig)
    # Diagnostic waveforms: raw baseline and all position averages, no alignment
    fig,ax=plt.subplots(figsize=(10,5))
    ax.plot(t0,baseline,label='initial baseline',linewidth=2)
    for d in data[1:]: ax.plot(t0,d['avg'],alpha=.5,label=f'Z = {d["row"].z_command:.4f} ({d["idx"]})')
    ax.set(xlabel='Time (ps)',ylabel='Electric field (scaled)',title='Measured waveforms: NO alignment applied')
    ax.grid(alpha=.25)
    if len(data)<=15: ax.legend(fontsize=7,ncol=2)
    save_with_zoom(fig,ax,out/'00_raw_waveforms.png',cfg.get('plot',{}).get('zoom_window_ps'),170)
    print('\nRESULTS:',out,'\n',pd.Series({k:v for k,v in summary.items() if k!='notes'}).to_string())
    print('WARNING: Optical displacement is not independent absolute stage accuracy.')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,default=Path(__file__).with_name('config.yaml'))
    args=ap.parse_args();main(args.config.resolve())
