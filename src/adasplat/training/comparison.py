"""Small CPU-only helpers for per-image and matched-subset comparisons."""
from pathlib import Path

def comparison(metrics, count, parameters, reference):
    if reference is None:
        return None
    result = {k:metrics[k]-reference['metrics'][k] for k in ('PSNR','SSIM','MS_SSIM')}
    result.update(N=count-reference['N'],trainable_parameters=parameters-reference['trainable_parameters'])
    return result

def format_comparison(image, step, count, metrics, delta):
    return ('[{0} step={1} N={2}] PSNR={3:.6f} ({4:+.6f} dB vs fixed p=2) | '
            'SSIM={5:.8f} ({6:+.8f}) | MS-SSIM={7:.8f} ({8:+.8f}) | N delta={9:+d}').format(
        Path(image).name,step,count,metrics['PSNR'],delta['PSNR'],metrics['SSIM'],delta['SSIM'],
        metrics['MS_SSIM'],delta['MS_SSIM'],delta['N'])

def paired_summary(jobs):
    pairs = [j['summary'] for j in jobs if j['status']=='complete' and
             j.get('summary',{}).get('fixed_p2_reference') is not None]
    if not pairs:
        return None
    profiles = {s['metric_profile'] for s in pairs}
    if len(profiles)!=1:
        raise ValueError('Cannot aggregate different metric protocols')
    keys = ('PSNR','SSIM','MS_SSIM')
    learned = {k:sum(s['metrics'][k] for s in pairs)/len(pairs) for k in keys}
    fixed = {k:sum(s['fixed_p2_reference']['metrics'][k] for s in pairs)/len(pairs) for k in keys}
    return dict(completed_pairs=len(pairs),metric_profile=next(iter(profiles)),
        learnable_mean=learned,fixed_p2_mean_on_same_images=fixed,
        delta_vs_fixed_p2={k:learned[k]-fixed[k] for k in keys},
        mean_N=sum(s['actual_final_N'] for s in pairs)/len(pairs),
        fixed_p2_mean_N=sum(s['fixed_p2_reference']['N'] for s in pairs)/len(pairs))
