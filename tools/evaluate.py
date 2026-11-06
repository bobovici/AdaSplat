"""Full-resolution RGB8 PSNR/SSIM/MS-SSIM, with no resize or training."""
from pathlib import Path
import argparse,json,math,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',required=True,type=Path);p.add_argument('--reconstruction',required=True,type=Path)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu');p.add_argument('--output',type=Path)
    args=p.parse_args()
    import numpy as np,torch
    from pytorch_msssim import ssim,ms_ssim
    from adasplat.codec.runtime import rgb8,psnr,sha256
    if args.output and args.output.exists():raise FileExistsError('Choose a new output path')
    torch.set_num_threads(2);torch.set_grad_enabled(False)
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    gt=rgb8(args.reference);pred=rgb8(args.reconstruction)
    if pred.shape!=gt.shape or min(gt.shape[:2])<=160:raise ValueError('Images must share dimensions, both >160')
    ss=[];ms=[]
    for c in range(3):
        a=torch.from_numpy(pred[:,:,c].copy())[None,None].to(args.device).float()/255
        b=torch.from_numpy(gt[:,:,c].copy())[None,None].to(args.device).float()/255
        ss.append(float(ssim(a,b,data_range=1,size_average=True)))
        ms.append(float(ms_ssim(a,b,data_range=1,size_average=True)))
    report=dict(PSNR=psnr(pred,gt),SSIM=sum(ss)/3,MS_SSIM=sum(ms)/3,width=gt.shape[1],height=gt.shape[0],
        reference_sha256=sha256(args.reference),prediction_sha256=sha256(args.reconstruction),
        domain='RGB8/255; full resolution; Gaussian11 sigma1.5; five-scale MS-SSIM',device=args.device)
    text=json.dumps(report,indent=2)
    if args.output:args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(text,encoding='utf-8')
    print(text)
if __name__=='__main__':main()
