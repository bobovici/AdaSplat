"""Portable encode, decode, inspect and render commands for complete .adagc files."""
from pathlib import Path
import argparse,json,os,shutil,sys,tempfile,time

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    subs=parser.add_subparsers(dest='command',required=True)
    enc=subs.add_parser('encode',help='Encode an existing final/milestone checkpoint; no training')
    enc.add_argument('--checkpoint',required=True,type=Path)
    enc.add_argument('--output',required=True,type=Path)
    enc.add_argument('--target',type=Path,help='Original RGB image for decoded quality guard')
    enc.add_argument('--max-loss-db',type=float,default=.02)
    enc.add_argument('--no-quality-guard',action='store_true',help='Fixed first setting; does not promise decoded quality')
    enc.add_argument('--threads',type=int,default=2)
    enc.add_argument('--gpu',type=int,default=None)
    enc.add_argument('--renderer',type=Path)
    enc.add_argument('--allow-renderer-change',action='store_true')
    enc.add_argument('--decoded-output',type=Path)
    dec=subs.add_parser('decode',help='CPU-only decode to physical-field NPZ; no target/checkpoint needed')
    dec.add_argument('--input',required=True,type=Path);dec.add_argument('--output',required=True,type=Path)
    ins=subs.add_parser('inspect',help='Inspect complete-stream header and actual bpp')
    ins.add_argument('--input',required=True,type=Path)
    ren=subs.add_parser('render',help='Render a .adagc stream or an inference .pt checkpoint')
    ren.add_argument('--input',required=True,type=Path);ren.add_argument('--output',required=True,type=Path)
    ren.add_argument('--renderer',type=Path);ren.add_argument('--allow-renderer-change',action='store_true')
    ren.add_argument('--gpu',type=int,default=None)
    args=parser.parse_args(argv)
    try:
        if getattr(args,'gpu',None) is not None:
            if args.gpu<0:raise ValueError('gpu must be nonnegative')
            os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu)
        if hasattr(args,'output') and args.output.exists():raise FileExistsError('Choose a new output file: '+str(args.output))
        from .runtime import decode_stream,sha256
        if args.command=='inspect':
            from .algorithms import read_stream
            h,_=read_stream(args.input)
            print(json.dumps(dict(header=h,bytes=args.input.stat().st_size,
                bpp=8*args.input.stat().st_size/(h['width']*h['height']),sha256=sha256(args.input)),indent=2))
            return 0
        if args.command=='decode':
            import numpy as np
            h,fields=decode_stream(args.input);args.output.parent.mkdir(parents=True,exist_ok=True)
            from adasplat.paths import ROOT
            sys.path.insert(0,str(ROOT/'tools'))
            from atomic_files import atomic_output
            with atomic_output(args.output) as temporary:
                with temporary.open('wb') as f:np.savez(f,**fields,header_json=np.array(json.dumps(h)))
            print(json.dumps(dict(N=h['N'],output=str(args.output),CPU_only=True,source_stream_sha256=sha256(args.input))))
            return 0
        if args.command=='render':return render(args)
        return encode(args)
    except (ValueError,FileNotFoundError,FileExistsError,RuntimeError) as error:
        parser.exit(1,'error: '+str(error)+'\n')

def save_image(array,path):
    from PIL import Image
    path.parent.mkdir(parents=True,exist_ok=True)
    Image.fromarray(array).save(path)

def render(args):
    from .runtime import decode_stream,header_for,load_checkpoint,physical_fields,render_fields,select_renderer,sha256
    from .algorithms import MAGIC
    with args.input.open('rb') as f:is_stream=f.read(len(MAGIC))==MAGIC
    if is_stream:
        h,fields=decode_stream(args.input);expected=None
    else:
        payload=load_checkpoint(args.input);h=header_for(payload)
        fields=physical_fields(payload);expected=payload['renderer']['sha256']
    runtime=select_renderer(args.renderer,expected,args.allow_renderer_change)
    image=render_fields(h,fields);save_image(image,args.output)
    print(json.dumps(dict(N=h['N'],width=h['width'],height=h['height'],output=str(args.output),
        source_sha256=sha256(args.input),renderer=runtime,training_updates=0),indent=2))
    return 0

def encode(args):
    import gc,numpy as np,torch
    from .encoder import encode as encode_fields
    from .runtime import decode_stream,header_for,load_checkpoint,physical_fields,render_fields,select_renderer,rgb8,psnr,sha256
    if args.threads<1 or not np.isfinite(args.max_loss_db) or args.max_loss_db<0:
        raise ValueError('threads must be positive; max-loss-db must be finite and nonnegative')
    if not args.no_quality_guard and args.target is None:
        raise ValueError('Provide --target for the 0.02-dB decoded quality guard, or explicitly use --no-quality-guard')
    metrics_path=args.output.with_suffix(args.output.suffix+'.json')
    if metrics_path.exists() or (args.decoded_output and args.decoded_output.exists()):
        raise FileExistsError('Output sidecar/image already exists')
    torch.set_num_threads(args.threads);torch.set_grad_enabled(False)
    start=time.monotonic();payload=load_checkpoint(args.checkpoint);fields=physical_fields(payload)
    h=header_for(payload);digest=sha256(args.checkpoint);runtime=None;target=None;reference_psnr=None
    if args.target:
        target=rgb8(args.target)
        if target.shape!=(h['height'],h['width'],3):raise ValueError('Target dimensions do not match the checkpoint')
    if not args.no_quality_guard or args.decoded_output:
        runtime=select_renderer(args.renderer,payload['renderer']['sha256'],args.allow_renderer_change)
        if target is not None:
            reference=render_fields(h,fields);reference_psnr=psnr(reference,target);del reference
    settings=[(320,16)] if args.no_quality_guard else [(320,16),(1280,16),(4096,16),(320,24),(1280,24)]
    args.output.parent.mkdir(parents=True,exist_ok=True);attempts=[];accepted=False;decoded_image=None
    with tempfile.TemporaryDirectory(prefix='adasplat-codec-',dir=str(args.output.parent)) as temp:
        seen=set()
        for requested_blocks,bits in settings:
            blocks=min(requested_blocks,int(payload['N'])-1)
            if (blocks,bits) in seen:continue
            seen.add((blocks,bits))
            candidate=Path(temp)/('rgb%d_b%d.adagc'%(bits,blocks))
            encode_fields(fields,payload,candidate,digest,blocks,bits)
            decoded_header,decoded=decode_stream(candidate)
            if decoded_header['N']!=payload['N'] or decoded_header['final_artifact_sha256']!=digest:
                raise ValueError('Decoded stream identity mismatch')
            quality=None;loss=None
            if runtime is not None:
                decoded_image=render_fields(decoded_header,decoded)
                if target is not None:
                    quality=psnr(decoded_image,target);loss=reference_psnr-quality
            accepted=args.no_quality_guard or (loss is not None and loss<=args.max_loss_db)
            attempts.append(dict(RGB_blocks=blocks,RGB_bits=bits,bytes=candidate.stat().st_size,
                decoded_PSNR=quality,loss_db=loss,accepted=accepted))
            print(json.dumps(dict(event='codec_attempt',**attempts[-1])),flush=True)
            del decoded;gc.collect()
            if torch.cuda.is_available():torch.cuda.empty_cache()
            if accepted:
                # The stream is a complete independently decodable file; no external tensor payloads.
                shutil.copyfile(candidate,args.output)
                break
        if not accepted:
            failure=args.output.with_suffix(args.output.suffix+'.failed.json')
            failure.write_text(json.dumps(dict(status='quality_guard_failed',attempts=attempts),indent=2),encoding='utf-8')
            raise RuntimeError('Quality guard exhausted; no output stream published. See '+str(failure))
    if args.decoded_output:
        if decoded_image is None:
            decoded_header,decoded=decode_stream(args.output)
            runtime=select_renderer(args.renderer,payload['renderer']['sha256'],args.allow_renderer_change)
            decoded_image=render_fields(decoded_header,decoded)
        save_image(decoded_image,args.decoded_output)
    report=dict(status='complete',codec='V6/V2 logscale RAHT/Laplace',N=h['N'],step=payload.get('step'),
        width=h['width'],height=h['height'],bytes=args.output.stat().st_size,
        bpp=8*args.output.stat().st_size/(h['width']*h['height']),stream_sha256=sha256(args.output),
        checkpoint_sha256=digest,target_sha256=sha256(args.target) if args.target else None,
        original_RGB8_PSNR=reference_psnr,decoded_RGB8_PSNR=attempts[-1]['decoded_PSNR'],
        loss_db=attempts[-1]['loss_db'],quality_guard_enabled=not args.no_quality_guard,
        max_loss_db=args.max_loss_db if not args.no_quality_guard else None,
        attempts=attempts,renderer=runtime,seconds=time.monotonic()-start,
        training_updates=0,pruned_primitives=0)
    metrics_path.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)
    return 0
