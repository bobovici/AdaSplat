"""Compile the included coordinate-precision CUDA sources for this machine."""
from pathlib import Path
import argparse,hashlib,json,os,shutil,sys,time
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jobs',type=int,default=2)
    parser.add_argument('--arch',help='CUDA architecture(s), e.g. 8.6 or 8.9; defaults to visible GPUs')
    parser.add_argument('--check',action='store_true',help='Check source/toolchain without compiling')
    args=parser.parse_args()
    if args.jobs<1:parser.error('jobs must be positive')
    import torch
    from torch.utils.cpp_extension import CUDA_HOME,load
    source=ROOT/'src/adasplat/core/gsplat/gsplat/cuda/csrc'
    sources=sorted(p for p in source.iterdir() if p.suffix in ('.cu','.cpp'))
    for name in ['forward.cu','backward.cu']:
        if (source/name).read_text(encoding='utf-8').count('const double dx64 = (double)center.x - (double)px;')!=1:
            raise ValueError('Coordinate-precision source check failed: '+name)
    if not CUDA_HOME or not torch.cuda.is_available():raise RuntimeError('CUDA PyTorch, CUDA GPU and nvcc are required')
    arch=args.arch or ';'.join(sorted({'.'.join(map(str,torch.cuda.get_device_capability(i))) for i in range(torch.cuda.device_count())}))
    report=dict(torch=str(torch.__version__),cuda=torch.version.cuda,CUDA_HOME=str(CUDA_HOME),arch=arch,
        source_sha256={p.name:sha(p) for p in sources},python=sys.version)
    if args.check:print(json.dumps(report,indent=2));return
    os.environ['MAX_JOBS']=str(args.jobs);os.environ['TORCH_CUDA_ARCH_LIST']=arch
    build=ROOT/'native/build';build.mkdir(parents=True,exist_ok=True)
    win=os.name=='nt'
    module=load(name='csrc',sources=[str(p) for p in sources],
        extra_include_paths=[str(source/'third_party/glm')],
        extra_cflags=['/O2','/std:c++17'] if win else ['-O3','-std=c++17'],
        extra_cuda_cflags=['-O3','--use_fast_math','--expt-relaxed-constexpr','-std=c++17'],
        with_cuda=True,build_directory=str(build),verbose=True)
    artifact=Path(module.__file__);binary=ROOT/('native/csrc.pyd' if win else 'native/csrc.so')
    if binary.exists():
        backup=ROOT/'native/backups'/time.strftime('%Y%m%d_%H%M%S');backup.mkdir(parents=True)
        shutil.copy2(binary,backup/binary.name)
        if (ROOT/'native/renderer.json').exists():shutil.copy2(ROOT/'native/renderer.json',backup/'renderer.json')
    if artifact.resolve()!=binary.resolve():shutil.copy2(artifact,binary)
    report.update(binary=binary.relative_to(ROOT).as_posix(),sha256=sha(binary))
    (ROOT/'native/renderer.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
