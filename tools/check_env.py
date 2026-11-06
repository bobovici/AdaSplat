"""Print environment details without changing packages or compiling CUDA."""
import importlib.metadata,json,platform,shutil,subprocess,sys
def main():
    result=dict(python=sys.version,executable=sys.executable,platform=platform.platform(),packages={})
    for name in ['torch','numpy','Pillow','pytorch-msssim','constriction','ninja']:
        try:result['packages'][name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:result['packages'][name]=None
    try:
        import torch
        from torch.utils.cpp_extension import CUDA_HOME
        result.update(torch_cuda=torch.version.cuda,cuda_available=torch.cuda.is_available(),CUDA_HOME=str(CUDA_HOME),
            GPUs=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
    except ImportError:pass
    for program in ['nvcc','g++','cl']:
        result[program]=shutil.which(program)
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
