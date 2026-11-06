"""Atomic output replacement that tolerates brief Windows reader locks."""
import json
import os
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path


def replace_with_retry(source, destination, max_wait=5.0):
    deadline=time.monotonic()+max_wait
    delay=.01
    while True:
        try:
            os.replace(source,destination)
            return
        except OSError as error:
            remaining=deadline-time.monotonic()
            if os.name!='nt' or getattr(error,'winerror',None) not in (5,32,33) or remaining<=0:
                raise
            time.sleep(min(delay,remaining))
            delay=min(.2,delay*2)


@contextmanager
def atomic_output(path, max_wait=5.0):
    path=Path(path)
    fd,name=tempfile.mkstemp(prefix='.'+path.name+'.',suffix=path.suffix,dir=str(path.parent))
    os.close(fd)
    temporary=Path(name)
    try:
        yield temporary
        replace_with_retry(temporary,path,max_wait=max_wait)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def atomic_json(path, data, max_wait=5.0):
    with atomic_output(path,max_wait=max_wait) as temporary:
        temporary.write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding='utf8')
