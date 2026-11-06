"""Reproduce actual Windows delete-sharing failures and verify recovery."""
import ctypes
import errno
import json
import os
import sys
import threading
from pathlib import Path
from ctypes import wintypes
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from atomic_files import atomic_json,atomic_output

def reader_handle(path,share_delete=False):
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.CreateFileW.argtypes=[wintypes.LPCWSTR,wintypes.DWORD,wintypes.DWORD,
        wintypes.LPVOID,wintypes.DWORD,wintypes.DWORD,wintypes.HANDLE]
    kernel.CreateFileW.restype=wintypes.HANDLE
    kernel.CloseHandle.argtypes=[wintypes.HANDLE]
    kernel.CloseHandle.restype=wintypes.BOOL
    handle=kernel.CreateFileW(str(path),0x80000000,3|(4 if share_delete else 0),None,3,128,None)
    if handle==wintypes.HANDLE(-1).value:raise ctypes.WinError(ctypes.get_last_error())
    return kernel,handle

@pytest.mark.skipif(os.name!='nt',reason='Windows sharing semantics')
def test_transient_reader_lock_is_retried_without_losing_json(tmp_path):
    path=tmp_path/'result.json';atomic_json(path,dict(step=10000))
    kernel,handle=reader_handle(path)
    probe=tmp_path/'probe.json';probe.write_text('{}',encoding='utf8')
    try:
        with pytest.raises(PermissionError) as caught:os.replace(probe,path)
        assert caught.value.winerror in (5,32,33)
        release=threading.Timer(.15,kernel.CloseHandle,args=[handle]);release.start()
        try:atomic_json(path,dict(step=11000),max_wait=2.)
        finally:release.join()
    except BaseException:
        kernel.CloseHandle(handle)
        raise
    assert json.loads(path.read_text())==dict(step=11000)

@pytest.mark.skipif(os.name!='nt',reason='Windows sharing semantics')
def test_shared_delete_reader_keeps_complete_old_snapshot_during_replacement(tmp_path):
    path=tmp_path/'result.json';atomic_json(path,dict(step=10000))
    kernel,handle=reader_handle(path,share_delete=True)
    errors=[]
    def replace():
        try:atomic_json(path,dict(step=11000),max_wait=2.)
        except BaseException as error:errors.append(error)
    writer=threading.Thread(target=replace);writer.start()
    try:
        kernel.ReadFile.argtypes=[wintypes.HANDLE,wintypes.LPVOID,wintypes.DWORD,
            ctypes.POINTER(wintypes.DWORD),wintypes.LPVOID]
        kernel.ReadFile.restype=wintypes.BOOL
        buffer=ctypes.create_string_buffer(1024);count=wintypes.DWORD()
        assert kernel.ReadFile(handle,buffer,1024,ctypes.byref(count),None)
        assert json.loads(buffer.raw[:count.value].decode('utf8'))==dict(step=10000)
    finally:kernel.CloseHandle(handle)
    writer.join(timeout=3.)
    assert not writer.is_alive() and not errors
    assert json.loads(path.read_text())==dict(step=11000)

@pytest.mark.skipif(os.name!='nt',reason='Windows sharing semantics')
def test_persistent_lock_fails_boundedly_and_preserves_original(tmp_path):
    path=tmp_path/'result.json';atomic_json(path,dict(step=10000))
    kernel,handle=reader_handle(path)
    try:
        with pytest.raises(PermissionError):atomic_json(path,dict(step=11000),max_wait=.05)
        assert json.loads(path.read_text())==dict(step=10000)
        assert sorted(p.name for p in tmp_path.iterdir())==['result.json']
    finally:kernel.CloseHandle(handle)

def test_non_sharing_error_is_not_hidden_by_retries(tmp_path,monkeypatch):
    path=tmp_path/'result.json';atomic_json(path,dict(step=10000));calls=[]
    def no_space(*args):
        calls.append(args);raise OSError(errno.ENOSPC,'disk full')
    monkeypatch.setattr(os,'replace',no_space)
    with pytest.raises(OSError) as caught:atomic_json(path,dict(step=11000))
    assert caught.value.errno==errno.ENOSPC and len(calls)==1
    assert json.loads(path.read_text())==dict(step=10000)

def test_failed_serialization_preserves_target_and_cleans_temporary(tmp_path):
    path=tmp_path/'result.json';atomic_json(path,dict(step=10000))
    with pytest.raises(TypeError):atomic_json(path,dict(value=object()))
    assert json.loads(path.read_text())==dict(step=10000)
    assert sorted(p.name for p in tmp_path.iterdir())==['result.json']

def test_atomic_output_handles_pytorch_checkpoint(tmp_path):
    import torch
    path=tmp_path/'resume.pt'
    with atomic_output(path) as temporary:torch.save(dict(step=11000,weights=torch.arange(4)),temporary)
    result=torch.load(path,weights_only=False)
    assert result['step']==11000 and torch.equal(result['weights'],torch.arange(4))
