"""Durable per-run events and a terminal/file tee, including native fd 1/2 output."""
import io
import ctypes
import json
import os
import sys
import threading
import traceback
from datetime import datetime,timezone
from pathlib import Path


class RunLogger:
    def __init__(self,directory):
        self.directory = Path(directory)
        self.sequence = 0
        self.reader_error = None

    def record(self,event,**data):
        self.sequence += 1
        row = dict(sequence=self.sequence,time_utc=datetime.now(timezone.utc).isoformat(),
                   pid=os.getpid(),event=event,**data)
        self.events.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
        self.events.flush()

    @staticmethod
    def _write_all(fd,data):
        while data:
            written = os.write(fd,data)
            data = data[written:]

    def _read_console(self):
        try:
            while True:
                chunk = os.read(self.read_fd,65536)
                if not chunk:
                    break
                try:
                    self.console.write(chunk)
                    self.console.flush()
                except OSError as error:
                    self.reader_error = error  # Keep draining; do not deadlock writers.
                try:
                    self._write_all(self.saved_stdout_fd,chunk)
                except OSError:
                    pass  # A detached terminal must not discard the saved log.
        except OSError as error:
            self.reader_error = error
        finally:
            os.close(self.read_fd)

    @staticmethod
    def _flush_native():
        # Flush buffered C stdio while descriptors still point at the log pipe.
        library = ctypes.CDLL('ucrtbase') if os.name=='nt' else ctypes.CDLL(None)
        library.fflush(None)

    def __enter__(self):
        # The caller owns a newly created run directory. Never truncate prior logs.
        self.console = (self.directory/'console.log').open('xb',buffering=0)
        try:
            self.events = (self.directory/'events.jsonl').open('x',encoding='utf-8',buffering=1)
        except BaseException:
            self.console.close()
            raise
        self.original_stdout,self.original_stderr = sys.stdout,sys.stderr
        self.original_stdout.flush();self.original_stderr.flush()
        self._flush_native()
        self.saved_stdout_fd,self.saved_stderr_fd = os.dup(1),os.dup(2)
        self.read_fd,write_fd = os.pipe()
        self.reader = threading.Thread(target=self._read_console,name='adasplat-console-log',daemon=True)
        self.reader.start()
        os.dup2(write_fd,1);os.dup2(write_fd,2);os.close(write_fd)
        # Explicit streams also capture Python output under notebook/pytest stream adapters.
        sys.stdout = io.TextIOWrapper(os.fdopen(os.dup(1),'wb',buffering=0),encoding='utf-8',
                                      errors='backslashreplace',write_through=True)
        sys.stderr = io.TextIOWrapper(os.fdopen(os.dup(2),'wb',buffering=0),encoding='utf-8',
                                      errors='backslashreplace',write_through=True)
        return self

    def __exit__(self,kind,error,tb):
        try:
            if error is not None:
                detail = ''.join(traceback.format_exception(kind,error,tb))
                self.record('exception',exception=kind.__name__,error=str(error),traceback=detail)
                sys.stderr.write(detail)
            self.record('log_closed',failed=error is not None)
        finally:
            sys.stdout.flush();sys.stderr.flush()
            self._flush_native()
            sys.stdout.close();sys.stderr.close()
            sys.stdout,sys.stderr = self.original_stdout,self.original_stderr
            os.dup2(self.saved_stdout_fd,1);os.dup2(self.saved_stderr_fd,2)
            self.reader.join()  # EOF after all writer descriptors close; drain the final traceback.
            os.close(self.saved_stdout_fd);os.close(self.saved_stderr_fd)
            self.console.close();self.events.close()
        if self.reader_error is not None and error is None:
            raise OSError('Could not save complete console log: '+str(self.reader_error))
        return False
