import sys

class Tee(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
    def flush(self):
        for f in self.files:
            if hasattr(f, 'flush'):
                f.flush()

    def isatty(self):
        # True if any of the files is a terminal
        return any(hasattr(f, 'isatty') and f.isatty() for f in self.files)
    
    def close(self):
        for f in self.files:
            if hasattr(f, 'close'):
                f.close()
    
    def fileno(self):
        for f in self.files:
            if hasattr(f, 'fileno'):
                return f.fileno()
        raise OSError("No underlying file object with fileno")

class Logger:
    def __init__(self, filename, mode='a'):
        self.file_handle = open(filename, mode)
        self.file_handle_stderr = open(filename.replace("run_log.", "run_log_stderr."), mode)
    
    def __enter__(self):
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        sys.stdout = Tee(sys.stdout, self.file_handle)
        sys.stderr = Tee(sys.stderr, self.file_handle_stderr)
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        self.file_handle.close()
        self.file_handle_stderr.close()

    def log(self, *args, **kwargs):
        # original_file = kwargs.pop('file', sys.stdout)
        
        # # Print to both console and file
        # print(*args, **kwargs, file=original_file)

        # print(*args, **kwargs, file=self.file_handle, flush=True)
        print(*args, **kwargs)
        self.file_handle.flush()
        self.file_handle_stderr.flush()
