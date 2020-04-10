import os

def list_files(dir, ext):
    '''List files by extension'''
    return (f for f in os.listdir(dir) if f.endswith('.' + ext))
