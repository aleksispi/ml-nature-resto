import datetime
import os
from paths import IMGS

def make_savedir(save_folder,path=IMGS,additional_info=''):
    """
    Creates a directory where images can be saved. It adds data and additional info to distinguish between runs
    """
    date=datetime.datetime.now()
    date=date.strftime("%Y-%m-%d_%H:%M")
    save_dir = path/save_folder/(date +' ' + additional_info)
    os.makedirs(save_dir, exist_ok=True) # Creates folder (and makes sure it doesn't already exist)
    return save_dir     # return ???
