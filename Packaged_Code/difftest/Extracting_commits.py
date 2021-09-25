import os
import inspect
import pandas as pd
from datetime import date, timedelta
import sys
import subprocess
from IPython.display import display, HTML
from tqdm import tqdm


# Set root folder for all libraries:
dl_library_root = "/Users/Alex/Desktop/BachelorThesis/DLL_Testing_Tool/DL_Libraries/"

# Input 1: Package name
package_name = 'scipy'

# Input 2: Deep Learning Library name and directory
dll_name = 'tensorflow_1.12.0'
dll_directory = dl_library_root + 'Tensorflow/tensorflow-1.12.0/tensorflow/python/'

# Input 3: Current version(i.e. date for simplicity) of the package (and optionally the desired version)
# Format: date(Year, month, day)
current_version_date = date(2018,11,6) # release date of TF 1.12.0
desired_version_date = date(2019,2,25) # release date of TF 1.13.1

# Input 4: Github Link of package (if not stored by the tool)
git_url = 'https://github.com/scipy/scipy.git'
#git_url = 'https://github.com/keras-team/keras.git'


def select_cases_from_package(df, package_name, additional_keywords=[]):
    """additional_keywords can accept multiple keywords to filter the differential test functions for."""
    # for filter keywords we use the '^' regex to mark the start of the string
    searchfor = ['^'+package_name+'\.']
    for keyword in additional_keywords:
        searchfor.append('^'+keyword+'\.')

    column_to_filter = 'Differential_Test_Function'

    # for multiple keywords, join them with the regex "or"
    if len(searchfor) > 1:
        filter_keyword = '|'.join(searchfor)
    else:
        filter_keyword = searchfor[0]

    filtered_df = df[df[column_to_filter].str.contains(filter_keyword, na=False)]

    return filtered_df


def produce_output_doc(filtered_df, current_date, desired_date, package_name, package_git_url, save_to='.'):
    """package_name should be as seen in code, e.g. keras.layers… = ‘keras’
    save_to: Default=current directory
    date format: date(year, month, day) using the datetime.date module
    """
    
    # Import the package that should be upgraded (used to find the files where extracted functions are defined)
    
    # Setup folder name for bare clone
    clone_folder_name = 'temp_bare_clone_' + package_name
    
    # Create a bare clone of the library, which only includes repository data
    # In this way, we do not have to download the code, but still get access to the commit log.
    # create a temporary directory for a bare clone of a give library
    try:
        os.mkdir(clone_folder_name)
    except:
        pass
    
    # Only execute this if the clone was not yet created
    if len(os.listdir(clone_folder_name)) == 0:

        # create the bare clone
        # by executing "git clone --bare {git_url} {clone_folder_name}"
        git_clone_command = ["git", "clone", "--bare", git_url, clone_folder_name]
        git_output = subprocess.run(git_clone_command, stdout=subprocess.PIPE).stdout.decode('utf-8')
        print(git_output)

    # cd to clone directory
    os.chdir(clone_folder_name)
    
    