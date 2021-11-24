from diffwatch import select_cases_from_package, produce_output_doc
import pandas as pd
from datetime import date
import scipy
from scipy import *

# Input 1: Package name
package_name = 'scipy'

# Input 2: Deep Learning Library name and directory
dll_name = 'tensorflow'

# Input 3: Current version (i.e. date for simplicity) of the package
current_version_date = date(2018,1,1)
desired_version_date = date.today()

# Input 4: Github Link of package
git_url = 'https://github.com/scipy/scipy.git'

# Import catalog of test cases
df = pd.read_csv('extracted_data/tensorflow_data.csv')

filtered_df = select_cases_from_package(df, package_name)

produce_output_doc(filtered_df, current_version_date, desired_version_date,
                   package_name, git_url)
