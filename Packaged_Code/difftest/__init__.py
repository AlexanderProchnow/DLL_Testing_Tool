# Global variables to control the extent of the logged information (for debugging)
LOG_ALL = 0
LOG_FINAL = 1
LOG_NONE = 2


# IDENTIFYING DIFFERENTIAL TEST CASES:

# debug_single_file_extraction:
#   Inputs: 
#       file_path
#       Optional parameter: print_mode: default=LOG_FINAL
#       Optional parameter: log_mode: default=LOG_ALL
#   Returns df

# extract_diff_test_cases:
#   Inputs:
#       library_root
#       Optional parameter: save_data_to: default=current dir/extracted_data, None=Don’t save
#   Returns df


from .Finding_differential_test_cases import debug_single_file_extraction, extract_diff_test_cases




# DATA EVALUATION:

# coverage_analysis:
#   Inputs:
#       df
#   Prints unsupported and nan counts etc.

from .Data_evaluation import coverage_analysis


# EXTRACTING COMMITS AND OUTPUT DOCUMENT GENERATION:

# select_cases_from_package
#   Inputs:
#       df
#       package name, e.g. ‘keras’
#   Returns filtered_df

# produce_output_doc
#   Inputs:
#       filtered_df
#       current_date
#       desired_date
#       package_name (comment: should be as seen in code, e.g. keras.layers… = ‘keras’)
#       package_git_url
#       Optional parameter: save_to: Default=current dir
#   Produces html document and stores it

from .Extracting_commits import select_cases_from_package, produce_output_doc