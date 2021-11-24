import os
import inspect
import pandas as pd
from datetime import date, timedelta
import sys
import subprocess
from IPython.display import display, HTML
from tqdm import tqdm
import scipy
from scipy import *

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





def get_function_file_location(extracted_function, _package_name='tensorflow'):
    """For step 1. Find where the function is defined."""
    
    # use the extracted_function string as if it were code, since 'inspect' can't deal with strings
    str_to_execute = 'extracted_function_file_location = inspect.getsourcefile({})'.format(extracted_function)
    
    # get local scope (necessary since exec does not work well inside of function definition scopes)
    lcls = locals()
    
    # execute the string as if it were code, setting the file location variable in the local scope
    exec(str_to_execute, globals(), lcls)
    
    # getting the variable from the local scope
    extracted_function_file_location = lcls["extracted_function_file_location"]
    
    #print(extracted_function_file_location)
    
    # remove the package root to get the relative file path 
    package_root_index = extracted_function_file_location.index(_package_name)
    extracted_function_file_location = extracted_function_file_location[package_root_index:]
    
    return extracted_function_file_location


def get_nearest_commit(version_date):
    """For step 2. Return commit ID and message of the nearest commit on or before version_date."""
    git_log_output = ''
    days = 1
    while git_log_output == '':
        git_log_command = ["git", "log", "--since", (version_date-timedelta(days=days)).strftime("%m-%d-%Y"), "--until", version_date.strftime("%d-%m-%Y")]
        #, "--", extracted_function_file_location]
        git_log_output = subprocess.run(git_log_command, stdout=subprocess.PIPE).stdout.decode('utf-8')
        
        #print("-" + str(days) + " " + git_log_output)
        
        days += 1
        
        # exit condition for when search takes too long
        if days > 100:
            return 'ERROR', 'No commit within 100 days of the entered date.', version_date
            

    commit_id = git_log_output[7:].splitlines()[0]
    
    commit_message_command = ["git", "log", "--format=%B", "-n", "1", commit_id]
    commit_message = subprocess.run(commit_message_command, stdout=subprocess.PIPE).stdout.decode('utf-8')
    
    commit_date = version_date-timedelta(days=days-2)
    
    return commit_id, commit_message, commit_date


def format_line_beginning(line):
    line_beginning = []
    for char in line:
        if char == ' ':
            line_beginning.append('&nbsp')
        else:
            break

    separator = ' '
    formatted_line = separator.join(line_beginning)
    formatted_line += line.lstrip()
    
    return formatted_line


def get_git_diff_output_formatted(commit_id_current, commit_id_desired, extracted_function_file_location):
    git_diff_command = ["git", "diff", commit_id_current, commit_id_desired, "--", extracted_function_file_location]

    git_diff_output = subprocess.run(git_diff_command, stdout=subprocess.PIPE).stdout.decode('utf-8')
    
    git_diff_processed = ''
    for line in git_diff_output.splitlines():
        if line.startswith('-'):
            line = line[1:]
            git_diff_processed += "<span style=\"color:red\">- " + format_line_beginning(line) + "</span>\n"
        
        elif line.startswith('+'):
            line = line[1:]
            git_diff_processed += "<span style=\"color:green\">+" + format_line_beginning(line) + "</span>\n"
        
        elif line.startswith(' '):
            git_diff_processed += format_line_beginning(line) + "\n"
            
        else:
            git_diff_processed += line + "\n"
    
    # formatting for html
    git_diff_processed = git_diff_processed.replace('\n', '\n<br>')#.replace(' ', '&nbsp ')
    
    return git_diff_processed


def produce_output_doc(filtered_df, current_date, desired_date, package_name, package_git_url, doc_name="tool_output"):
    """package_name should be as seen in code, e.g. keras.layers… = ‘keras’
    doc_name: Default=tool_output_<package_name>
    date format: date(year, month, day) using the datetime.date module
    """

    doc_name = "tool_output_{}".format(package_name)
    
    
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

    # the following line is to correctly remove the user specific part of the file path,
    # e.g. /Users/Alex/Desktop etc. from the extracted functions source file location
    # for testing tf.keras in tensorflow 1.12.0 this should be changed to 'tensorflow'
    # and for np this should be changed to 'numpy'
    package_name_in_root = package_name
    #package_name_in_root = 'tensorflow'
    #package_name_in_root = 'numpy'


    tool_output_destination = "../{}.html".format(doc_name)
    tool_output = open(tool_output_destination, "w+", encoding='utf-8')
    tool_output.write("""
        <!DOCTYPE html>
        <html>
        <head>
        <style>
        .collapsible {
          background-color: #777;
          color: white;
          cursor: pointer;
          padding: 18px;
          width: 100%;
          border: none;
          text-align: left;
          outline: none;
          font-size: 15px;
        }

        .active, .collapsible:hover {
          background-color: #555;
        }

        .content {
          padding: 0 18px;
          display: none;
          overflow: hidden;
          background-color: #f1f1f1;
        }
        </style>
        </head>
        <body>\n
    """)

    error_list = []
    extr_func_file_location_list = []

    for extracted_function in filtered_df.Differential_Test_Function:

        # 1:   
        try:
            extracted_function_file_location = get_function_file_location(extracted_function, _package_name=package_name_in_root)
            #print(extracted_function_file_location) # useful for debugging
        except Exception as exc:
            error_list.append(extracted_function + " : " + str(exc))
            extr_func_file_location_list.append("ERROR " + str(exc))
            continue
        
        extr_func_file_location_list.append(extracted_function_file_location)
        
    filtered_df.loc[:, 'Extracted_Function_File_Location'] = extr_func_file_location_list
    

    # 2:
    commit_id_current, commit_message_current, commit_date_current = get_nearest_commit(current_date)
    tool_output.write("\n <br>Commit id closest to current version: " + commit_id_current + "\n<br>Date: " + commit_date_current.strftime("%d-%b-%Y") + "\n")
    tool_output.write("\n <br>Commit message: " + commit_message_current.replace('\n', '<br>') + "\n")

    commit_id_desired, commit_message_desired, commit_date_desired = get_nearest_commit(desired_date)
    tool_output.write("<br>Commit id closest to desired version: " + commit_id_desired + "\n<br>Date: " + commit_date_desired.strftime("%d-%b-%Y") + "\n")
    tool_output.write("\n <br>Commit message: " + commit_message_desired.replace('\n', '<br>') + "\n<br>")


    for extracted_function_file_location in tqdm(filtered_df.Extracted_Function_File_Location.unique()):
        
        tool_output.write("_____________________________________" + extracted_function_file_location + "_________________________________________\n")
        
        tool_output.write(filtered_df[filtered_df['Extracted_Function_File_Location'] == extracted_function_file_location].to_html())
        tool_output.write("\n<br>")
        
        
        # 3:
        git_diff_processed = get_git_diff_output_formatted(commit_id_current, commit_id_desired, extracted_function_file_location)
        
        # (optional) also include the git diff of another file, e.g. the one that the test case was found in:
        #git_diff_processed += "\n<br>" + get_git_diff_output_formatted(commit_id_current, commit_id_desired, 'tensorflow/python/kernel_tests/rnn_test.py')
        
        # add git diff as collapsible section
        tool_output.write("<button type=\"button\" class=\"collapsible\">Git Diff</button>\n<div class=\"content\">\n<p>" + git_diff_processed + "</p>\n</div>\n<br><br><br>")

    # Add script to html to make git diff collapsible
    tool_output.write("""
    <br>
    <script>
    var coll = document.getElementsByClassName("collapsible");
    var i;

    for (i = 0; i < coll.length; i++) {
      coll[i].addEventListener("click", function() {
        this.classList.toggle("active");
        var content = this.nextElementSibling;
        if (content.style.display === "block") {
          content.style.display = "none";
        } else {
          content.style.display = "block";
        }
      });
    }
    </script>
    </body>
    </html>""")
    tool_output.close()
    print(str(len(error_list)) + " errors: " + str(error_list))
    print("Tool output saved to " + tool_output_destination)
        
    
