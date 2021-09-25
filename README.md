# DLL_Testing_Tool
A tool designed to aid the developers of Deep Learning Libraries to manage their differential testing by automatically cataloging differential test cases.
The tool additionally allows developers to monitor changes in external libraries that are relevant to their differential testing.

# Overview of the repository structure
The Code folder contains the Juypter Notebooks alongside the extracted data and tool output documents.  

The Packaged_Code folder contains the Python package of our tool. The code found here is a packaged version of the code found within the notebooks.
The functions of this Python package can be accessed for example via  `from difftest import extract_diff_test_cases`. 
An overview of the available packaged functions can be found in the `__init__.py`
