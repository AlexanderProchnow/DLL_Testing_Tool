from diffwatch import debug_single_file_extraction, extract_diff_test_cases

# Set library root folder:
library_root = "A:/BachelorThesis/DLL_Testing_Tool/DL_Libraries/Tensorflow/tensorflow-master/tensorflow/python/"

file_path = library_root + "kernel_tests/distributions/gamma_test.py"
debug_single_file_extraction(file_path)

extract_diff_test_cases(library_root)

