from difftest import LOG_ALL, LOG_FINAL, LOG_NONE
import ast
import sys
import numpy as np
import os
from datetime import datetime
from tqdm import tqdm
import copy
import pandas as pd

library_root = "A:/BachelorThesis/DLL_Testing_Tool/DL_Libraries/Tensorflow/tensorflow-1.12.0/tensorflow/python/" 
save_data_to = "extracted_data/tensorflow_1.12.0__TESTING_data.csv"

#source = None

#tree = None

log = None


#def set_locations(library_loc, save_data_to_loc):
#    global library_root
#    library_root = library_loc
    
#    global save_data_to
#    save_data_to = save_data_to_loc

    



# Define which assert functions to look for
approximation_asserts = ["assertAlmostEqual", "assertAlmostEquals", "assertAllClose", "assertAllLessEqual",
                         "assertAllCloseAccordingToType", "assertArrayNear", "assert_list_pairwise",
                         "assertNear", "assertLess", "assertAllLess", "assertLessEqual", 
                         "assertNDArrayNear", "assert_allclose", "assert_array_almost_equal",
                         "assert_almost_equal","assert_array_less", 
                         "isclose", "allclose", "gradcheck", "gradgradcheck"]

bool_asserts = ["assertTrue", "assertFalse", "assertIs", "assertIsNot"]

other_asserts = ["assertAllEqual","assertEquals", "assertEqual", "assertAllGreater",
                                                          "assertAllGreaterEqual", "assertAllInRange", "assertAllInSet",
                                                          "assertCountEqual", "assertDTypeEqual", "assertDictEqual",
                                                          "assertSequenceEqual", "assertShapeEqual",
                                                          "assertTupleEqual", "assert_array_equal"]

numpy_asserts = ["assert_raises", "assert_equal", "assert_",
    "assert_warns", "assert_raises_regex"]

all_asserts = approximation_asserts + bool_asserts + other_asserts + numpy_asserts


class CustomLogger():
    def __init__(self, print_mode=LOG_FINAL, log_mode=LOG_ALL):
        """Open Log and set print mode. LOG_FINAL prints only output for users. Other log modes are for debugging."""
        self.print_mode = print_mode
        self.log_mode = log_mode
        
        # for creating a csv with the collected data
        self.file_path = ""
        #self.data = np.array(["File Path", "Line Number", "Found in Function", "Function Definition Line Number" "Assert Statement Type", "Oracle Argument Position", "Differential Function Line Number", "Differential Test Function"])
        self.data = np.array(["File_Path", "Line_Number", "Found_in_Function", "Function_Definition_Line_Number", "Assert_Statement_Type", "Oracle_Argument_ Position", "Differential_Function_Line_Number", "Differential_Test_Function"])
    
        # create a .txt log file
        if not os.path.exists("extraction_logs"):
            os.mkdir("extraction_logs")
        logfile_name = "extraction_logs/" + "log_" + datetime.now().strftime("%d_%b_%Y__%H_%M") + ".txt"
        # open the log file in append mode, create it if it does not exist
        self.log = open(logfile_name, 'a+')
        
    def add(self, string, mode=LOG_ALL, end='\n'):
        """Add text to the log and print if it matches the chosen print_mode"""
        #TODO Write log to txt file (for the full tool later) 
        
        # print the string if print mode matches
        if mode >= self.print_mode:
            print(string, end=end)
            
        if mode >= self.log_mode:
            self.log.write(string + end)
    
    def createEntry(self, line_no, found_in_function, function_def_line_no, assert_statement_type, oracle_arg_pos, diff_func_line_no, diff_test_function_name):
        """Add an entry to the data."""
        self.data = np.vstack((self.data, [self.file_path, line_no, found_in_function, function_def_line_no, assert_statement_type, oracle_arg_pos, diff_func_line_no, diff_test_function_name]))
    
    def set_file_path_variable(self, file_path):
        """Set the file path variable for upcoming data entries."""
        self.file_path = file_path
        
    def get_data(self):
        return self.data
    
    def save_data_to_csv(self, path="extracted_data/data.csv"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savetxt(path, self.data, fmt='%s', delimiter=",")
        print("Data saved to " + path)
        # close log
        self.log.close()


class TreeTraverser(ast.NodeVisitor):
    def __init__(self, tree):
        
        # store the parsed source file
        self.tree = tree
        
        # always stores the last function definition node that was visited along with its arguments
        self.last_function_definition_node = None
        self.last_func_arg_list = []
        
        # variables to ignore when recursively tracing back variable definitions 
        self.ignore_list = []
        # record all nodes visited to avoid recursion loops
        self.nodes_visited = []
        
    def visit_Call(self, node):
        
        # (1) Identify if it is an assert call that we are looking for:    
        assert_statement = ""
        
        # for nodes that call a function of an object, e.g. self.assert(...)
        if isinstance(node.func, ast.Attribute):
            assert_statement = node.func.attr
            
        # or directly, e.g. assert_(...)    
        elif isinstance(node.func, ast.Name):
            assert_statement = node.func.id
            
            
        # if the name of the called function is one we are searching for
        if assert_statement in all_asserts:

            # print the assert function name and line number
            log.add("______________" + assert_statement + "__(line " + str(node.lineno) + ")______________", mode=LOG_FINAL)

            # print the node and its structure
            log.add(ast.dump(node, indent='\t') + '\n')
            log.add("Found in: " + self.last_function_definition_node.name)

            # (2) Identify if the assert call uses a differential test oracle:

            # Go through arguments of the assert call (only the first two, other arguments are tolerances)
            for argument_position, assert_argument in enumerate(node.args[:2]):

                # have argument positions as position 1, position 2 etc.
                _argument_position = argument_position+1

                # reflect in the log which argument is currently being looked at
                log.add("\nArgument " + str(_argument_position) + ": ")

                # return the definitions of the arguments to check if they are differential testing functions
                arg_definitions = self._getDefinition(assert_argument)

                # check which node is the oracle
                if arg_definitions is not None:
                    for definition in arg_definitions:
                        # print the ast structure of the definitions
                        log.add("\n(line " + str(definition.lineno) + ")\narg" + str(_argument_position) + " = " + ast.dump(definition, indent='\t') + '\n')

                        # extract name
                        diff_test_function_name = self._getDifferentialOracleName(definition)
                        log.add("Extracted name arg" + str(_argument_position) + ": " + diff_test_function_name + "\n", mode=LOG_FINAL)

                        # create data entry
                        log.createEntry(node.lineno, self.last_function_definition_node.name, self.last_function_definition_node.lineno, assert_statement, _argument_position, definition.lineno, diff_test_function_name)


                # clear ingore list for next definition
                self.ignore_list = []

        # visit child nodes (neccessary?)
        self.generic_visit(node)
    
    def _getDefinition(self, node):
        """Check if the argument node is the oracle and return its definitions, 
        i.e. a list of each value that was assigned to it.
        
        Returns None if the node is not an oracle.
        """
        
        if node in self.nodes_visited:
            return []
        
        self.nodes_visited.append(node)
        
        definitions = []
            
        # if the argument uses self.evaluate, we can be sure that this is the argument to test
        # therefore the other argument is the oracle and we can stop analyzing this one
        if self._isSelf_Evaluate(node):
            return None
        
        # if the argument is a named variable we need to trace it back to its definition to see if it uses another
        # function or library for differential testing
        if isinstance(node, ast.Name):
            log.add(node.id + " is a named variable!")
            
            # to prevent loops, check if variable is on ignore list
            if node.id in self.ignore_list:
                log.add(node.id + " was found on the ignore list and will not be checked further.\n")
                return []
            
            # check if the variable is in the argument list of the function in which this test case was defined
            if node.id in self.last_func_arg_list:
                log.add(node.id + " will not be checked further because it is in the argument list of the parent function.\n")
                return []
                    
            # store name of the variable to search for and ignore it for later searches for this test case
            variable_name = node.id
            self.ignore_list.append(variable_name)
            
            # iterate through each variable assignment in the function that the assert is called from 
            # and that is above the assert statement, then return all assigned values of our assert argument variable
            for child in ast.walk(self.last_function_definition_node):
                if isinstance(child, ast.Assign) and child.lineno <= node.lineno:
                    for target in child.targets:
                        
                        #print("ASSIGN TARGET: " + ast.dump(target, indent='\t'))
                        
                        # for list assignments of the form e.g. [a,b] = func([a,b]) or
                        # for tuple assignments, i.e. a,b = c
                        if isinstance(target, ast.List) or isinstance(target, ast.Tuple):
                            for list_target in target.elts:
                                if isinstance(list_target, ast.Name) and list_target.id == variable_name:
                                    
                                    # check if the list uses self.evaluate as a function, i.e. self.evaluate([a,b])
                                    if self._isSelf_Evaluate(child.value):
                                        return None
                                    
                                    definitions.append(child.value)
                                    
                        
                        # for regular variable assignments
                        if isinstance(target, ast.Name) and target.id == variable_name:
                            
                            # if the assignment includes a self.evaluate statement, return None
                            if self._isSelf_Evaluate(child.value):
                                return None
                            
                            # otherwise return the value of the variable 
                            definitions.append(child.value)

            
        # if the argument is a function call, extract the function name and trace its arguments back to their definitions
        if isinstance(node, ast.Call):
            log.add("Argument is a function call!")
            log.add(ast.dump(node, indent='\t'))

            # Check variable that the function is called on, e.g. check a when oracle is a.mean():
            # for regular calls e.g. a.mean()
            if hasattr(node, 'func') and hasattr(node.func, 'value'):
                # check if the function is an imported one
                if self._checkImports(node.func.value):
                    definitions.append(node.func)
                
                # get all definitions of the variable
                variable_definition = self._getDefinition(node.func.value)
                
            # for calls to defined functions
            elif hasattr(node, 'func') and hasattr(node.func, 'id'):                              
                # node.func will contain a ast.Name, where the Name.id is the name of the defined function
                variable_definition = [node.func]
                
                
            # for double function calls, e.g. func(a)(b)
            elif hasattr(node, 'func') and hasattr(node.func, 'func'):
                variable_definition = self._getDefinition(node.func.func)
            
            # for subscripts, e.g. a[:2].mean()
            elif hasattr(node, 'value'):
                variable_definition = self._getDefinition(node.value)
                
            
            # append the found definitions or return None when an oracle was found
            if self._appendDefinitionsUtil(definitions, variable_definition) is False:
                return None
            
            
            # Check arguments of the function
            for arg in node.args:
                log.add("Checking function argument:")
                variable_definition = self._getDefinition(arg)
            
                # append the found definitions or return None when an oracle was found
                if self._appendDefinitionsUtil(definitions, variable_definition) is False:
                    return None       
        
        
        if isinstance(node, ast.Attribute):
            variable_definition = self._getDefinition(node.value)
            
            # append the found definitions or return None when an oracle was found
            if self._appendDefinitionsUtil(definitions, variable_definition) is False:
                return None
      
    
        # if the argument is a starred expression, e.g. *[a,b] or a subscript, e.g. a[:2]
        if isinstance(node, ast.Starred): #or isinstance(node, ast.Subscript):
            log.add("Argument is a starred expression or subscript!")
            
            definitions = self._getDefinition(node.value)
        
        
        # if the argument is a list
        if isinstance(node, ast.List) or isinstance(node, ast.Tuple):
            log.add("Argument is a list or tuple")        

            # get the definition of each element
            for element in node.elts:
                element_definition = self._getDefinition(element)
                
                # append the found definitions or return None when an oracle was found
                if self._appendDefinitionsUtil(definitions, element_definition) is False:
                    return None

        
        # if the argument is a constant we do not need to look at this argument further,
        # i.e. we return an empty list
        if isinstance(node, ast.Constant):
            log.add("Argument is a constant")
            
        
        if isinstance(node, ast.BinOp):
            log.add("Argument is a binary operation")
            definitions.append(node)
            
        
        # For compare operators: Check all values that are compared
        if isinstance(node, ast.Compare):
            log.add("Argument is a compare operation")
            
            # append left side
            definitions.append(node.left)
            
            # append right side
            if self._appendDefinitionsUtil(definitions, node.comparators) is False:
                return None
            
            
        
        if definitions is not None:    
        
            # trace the variables another level back:    
            copy_defs = set(definitions.copy())
            for definition in copy_defs:
                
                if definition in self.nodes_visited:
                    continue
                
                log.add("def: " + ast.dump(definition, indent='\t'))

                
                # if the definition is a starred expression, e.g. *[a,b] or a subscript, e.g. a[:2]
                if isinstance(definition, ast.Starred) or isinstance(definition, ast.Subscript):
                    log.add("Tracing starred/subscript definition back")

                    definitions.remove(definition)

                    variable_definition = self._getDefinition(definition.value)

                    # append the found definitions or return None when an oracle was found
                    if self._appendDefinitionsUtil(definitions, variable_definition) is False:
                        return None

                    
                # if the definition is a list     
                elif isinstance(definition, ast.List) or isinstance(definition, ast.Tuple):
                    log.add("Unwrapping list or tuple")
                    
                    definitions.remove(definition)
                    
                    for i, element in enumerate(definition.elts):
                        log.add("Element " + str(i) + ": ", end='')
                        element_definition = self._getDefinition(element)
                        
                        # append the found definitions or return None when an oracle was found
                        if self._appendDefinitionsUtil(definitions, element_definition) is False:
                            return None

    
                # if the definition is an if-else statement, e.g. arg = 20 if dt == dtypes.float16 else 10
                elif isinstance(definition, ast.IfExp):
                    log.add("Definition is an If-Else statement")
                    
                    definitions.remove(definition)
                
                    variable_definitions = []
                    log.add("If-Body: ", end='')
                    variable_definitions.append(self._getDefinition(definition.body))
                    log.add("Else-Body: ", end='')
                    variable_definitions.append(self._getDefinition(definition.orelse))
                    
                    for variable_definition in variable_definitions:
                        # append the found definitions or return None when an oracle was found
                        if self._appendDefinitionsUtil(definitions, variable_definition) is False:
                            return None
                
                # if the definition is an Attribute statement, e.g. arg.value
                elif isinstance(definition, ast.Attribute):
                    log.add("Definition is an Attribute")
                    
                    definitions.remove(definition)
                    
                    # Check each assign statement if it matches the definition, then get the value assigned
                    # e.g. arg.value = func() will append the func() node to the definitions
                    for child in ast.walk(self.last_function_definition_node):
                        if isinstance(child, ast.Assign):
                            for target in child.targets:
                                
                                while isinstance(target, ast.Attribute):
                                    # Iterate through attributes and compare
                                    if target.attr in ast.dump(definition):
                                        # append the assigned value (right hand side of the assignment)
                                        definitions.append(child.value)
                                        break
                                    
                                    # move to next attribute
                                    target = target.value
                                    
                
                # if the definition is a binary operation, e.g. a+b, get the definitions of the left and
                # right hand side of the operation
                elif isinstance(definition, ast.BinOp):
                    log.add("Definition is a binary operation")
                    
                    definitions.remove(definition)
                
                    variable_definitions = []
                    log.add("Left hand side: ", end='')
                    variable_definitions.append(self._getDefinition(definition.left))
                    log.add("Right hand side: ", end='')
                    variable_definitions.append(self._getDefinition(definition.right))
                    
                    for variable_definition in variable_definitions:
                        # append the found definitions or return None when an oracle was found
                        if self._appendDefinitionsUtil(definitions, variable_definition) is False:
                            return None
                        
                        
                # if the definition resolves to a constant
                elif isinstance(definition, ast.Constant):
                    log.add("Definition is a constant")
                    
                    definitions.remove(definition)
                    
                    
                # if the definition is a function name
                elif isinstance(definition, ast.Name):
                    log.add("Definition is a defined function name")
                    
                    definitions.remove(definition)

                    # check to make sure the function was not passed to this test case from the arguments of the 
                    # function that the test case was defined in
                    if definition.id in self.last_func_arg_list:
                        log.add("Definition is an argument of the parent function.")
                        return []

                    # search for its definition within the function and check the return value.
                    # if it is not within the function, it might be a function that is defined for the 
                    # entire file, e.g. the entropy function in distribution/bernoulli_test.py
                    found_return_value = False
                    for search_location in [self.last_function_definition_node, self.tree]:

                        for child in ast.walk(search_location):
                            if isinstance(child, ast.FunctionDef) and child.name == definition.id:

                                log.add("Function definition found: "+ ast.dump(child, indent='\t'))

                                # find the return value of the function
                                for field in ast.walk(child):
                                    if isinstance(field, ast.Return):
                                        log.add("Found return value: ", end='')
                                        found_return_value = True

                                        return_value_definition = self._getDefinition(field.value)

                                        # append the found definitions or return None when an oracle was found
                                        if self._appendDefinitionsUtil(definitions, return_value_definition) is False:
                                            return None

                        if found_return_value:
                            break

                            
                # if the definition is a function, continue the analysis
                elif isinstance(definition, ast.Call):
                    #definitions.remove(definition)
                    
                    call_definition = self._getDefinition(definition)
                        
                    # append the found definitions or return None when an oracle was found
                    if self._appendDefinitionsUtil(definitions, call_definition) is False:
                        return None

  
             
        return definitions
    
    
    
    def _checkImports(self, node):
        """Check if the node (here usually an ast.Name) is an imported class"""
        _node = copy.deepcopy(node)
        if isinstance(_node, ast.Attribute):
            _node = _node.value
            
        if not isinstance(_node, ast.Name):
            return False
        
        for child in ast.walk(self.tree):
            if isinstance(child, ast.Import) or isinstance(child, ast.ImportFrom):
                # Imports always contain the field 'names': Import( names=[ alias(name='numpy', asname='np') ])
                # compare the asname to the node.id
                for alias in child.names:
                    if (hasattr(alias, 'asname') and alias.asname == _node.id) or (hasattr(alias, 'name') and alias.name == _node.id):
                        log.add("Corresponding import found: \n" + ast.dump(child, indent="\t"))
                        return True
                    
        return False
    
    
    def _appendDefinitionsUtil(self, definitions, definitions_to_append):
        """Append the found definitions or return False when an oracle was found."""
         # check if it is the oracle
        if definitions_to_append is not None:
            # append definitions 
            [definitions.append(defin) for defin in definitions_to_append]

        # if it is not the oracle
        else:
            return False
            
        
    def _getDifferentialOracleName(self, node):
        """Get the name of the function used in the differential test case."""
        
        oracle_attributes = []
        oracle_name = ""
        
        # if it is a function call or attribute
        if isinstance(node, ast.Call) or isinstance(node, ast.Attribute):
            
            # for function: move into the function of the call node (ignoring the arguments of the function call)
            _node = node.func if isinstance(node, ast.Call) else node
            
            # if the function call has multiple attributes, i.e. np.linalg.solve()
            while isinstance(_node, ast.Attribute):
                # extract the name of the function
                oracle_attributes.append(_node.attr)
                
                # move another layer deeper into the node
                _node = _node.value
            
            # if this is the last attribute, i.e. we are at the deepest level of the call node
            if isinstance(_node, ast.Name):
                oracle_attributes.append(_node.id)
            
            # construct the oracle name by going through the reversed attributes array
            for attr in oracle_attributes[::-1]:
                oracle_name += attr + "."
            
            # remove the dot at the end
            oracle_name = oracle_name[:-1]
        
        
        # if it is a list comprehension, find func when node is e.g. [func(a) for a in b]
        if isinstance(node, ast.ListComp):
            # recursively call this function to extract the function name
            oracle_name = self._getDifferentialOracleName(node.elt)               
                
                
        # Catch errors: 
        
        # Should have checked the variables of binary operations e.g. arg1 = a/b
        if isinstance(node, ast.BinOp):
            oracle_name = "UNSUPPORTED Binary Operation"
            
        if isinstance(node, ast.UnaryOp):
            oracle_name = "UNSUPPORTED Unary Operation"
            
        # Should have been disregarded by the _getDefinition function
        if isinstance(node, ast.Constant):
            oracle_name = "UNSUPPORTED Constant"
            
        # Should have been analysed further by the _getDefinition function
        if isinstance(node, ast.Name):
            oracle_name = "UNSUPPORTED Name (named variable or defined function: " + node.id + ")"
        
        if isinstance(node, ast.Compare):
            oracle_name = "UNSUPPORTED Compare"
            
        if isinstance(node, ast.ListComp):
            oracle_name = "UNSUPPORTED List Comprehension"
        
        # return the oracle function name
        return oracle_name
                
        
    def _isSelf_Evaluate(self, node):
        """Check if the node is a self.evaluate call. If so, the node is not an oracle."""
        
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'evaluate':
            log.add("Argument is an evaluate function call!")
            return True
        return False
        
    def visit_FunctionDef(self, node):
        
        # check if the function definition contains self
        if 'self' in ast.dump(node.args, indent='\t'):
            # remember the  last class-level function definition seen
            self.last_function_definition_node = node
            
            # store all function arguments to check if we encounter one later
            self.last_func_arg_list = [n.arg for n in node.args.args]
            
            # TESTING: to catch cases
            if node.args.posonlyargs != [] or node.args.kwonlyargs != []:
                log.add("NEW ARGS FOUND", mode=LOG_ALL)
            
        # catch the case where the first function of a file is not a class member 
        elif self.last_function_definition_node == None:
            self.last_function_definition_node = node
            
        # visit child nodes
        self.generic_visit(node)


        
        
def debug_single_file_extraction(file_path, print_mode=LOG_FINAL, log_mode=LOG_ALL):
    """Extract differential test cases of a single file.
        file_path: Path to the file on which the extraction should run.
        print_mode: Use difftest.LOG_ALL (=0) to see a full output log,
        difftest.LOG_FINAL (=1) to only see test cases and extracted differential functions,
        and difftest.LOG_NONE (=2) to surpress a log output.
        save_log_to: Location to save the generated log of the extraction to.
        Returns df: Pandas dataframe containing information on the identified test cases.
        """
    
    # e.g. ".../tensorflow/python/kernel_tests/distributions/gamma_test.py"
    source = open(file_path)

    # generate abstract syntax tree
    tree = ast.parse(source.read())

    # Initialize custom logger globally
    # Use the print mode LOG_FINAL for final output and LOG_ALL for debugging
    global log
    log = CustomLogger(print_mode, log_mode)
    
    log.set_file_path_variable(file_path)

    # start the tree traversal
    TreeTraverser(tree).visit(tree)

    log.log.close()
    
    data = log.get_data()
    df = pd.DataFrame(data, columns=data[0])
    
    return df


def extract_diff_test_cases(library_root, save_data_to='extracted_data/data.csv'):
    """TODO
        library_root: path to the root of the library to be analysed, e.g. .../Tensorflow/tensorflow-2.6.0/tensorflow/python/
        save_data_to: File to save the data to, by default saved to /extracted_data/data.csv"""
    # fill test_files with the file paths to all python files relative to library_root that contain 'test' in their name 
    test_files = []
    for subdir, _, files in os.walk(library_root):
        #print(subdir)
        for file in files:
            if file.endswith(".py") and 'test' in file:
                relative_dir = subdir.replace(library_root, '')
                filepath = relative_dir + os.sep + file
                test_files.append(filepath) 
                #print("\t"+filepath)




    # Initialize custom logger
    global log
    log = CustomLogger(LOG_NONE)

    error_list = []
    # go through each file and extract data
    for file in tqdm(test_files):
        # set file path that appears in the data entries for this file 
        log.set_file_path_variable(file)
        log.add("<<<<<<<<<<<<" + file + ">>>>>>>>>>>>", LOG_FINAL)
        
        source = open(library_root + file, encoding="utf8")
        
        try:
            # generate abstract syntax tree and start the tree traversal as before
            tree = ast.parse(source.read())
            TreeTraverser(tree).visit(tree)
        
        except:
            error_list.append(file)

    # print collected data and save to csv
    #print(log.get_data())
    print("Remaining errors in files: " + str(len(error_list)) + " " + str(error_list))
    log.save_data_to_csv(save_data_to)
