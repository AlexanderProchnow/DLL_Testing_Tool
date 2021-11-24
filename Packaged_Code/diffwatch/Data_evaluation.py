import pandas as pd
import os

def coverage_analysis(df):
    
    # select all data entries that are unsupported or NaN.
    not_covered_df = df[df['Differential_Test_Function'].str.contains('UNSUPPORTED', na=False) | df['Differential_Test_Function'].isna()]
    
    # print coverage numbers and percent
    print(str(len(not_covered_df)) + " out of " + str(len(df)) +   " cases not covered ({}%)".format(round(len(not_covered_df)/len(df)*100, 2)))
    
    # print a count of how often each different unsupported case and NaN occurs
    print(not_covered_df.Differential_Test_Function.value_counts(dropna=False))




UNEVALUATED_STRING = "UNEVALUATED"


def evaluation_analysis(eval_data):
    """Print a count of the evaluations, i.e. how many were evaluated with y/n/unevaluated."""
    evaluation_counts = eval_data.Evaluation.value_counts()

    print(evaluation_counts)

    total_cases_evaluated = len(eval_data) - evaluation_counts[UNEVALUATED_STRING] 

    try:
        print("\nn: " + str(round((evaluation_counts['n'] / total_cases_evaluated)*100)) + " %")

        print("y: " + str(round((evaluation_counts['y'] / total_cases_evaluated)*100)) + " %")

        print("?: " + str(round(((total_cases_evaluated - evaluation_counts['y'] - evaluation_counts['n']) / total_cases_evaluated)*100)) + " %")
        
    except:
        print("\nNo evaluations of 'y' or 'n' found!")



class EvaluationAutomator:
    def __init__(self, df, library_root, save_data_to):
        """Initialize the evaluation automator.
        
        df: Dataframe to evaluate.
        library_root: The root folder of the DL library
        save_data_to: Relative location to load/save the evaluation data
        """
        self.df = df
        self.save_data_to = save_data_to
        self.library_root = library_root
        
        # try importing evaluation data if it already exists
        if os.path.isfile(self.save_data_to): 
            self.eval_df = pd.read_csv(self.save_data_to)
            print("Evaluation data opened.")
        
        # otherwise initialize evaluation df and add new column for the evaluation result
        else:
            self.eval_df = df.copy()
            todo_list = [UNEVALUATED_STRING] * len(self.eval_df.index)
            self.eval_df.insert(len(df.columns), 'Evaluation', todo_list)
            self.eval_df.to_csv(self.save_data_to)
            print("New evaluation data created.")
            
    def getEvalData(self):
        """Returns the data frame containing the evaluation data."""
        return self.eval_df
    
    def evaluate(self, index):
        """Present the data entry at the given index for evaluation."""
        
        # present the data entry
        print(self.df.iloc[index])
        print("\n")
        
        # check if it has already been evaluated
        if self.eval_df.at[index, 'Evaluation'] != UNEVALUATED_STRING:
            print("Already evaluated! Previous evaluation: " + self.eval_df.at[index, 'Evaluation'])
            if input("Re-evaluate? (y / n) ") != "y":
                return
            
        
        # print the relevant source code lines:
        
        # get source file of current test case and open it as an array of lines
        source_file_path = self.df.iloc[index]['File_Path'].replace('\\', '/')
        source = open(self.library_root + source_file_path).readlines()

        # set beginning and end line number for the code section to display
        beginning_line_no = self.df.iloc[index]['Function_Definition_Line_Number']
        end_line_no = self.df.iloc[index]['Line_Number']

        # print these lines 
        for line in range(beginning_line_no, end_line_no+1):
            print(str(line) + ": " + source[line-1])
            
        # check if the last line with the assert statement is complete or if the
        # assert arguments were moved to new lines, in which case: print more lines
        line = end_line_no
        last_line = source[line-1]       
        
        # we can check if the assert statement is complete if it ends with a closed bracket
        while not last_line.rstrip().endswith(")"):
            line += 1
            last_line = source[line-1]
            print(str(line) + ": " + last_line)
            
        # ask for a decision from the evaluator:
        decision_bool = True
        while decision_bool:
            decision = input("Correctly identified? (y / n / ?): ")
            
            if decision in ["y", "n"]:
                decision_bool = False

            elif decision == "?":
                decision = input("Please comment on this case: ")
                decision_bool = False
                
            else:
                print("Error. Please specify y/n/?")
                decision_bool = True
                
        # write the decision to the evaluation data
        self.eval_df.at[index, 'Evaluation'] = decision
        self.eval_df.to_csv(self.save_data_to, index=False)

    def sample_cases_and_evaluate(self, num_cases=50, random_seed=42):
        """Sample cases and present them to the user for evaluation."""
        sampled_cases = self.eval_df.sample(n=num_cases, random_state=random_seed)

        sample_counter = 0

        # iterate over each case and evaluate
        for i, row in sampled_cases.iterrows():
            print("\nCase " + str(i) + " (" + str(sample_counter) + " / " + str(len(sampled_cases)) + ")\n")
            self.evaluate(i)
            sample_counter += 1

