import pandas as pd
import os

def coverage_analysis(df):
    
    # select all data entries that are unsupported or NaN.
    not_covered_df = df[df['Differential_Test_Function'].str.contains('UNSUPPORTED', na=False) | df['Differential_Test_Function'].isna()]
    
    # print coverage numbers and percent
    print(str(len(not_covered_df)) + " out of " + str(len(df)) +   " cases not covered ({}%)".format(round(len(not_covered_df)/len(df)*100, 2)))
    
    # print a count of how often each different unsupported case and NaN occurs
    print(not_covered_df.Differential_Test_Function.value_counts(dropna=False))