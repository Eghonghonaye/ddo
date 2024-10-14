# from click import FloatRange
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
from pathlib import Path



def main(args):
    # file = "/home/eaeigbe/Documents/PhD/ddo/experiments/summary_w1_10_w2_100.csv"
    files = args.input

    fig, axs = plt.subplots(ncols=len(args.input))
    if len(args.input) == 1:
        axs = [axs]

    for index,file in enumerate(files):
        df = pd.read_csv(file,header=0)
        all_df = []
        
        df["RealUpper"] = df.groupby('Name')['Upper'].transform('min')
        df["RealGap"] = (df['Upper'] - df['RealUpper'])/df['Upper']
        print(df["RealGap"].min(),df["RealGap"].max())
        print(df)
        all_df.append(df)
        all_df = pd.concat(all_df)

        ax = sns.lineplot(x="Name", y="RealGap", hue="Solver", data=all_df, ax=axs[index])
        ax.legend_.set_title(None)
        ax.set_ylabel("Optimality Gap")
        ax.set_xlabel("Instance")
        name = file.split("/")[-1].split(".")[0].split("_")[-1]
        ax.set_title(f"width {name}")
    
    plt.show()

        

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Choose full or abridged verification.')
    '''custom argument type for string list'''
    def list_of_strings(arg):
        return arg.split(',')
    parser.add_argument('--input',
                        '-i', 
                        type=list_of_strings,
                        help='input files to plot results from')



    args = parser.parse_args()
    main(args)