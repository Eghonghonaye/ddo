from click import FloatRange
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
from pathlib import Path



def main(args):
    # file = "/home/eaeigbe/Documents/PhD/ddo/experiments/summary_w1_10_w2_100.csv"
    # plot_how_many_better(args)
    plot_bound_width(args)
    

def plot_how_many_better(args):
    files = args.input

    fig, axs = plt.subplots(ncols=len(args.input))

    for index,file in enumerate(files):
        df = pd.read_csv(file,header=0)
        all_df = []

        if args.type == "max":
            df["RealUpper"] = df.groupby('Name')['Upper'].transform('min')
            df["RealGap"] = (df['Upper'] - df['RealUpper'])/df['Upper']
            print(df["RealGap"].min(),df["RealGap"].max())
            print(df)
            for i in np.arange(df["RealGap"].min(),df["RealGap"].max()+0.1,0.01):
                
                new_df = df.groupby(["Solver","Cluster"],as_index=False)["RealGap"].agg(lambda x: 
                                                                        len(x[x<=i])/len(x)
                                                                        )

                
                # new_df.insert(0,"Gap_Level",[i for j in range(len(new_df))],True)
                new_df["Gap_Level"] = i
                all_df.append(new_df)
        else: #is min
            df["RealUpper"] = df.groupby('Name')['Upper'].transform('max')
            df["RealGap"] = (df['RealUpper'] - df['Upper'])/df['RealUpper']
            print(df["RealGap"].min(),df["RealGap"].max())
            print(df)
            for i in np.arange(df["RealGap"].min(),df["RealGap"].max()+0.1,0.01):
                
                new_df = df.groupby(["Solver","Cluster"],as_index=False)["RealGap"].agg(lambda x: 
                                                                        len(x[x>=i])/len(x)
                                                                        )

                
                # new_df.insert(0,"Gap_Level",[i for j in range(len(new_df))],True)
                new_df["Gap_Level"] = i
                all_df.append(new_df)
        all_df = pd.concat(all_df)
        all_df['Solver'] = all_df['Solver'].str.strip()
        all_df['Cluster'] = all_df['Cluster'].str.strip()
        all_df['Solver'] = all_df['Solver'].map({'IR': 'incremental', 'TD': 'top-down', 'BB': 'branch-bound'})
        all_df['Cluster'] = all_df['Cluster'].map({'true': 'cluster', 'false': ''})
        all_df['Label'] = all_df['Solver'] + "-" + all_df['Cluster']
        print(all_df['Solver'].unique())
        print(all_df.columns)
        #to use multiple columns for label or hue
        #hue=all_df[["Solver", "Cluster"]].apply(tuple, axis=1)
        ax = sns.lineplot(x="Gap_Level", y="RealGap", hue="Label", data=all_df, ax=axs[index])
        ax.legend_.set_title(None)
        if args.type == "min":
            ax.invert_xaxis()
        ax.set_ylabel("% of Instances with Better Bound")
        ax.set_xlabel("Normalised Bound")
        name = file.split("/")[-1].split(".")[0].split("_")[-1]
        ax.set_title(f"width {name}")
    
    plt.show()

def plot_bound_width(args):
    files = args.input

    # fig, axs = plt.subplots(ncols=len(args.input))
    all_df = []
    for index,file in enumerate(files):
        df = pd.read_csv(file,header=0)
        

        if args.type == "max":
            df["RealUpper"] = df.groupby('Name')['Upper'].transform('min')
            df["RealGap"] = (df['Upper'] - df['RealUpper'])/df['Upper']
            print(df["RealGap"].min(),df["RealGap"].max())
            print(df)
            new_df = df.groupby(["Solver","Cluster","Width"],as_index=False)["RealGap"].mean()
            all_df.append(new_df)
        else: #is min
            df["RealUpper"] = df.groupby('Name')['Upper'].transform('max')
            df["RealGap"] = (df['RealUpper'] - df['Upper'])/df['RealUpper']
            print(df["RealGap"].min(),df["RealGap"].max())
            print(df)
            new_df = df.groupby(["Solver","Cluster","Width"],as_index=False)["RealGap"].mean()
            all_df.append(new_df)
 
        
    all_df = pd.concat(all_df)
    print(all_df)
    all_df['Solver'] = all_df['Solver'].str.strip()
    all_df['Cluster'] = all_df['Cluster'].str.strip()
    all_df['Solver'] = all_df['Solver'].map({'IR': 'incremental', 'TD': 'top-down', 'BB': 'branch-bound'})
    all_df['Cluster'] = all_df['Cluster'].map({'true': 'cluster', 'false': ''})
    all_df['Label'] = all_df['Solver'] + "-" + all_df['Cluster']
    # print(all_df['Solver'].unique())
    # print(all_df.columns)
    #to use multiple columns for label or hue
    #hue=all_df[["Solver", "Cluster"]].apply(tuple, axis=1)
    ax = sns.lineplot(x="Width", y="RealGap", hue="Label", data=all_df)
    ax.legend_.set_title(None)
    # if args.type == "min":
    #     ax.invert_xaxis()
    ax.set_ylabel("Normalised Bound")
    ax.set_xlabel("Width")
    name = file.split("/")[-1].split(".")[0].split("_")[-1]
    # ax.set_title(f"width {name}")
    
    plt.show()

    

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Choose full or abridged verification.')
    '''custom argument type for string list'''
    def list_of_strings(arg):
        return arg.split(',')
    parser.add_argument('--input',
                        '-i', 
                        type=list_of_strings,
                        help='input file to plot results from')
    parser.add_argument('--type',
                        '-t',  
                        type=str,
                        default='max',
                        help='max(min) for maximisation(minimisation)')



    args = parser.parse_args()
    main(args)