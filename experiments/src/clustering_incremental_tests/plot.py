from os import listdir
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
    save_path = files[0].split("summary")[0]

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
                
                new_df = df.groupby(["Solver","CompileCluster"],as_index=False)["RealGap"].agg(lambda x: 
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
                
                new_df = df.groupby(["Solver","CompileCluster"],as_index=False)["RealGap"].agg(lambda x: 
                                                                        len(x[x>=i])/len(x)
                                                                        )

                
                # new_df.insert(0,"Gap_Level",[i for j in range(len(new_df))],True)
                new_df["Gap_Level"] = i
                all_df.append(new_df)
        all_df = pd.concat(all_df)
        all_df['Solver'] = all_df['Solver'].str.strip()
        all_df['CompileCluster'] = all_df['CompileCluster'].str.strip()
        all_df['Solver'] = all_df['Solver'].map({'IR': 'incremental', 'TD': 'top-down', 'BB': 'branch-bound'})
        all_df['CompileCluster'] = all_df['CompileCluster'].map({'true': 'CompileCluster', 'false': ''})
        all_df['Label'] = all_df['Solver'] + "-" + all_df['CompileCluster']
        print(all_df['Solver'].unique())
        print(all_df.columns)
        #to use multiple columns for label or hue
        #hue=all_df[["Solver", "CompileCluster"]].apply(tuple, axis=1)
        ax = sns.lineplot(x="Gap_Level", y="RealGap", hue="Label", data=all_df, ax=axs[index])
        ax.legend_.set_title(None)
        if args.type == "min":
            ax.invert_xaxis()
        ax.set_ylabel("% of Instances with Better Bound")
        ax.set_xlabel("Normalised Bound")
        name = file.split("/")[-1].split(".")[0].split("_")[-1]
        ax.set_title(f"width {name}")
    print(save_path)
    plt.savefig(f'{save_path}/plot.png')
    # plt.show()

def plot_bound_width(files,type):
    # files = args.input
    save_path = files[0].split("summary")[0]

    # fig, axs = plt.subplots(ncols=len(args.input))
    all_df = []
    for index,file in enumerate(files):
        df = pd.read_csv(file,header=0)
        

        if type == "max":
            df["RealUpper"] = df.groupby('Name')['Upper'].transform('min')
            df["RealGap"] = (df['Upper'] - df['RealUpper'])/df['Upper']
            print(df["RealGap"].min(),df["RealGap"].max())
            print(df)
            new_df = df.groupby(["Solver","CompileCluster","Dominance","RefineCluster","Binary","Width"],as_index=False)["RealGap"].mean()
            all_df.append(new_df)
        else: #is min
            # df["Upper"] = df["Upper"]*(-1)
            df["RealUpper"] = df.groupby('Name')['Upper'].transform('max')
            df["RealGap"] = (df['RealUpper'] - df['Upper'])/df['RealUpper']
            print(df["RealGap"].min(),df["RealGap"].max())
            print(df)
            new_df = df.groupby(["Solver","CompileCluster","Dominance","RefineCluster","Binary","Width"],as_index=False)["RealGap"].mean()
            all_df.append(new_df)
 
        
    all_df = pd.concat(all_df)
    # print(all_df.to_string())
    all_df['Solver'] = all_df['Solver'].str.strip()
    all_df['CompileCluster'] = all_df['CompileCluster'].str.strip()
    all_df['Dominance'] = all_df['Dominance'].str.strip()
    all_df['RefineCluster'] = all_df['RefineCluster'].str.strip()
    all_df['Binary'] = all_df['Binary'].str.strip()

    all_df['Solver'] = all_df['Solver'].map({'IR': 'incremental', 'TD': 'top-down', 'BB': 'branch-bound'})

    # mask = all_df['Solver'].isin(['branch-bound'])
    # all_df = all_df[~mask]

    # all_df['CompileCluster'] = all_df['CompileCluster'].map({'true': 'CC', 'false': 'X'})
    all_df['Dominance'] = all_df['Dominance'].map({'true': 'DD', 'false': 'X'})
    all_df['RefineCluster'] = all_df['RefineCluster'].map({'true': 'RC', 'false': 'X'})
    all_df['Binary'] = all_df['Binary'].map({'true': 'B', 'false': 'X'})

    # all_df['Label'] = all_df['Solver'] + "-" + all_df['CompileCluster'] + "-" + all_df['RefineCluster'] + "-" + all_df['Binary']
    all_df['Label'] = all_df['Solver'] + "-" + all_df['Dominance'] + "-" + all_df['RefineCluster'] + "-" + all_df['Binary']
    return(all_df)
    # print(all_df['Solver'].unique())
    # print(all_df.columns)
    #to use multiple columns for label or hue
    #hue=all_df[["Solver", "CompileCluster"]].apply(tuple, axis=1)
    ax = sns.lineplot(x="Width", y="RealGap", hue="Label", data=all_df)
    ax.legend_.set_title(None)
    # if args.type == "min":
    #     ax.invert_xaxis()
    ax.set_ylabel("Normalised Bound")
    ax.set_xlabel("Width")
    name = file.split("/")[-1].split(".")[0].split("_")[-1]
    # ax.set_title(f"width {name}")
    
    print(save_path)
    # plt.savefig(f'{save_path}/{args.name}')
    plt.show()
    

def plot_cluster(problem_names):
    all_df = []
    for (problem,folder,type) in problem_names:
        print(problem)
        path = f"/home/eaeigbe/Documents/PhD/ddo/experiments/results/Cluster/{folder}"
        filenames = [ path+"/"+filename for filename in listdir(path) if filename.endswith( "csv" ) ]
        df = plot_bound_width(filenames,type)
        df["Problem"] = problem
        # print(df.to_string())
        all_df.append(df)

    all_df = pd.concat(all_df)
    all_df.reset_index(level=None, drop=False, inplace=True, col_level=0, col_fill="")

    # print(all_df.to_string())
    all_df = all_df.drop(all_df[all_df["Solver"] == 'incremental'].index)
    all_df = all_df.drop(all_df[all_df["Solver"] == 'branch-bound'].index)
    # print(all_df.to_string())

    g = sns.FacetGrid(all_df, col="Problem", col_wrap=2)
    g.map_dataframe(sns.lineplot,x="Width", y="RealGap", hue="Label").add_legend() 
    g.set_ylabels("Normalised Bound")
    g.set_xlabels("Width")
    plt.show() 


def plot_dominance(problem_names):
    all_df = []
    for (problem,folder,type) in problem_names:
        print(problem)
        path = f"/home/eaeigbe/Documents/PhD/ddo/experiments/results/Dominance/{folder}"
        filenames = [ path+"/"+filename for filename in listdir(path) if filename.endswith( "csv" ) ]
        df = plot_bound_width(filenames,type)
        df["Problem"] = problem
        # print(df.to_string())
        all_df.append(df)

    all_df = pd.concat(all_df)
    all_df.reset_index(level=None, drop=False, inplace=True, col_level=0, col_fill="")

    # print(all_df.to_string())
    all_df = all_df.drop(all_df[all_df["Solver"] == 'incremental'].index)
    all_df = all_df.drop(all_df[all_df["Solver"] == 'branch-bound'].index)
    # print(all_df.to_string())

    g = sns.FacetGrid(all_df, col="Problem", col_wrap=2)
    g.map_dataframe(sns.lineplot,x="Width", y="RealGap", hue="Label").add_legend() 
    g.set_ylabels("Normalised Bound")
    g.set_xlabels("Width")
    plt.show() 


if __name__ == "__main__":
    # plot_cluster([
    #     ("talentsched", "talentsched", "min"),
    #     ("srflp", "srflp", "min"),
    #     ("tsptw", "tsptw/AFG", "min"),
    #     ("misp", "misp", "max"),
    #     ("sop", "sop", "min"),
    #     ("mcp", "mcp", "max"),
    #     ("knapsack", "knapsack", "max"),
    #     ("max2sat", "max2sat", "max"),
    #     ("psp", "psp/instancesWith2items", "min"),
    #     ("lcs", "lcs", "max"),

    #     ])
    

    plot_dominance([
        ("knapsack", "knapsack", "max"),
        ])


    parser = argparse.ArgumentParser(description='Choose full or abridged verification.')
    # '''custom argument type for string list'''
    # def list_of_strings(arg):
    #     return arg.split(',')
    # parser.add_argument('--input',
    #                     '-i', 
    #                     type=list_of_strings,
    #                     help='input file to plot results from')
    # parser.add_argument('--name',
    #                     '-n', 
    #                     type=str,
    #                     help='problem/output file name')
    # parser.add_argument('--type',
    #                     '-t',  
    #                     type=str,
    #                     default='max',
    #                     help='max(min) for maximisation(minimisation)')
    



    # args = parser.parse_args()
    # main(args)