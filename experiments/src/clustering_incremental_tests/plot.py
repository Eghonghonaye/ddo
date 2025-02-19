import json
from os import listdir
import os
import sys
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


def plot_bound_width(files,type):
    # files = args.input
    # save_path = files[0].split("summary")[0]

    # fig, axs = plt.subplots(ncols=len(args.input))
    all_df = []
    for index,file in enumerate(files):
        df = pd.read_csv(file,header=0)
        print(file)

        if type == "max":
            df["MergeQuality"] = abs(df["MergeQuality"].astype(float))
            df['Upper'] = df['Upper'].astype(float)

            df["RealUpper"] = df.groupby('Name')['Upper'].transform('min')
            df["RealGap"] = (df['Upper'] - df['RealUpper'])/df['Upper']
            print(df["RealGap"].min(),df["RealGap"].max())
            print(file,df.columns)
            # new_df = df
            # new_df =df.groupby(["Solver","CompileCluster","RefineCluster","Binary","MergeQuality","Width"],as_index=False)["RealGap"].mean()
            new_df =df.groupby(["Solver","CompileCluster","RefineCluster","Binary","Width"],as_index=False).agg({"RealGap":"mean","MergeQuality":"mean"})
            all_df.append(new_df)
        else: #is min
            # df["Upper"] = df["Upper"]*(-1)
            df["MergeQuality"] = abs(df["MergeQuality"].astype(float))
            df['Upper'] = df['Upper'].astype(float)

            df["RealUpper"] = df.groupby('Name')['Upper'].transform('max')
            df["RealGap"] = (df['RealUpper'] - df['Upper'])/df['RealUpper']
            print(df["RealGap"].min(),df["RealGap"].max())
            print(file,df.columns)
            # new_df = df
            # new_df = df.groupby(["Solver","CompileCluster","RefineCluster","Binary","MergeQuality","Width"],as_index=False)["RealGap"].mean()
            new_df =df.groupby(["Solver","CompileCluster","RefineCluster","Binary","Width"],as_index=False).agg({"RealGap":"mean","MergeQuality":"mean"})
            all_df.append(new_df)
 
        
    all_df = pd.concat(all_df)
    # print(all_df.to_string())
    all_df['Solver'] = all_df['Solver'].str.strip()
    all_df['CompileCluster'] = all_df['CompileCluster'].str.strip()
    # all_df['Dominance'] = all_df['Dominance'].str.strip()
    all_df['RefineCluster'] = all_df['RefineCluster'].str.strip()
    all_df['Binary'] = all_df['Binary'].str.strip()

    all_df['Solver'] = all_df['Solver'].map({'IR': 'incremental', 'TD': 'top-down', 'BB': 'branch-bound'})

    # mask = all_df['Solver'].isin(['branch-bound'])
    # all_df = all_df[~mask]

    # all_df['CompileCluster'] = all_df['CompileCluster'].map({'true': 'CC', 'false': 'X'})
    # all_df['Dominance'] = all_df['Dominance'].map({'true': 'DD', 'false': 'X'})
    all_df['RefineCluster'] = all_df['RefineCluster'].map({'true': 'RC', 'false': 'X'})
    all_df['Binary'] = all_df['Binary'].map({'true': 'B', 'false': 'X'})

    # all_df['Label'] = all_df['Solver'] + "-" + all_df['CompileCluster'] + "-" + all_df['RefineCluster'] + "-" + all_df['Binary']
    # all_df['Label'] = all_df['Solver'] + "-" + all_df['Dominance'] + "-" + all_df['RefineCluster'] + "-" + all_df['Binary']
    return(all_df)


def analyse(input,output):
    # Open and read the JSON file
    pathlist = Path(input).glob("*")
    outputfile = open(output,'a')
    for file in pathlist:
        filename = os.fsdecode(file)
        with open(filename, 'r') as f:
            results = json.load(f)
            outputfile.write(f"""{filename.split("/")[-1].split(".")[0]}, \
            {abs(float(results["Lower Bnd"]))}, \
            {abs(float(results["Upper Bnd"]))}, \
            {results["Duration"]},\
            {results["Aborted"]}, \
            {results["Refine Cluster"]},\
            {results["Compile Cluster"]},\
            {"-" if "Dominance" not in results else results["Dominance"]},\
            {float(0) if "MergeQuality" not in results else float(results["MergeQuality"])},\
            {results["Binary Split"]},\
            {results["Solver"]},\
            {results["Width"]},\
            {results["Gap"]},\
            {float(results["Objective"])}\n""")
            # {str(results["Solution"]) } \n""")



def plot(problem_names):
    all_df = []
    for (problem,subfolder,experiments,type) in problem_names:
        for folder in experiments:           
            path = f"/home/eaeigbe/Documents/PhD/ddo/experiments/results/{folder}/{subfolder}"
            summary_file = f"/home/eaeigbe/Documents/PhD/ddo/experiments/results/{folder}/{subfolder}/summary.csv"
            if os.path.exists(summary_file):
                os.remove(summary_file)
            with open(summary_file,'a') as f:
                f.write("Name,Lower,Upper,Duration,Aborted,RefineCluster,CompileCluster,Dominance,MergeQuality,Binary,Solver,Width,Gap,Objective\n")


            directories = [x[0] for x in os.walk(path) if x[0]!=path]
            directories.append(f"/home/eaeigbe/Documents/PhD/ddo/experiments/results/BnB/{subfolder}/TD_B_n_B")


            for directory in directories:
                analyse(directory,summary_file)
 

            # filenames = [ path+"/"+filename for filename in listdir(path) if filename.endswith( "csv" ) ]
            filenames = [summary_file]
            # sys.exit()
            df = plot_bound_width(filenames,type)
            df["Problem"] = problem
            df["Conditions"] = folder
            df['Label'] = df['Solver'] + "-->" + df['Conditions']
            print(df.to_string())
            all_df.append(df)

    all_df = pd.concat(all_df)
    all_df.reset_index(level=None, drop=False, inplace=True, col_level=0, col_fill="")

    # print(all_df.to_string())
    # all_df = all_df.drop(all_df[all_df["Solver"] == 'top-down'].index)
    all_df = all_df.drop(all_df[all_df["Solver"] == 'branch-bound'].index)
    all_df["MergeQuality"] =  all_df["MergeQuality"].astype(float)
    all_df = all_df.dropna()
    all_df.to_csv(f"experiments/all_{experiments}.csv")

    def joint_plot(data: pd.DataFrame, x_name: str, y1_name: str, y2_name: str, ylabel1: str, ylabel2: str,color="blue",label=0):
        ax1 = plt.gca()
        sns.lineplot(x=data[x_name], y=data[y1_name], alpha=.7, color='red',ax=ax1)
        ax1.set_ylabel(ylabel1)

        ax2 = ax1.twinx()
        sns.lineplot(x=data[x_name], y=data[y2_name], alpha=.7,ax=ax2)
        ax2.set_ylabel(ylabel2)


    # g = sns.FacetGrid(all_df, row="Problem", col="Conditions",margin_titles=True)
    g = sns.FacetGrid(all_df, col="Problem", margin_titles=True)
    g.map_dataframe(sns.lineplot,x="Width", y="RealGap", hue="Label").add_legend() 
    # g.map_dataframe(joint_plot, x_name="Width", y1_name="RealGap", y2_name="MergeQuality",
    #                 ylabel1="Gap", ylabel2="Error")
    # g.set_ylabels("Normalised Bound")
    # g.set_xlabels("Width")
    g.set_titles(row_template="{row_name}",col_template="{col_name}", fontweight='bold', size=14) 
    plt.show() 
    # plt.savefig(f'{experiments}plot.png')

    # ax =  sns.lineplot(x="Width", y="RealGap", hue="Label", data=all_df)
    # ax2 = ax.twinx()
    # sns.lineplot(x="Width", y="MergeQuality", hue="Label", ax=ax2, data=all_df)
    # # ax.figure.legend()
    # plt.show()





if __name__ == "__main__":
    # plot([
    #     ("tsptw", "tsptw/AFG",  ["Gewoon"],"min"),
    #     ("knapsack", "knapsack", ["Gewoon"],"max"),
    #     ("sop", "sop", ["Gewoon"],"min"),
    #     ])
    
    # plot([
    #     ("tsptw", "tsptw/AFG",  ["Dominance","Dominance+Cluster","Gewoon","Cluster"],"min"),
    #     ("knapsack", "knapsack", ["Dominance","Dominance+Cluster","Gewoon","Cluster"],"max"),
    #     ("sop", "sop", ["Dominance","Dominance+Cluster","Gewoon","Cluster"],"min"),
    #     ])
    
    # plot([
        # ("talentsched", "talentsched", ["Gewoon","Cluster"],"min"),
        # ("srflp", "srflp", ["Gewoon","Cluster"],"min"),
        # ("tsptw", "tsptw/AFG",  ["Gewoon","Cluster"],"min"),
        # ("misp", "misp", ["Gewoon","Cluster"],"max"),
        # ("sop", "sop", ["Gewoon","Cluster"],"min"),
        # ("mcp", "mcp", ["Gewoon","Cluster"],"max"),
        # ("knapsack", "knapsack", ["Gewoon","Cluster"],"max"),
        # ("max2sat", "max2sat", ["Gewoon","Cluster"],"max"),
        # ("psp", "psp/instancesWith2items", ["Gewoon","Cluster"],"min"),
        # ("lcs", "lcs",  ["Gewoon","Cluster"],"max"),
        # ("sop", "sop",  ["Gewoon","Dominance","Cluster","Dominance+Cluster"],"min"),
        # ("alp", "alp",  ["Gewoon","Dominance","Cluster","Dominance+Cluster"],"min"),
        # ])
    
    plot([
        ("tsptw", "tsptw/AFG",  ["Dominance","Dominance+Cluster","Gewoon","Cluster"],"min"),
        ("knapsack", "knapsack", ["Dominance","Dominance+Cluster","Gewoon","Cluster"],"max"),
        ("sop", "sop", ["Dominance","Dominance+Cluster","Gewoon","Cluster"],"min"),
        ("alp", "alp", ["Dominance","Dominance+Cluster","Gewoon","Cluster"],"min"),
        ("lcs", "lcs",  ["Dominance","Dominance+Cluster","Gewoon","Cluster"],"max"),
        # ("misp", "misp", ["Dominance","Dominance+Cluster","Gewoon","Cluster"],"max"),
        ])
    
    # plot([
    #     ("tsptw", "tsptw/AFG",  ["BinaryGewoon","Gewoon","Cluster"],"min"),
    #     ("knapsack", "knapsack", ["BinaryGewoon","Gewoon","Cluster"],"max"),
    #     # ("sop", "sop", ["BinaryGewoon","Gewoon","Cluster"],"min"),
    #     ("misp", "misp", ["BinaryGewoon","Gewoon","Cluster"],"max"),
    #     ])

    # plot([
    #     # ("max2sat", "max2sat",  ["Cluster","VarOrd","Cluster+VarOrd","Gewoon"],"max"),
    #     ("knapsack", "knapsack", ["Cluster","VarOrd","Cluster+VarOrd","Gewoon"],"max"),
    #     ("misp", "misp",  ["Cluster","VarOrd","Cluster+VarOrd","Gewoon"],"max"),
    #     ])
    
    # plot([
    #     ("tsptw", "tsptw/AFG",  ["Dominance","Gewoon"],"min"),
    #     ("knapsack", "knapsack", ["Dominance","Gewoon"],"max"),
    #     ("sop", "sop", ["Dominance","Gewoon"],"min"),
    #     ("alp", "alp", ["Dominance","Gewoon"],"min"),
    #     ])
    
    # plot([
    #     ("knapsack", "knapsack_subset", ["BinaryConflict","BinaryGewoon"],"max"),
    #     ("knapsack", "knapsack_subset", ["BinaryConflict","BinaryGewoon"],"max"),
    #     ])

    # plot([
    #     ("knapsack", "knapsack", ["Dominance","VarOrd","Dominance+VarOrd","Gewoon"],"max"),
    #     ("misp", "misp", ["Dominance","VarOrd","Dominance+VarOrd","Gewoon"],"max"),
    #     ])
