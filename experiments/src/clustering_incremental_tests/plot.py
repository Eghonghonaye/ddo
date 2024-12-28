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
    # save_path = files[0].split("summary")[0]

    # fig, axs = plt.subplots(ncols=len(args.input))
    all_df = []
    for index,file in enumerate(files):
        df = pd.read_csv(file,header=0)
        print(file)

        if type == "max":
            df["MergeQuality"] = df["MergeQuality"].astype(float)
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
            df["MergeQuality"] = df["MergeQuality"].astype(float)
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
    


def plot(problem_names):
    all_df = []
    for (problem,subfolder,experiments,type) in problem_names:
        for folder in experiments:
            print(problem)
            path = f"/home/eaeigbe/Documents/PhD/ddo/experiments/results/{folder}/{subfolder}"
            filenames = [ path+"/"+filename for filename in listdir(path) if filename.endswith( "csv" ) ]
            df = plot_bound_width(filenames,type)
            df["Problem"] = problem
            df["Conditions"] = folder
            df['Label'] = df['Solver'] + "-->" + df['Conditions']
            # print(df.to_string())
            all_df.append(df)

    all_df = pd.concat(all_df)
    all_df.reset_index(level=None, drop=False, inplace=True, col_level=0, col_fill="")

    # print(all_df.to_string())
    all_df = all_df.drop(all_df[all_df["Solver"] == 'incremental'].index)
    all_df = all_df.drop(all_df[all_df["Solver"] == 'branch-bound'].index)
    all_df["MergeQuality"] =  all_df["MergeQuality"].astype(float)
    all_df = all_df.dropna()
    print(all_df.to_string())

    def joint_plot(data: pd.DataFrame, x_name: str, y1_name: str, y2_name: str, ylabel1: str, ylabel2: str,color="blue",label=0):
        ax1 = plt.gca()
        sns.lineplot(x=data[x_name], y=data[y1_name], alpha=.7, color='red',ax=ax1)
        ax1.set_ylabel(ylabel1)

        ax2 = ax1.twinx()
        sns.lineplot(x=data[x_name], y=data[y2_name], alpha=.7,ax=ax2)
        ax2.set_ylabel(ylabel2)


    g = sns.FacetGrid(all_df, col="Problem", col_wrap=2)
    # g.map_dataframe(sns.lineplot,x="Width", y="RealGap", hue="Label").add_legend() 
    g.map_dataframe(joint_plot, x_name="Width", y1_name="RealGap", y2_name="MergeQuality",
                    ylabel1="Normalised Bound", ylabel2="Merge Error")
    g.set_ylabels("Normalised Bound")
    g.set_xlabels("Width")
    plt.show() 
    # plt.savefig(f'{experiments}plot.png')

    # ax =  sns.lineplot(x="Width", y="RealGap", hue="Label", data=all_df)
    # ax2 = ax.twinx()
    # sns.lineplot(x="Width", y="MergeQuality", hue="Label", ax=ax2, data=all_df)
    # # ax.figure.legend()
    # plt.show()


if __name__ == "__main__":


    ["Cluster","Dominance","RUB","VarOrd" ,
     "Cluster+VarOrd","Dominance+VarOrd","RUB+VarOrd",
     "RUB+Cluster","RUB+Dominance" ,"Dominance+Cluster",
      "All","Gewoon"]
    
    ["Dominance","VarOrd","Dominance+VarOrd","Gewoon"]
    ["VarOrd","Gewoon"]
			
    #RUB does nothing??? even in tsptw?


    # #Dominance
    # plot([
    #     ("talentsched", "talentsched", ["Gewoon"],"min"),
    #     ("srflp", "srflp", ["Gewoon"],"min"),
    #     ("tsptw", "tsptw/AFG",  ["Dominance","Gewoon"],"min"),
    #     ("misp", "misp", ["VarOrd","Gewoon"],"max"),
    #     ("sop", "sop", ["Gewoon"],"min"),
    #     ("mcp", "mcp", ["Gewoon"],"max"),
    #     ("knapsack", "knapsack", ["Dominance","VarOrd","Dominance+VarOrd","Gewoon"],"max"),
    #     ("max2sat", "max2sat", ["VarOrd","Gewoon"],"max"),
    #     ("psp", "psp/instancesWith2items", ["Gewoon"],"min"),
    #     ("lcs", "lcs",  ["Dominance","Gewoon"],"max"),
    #     ])
    
    # #VarOrd
    # plot([
    #     ("talentsched", "talentsched", ["Gewoon"],"min"),
    #     ("srflp", "srflp", ["Gewoon"],"min"),
    #     ("tsptw", "tsptw/AFG",  ["Dominance","Gewoon"],"min"),
    #     ("misp", "misp", ["VarOrd","Gewoon"],"max"),
    #     ("sop", "sop", ["Gewoon"],"min"),
    #     ("mcp", "mcp", ["Gewoon"],"max"),
    #     ("knapsack", "knapsack", ["Dominance","VarOrd","Dominance+VarOrd","Gewoon"],"max"),
    #     ("max2sat", "max2sat", ["VarOrd","Gewoon"],"max"),
    #     ("psp", "psp/instancesWith2items", ["Gewoon"],"min"),
    #     ("lcs", "lcs",  ["Dominance","Gewoon"],"max"),
    #     ])
    
    # #RUB
    # plot([
    #     ("talentsched", "talentsched", ["Gewoon"],"min"),
    #     ("srflp", "srflp", ["Gewoon"],"min"),
    #     ("tsptw", "tsptw/AFG",  ["Dominance","Gewoon"],"min"),
    #     ("misp", "misp", ["VarOrd","Gewoon"],"max"),
    #     ("sop", "sop", ["Gewoon"],"min"),
    #     ("mcp", "mcp", ["Gewoon"],"max"),
    #     ("knapsack", "knapsack", ["Dominance","VarOrd","Dominance+VarOrd","Gewoon"],"max"),
    #     ("max2sat", "max2sat", ["VarOrd","Gewoon"],"max"),
    #     ("psp", "psp/instancesWith2items", ["Gewoon"],"min"),
    #     ("lcs", "lcs",  ["Dominance","Gewoon"],"max"),
    #     ])
    

    # plot([
    #     # ("talentsched", "talentsched", ["Gewoon"],"min"),
    #     # ("srflp", "srflp", ["Gewoon"],"min"),
    #     # ("tsptw", "tsptw/AFG",  ["Dominance","Gewoon"],"min"),
    #     # ("misp", "misp", ["VarOrd","Gewoon"],"max"),
    #     # ("sop", "sop", ["Gewoon"],"min"),
    #     # ("mcp", "mcp", ["Gewoon"],"max"),
    #     ("knapsack", "knapsack", ["Gewoon"],"max"),
    #     # ("max2sat", "max2sat", ["VarOrd","Gewoon"],"max"),
    #     # ("psp", "psp/instancesWith2items", ["Gewoon"],"min"),
    #     # ("lcs", "lcs",  ["Dominance","Gewoon"],"max"),
    #     ])

    plot([
        ("talentsched", "talentsched", ["Gewoon"],"min"),
        ("srflp", "srflp", ["Gewoon"],"min"),
        ("tsptw", "tsptw/AFG",  ["Gewoon"],"min"),
        ("misp", "misp", ["Gewoon"],"max"),
        ("sop", "sop", ["Gewoon"],"min"),
        ("mcp", "mcp", ["Gewoon"],"max"),
        ("knapsack", "knapsack", ["Gewoon"],"max"),
        ("max2sat", "max2sat", ["Gewoon"],"max"),
        ("psp", "psp/instancesWith2items", ["Gewoon"],"min"),
        ("lcs", "lcs",  ["Gewoon"],"max"),
        ])
    
    # plot([
    #     ("talentsched", "talentsched", ["Cluster"],"min"),
    #     ("srflp", "srflp", ["Cluster"],"min"),
    #     ("tsptw", "tsptw/AFG",  ["Cluster"],"min"),
    #     ("misp", "misp", ["Cluster"],"max"),
    #     ("sop", "sop", ["Cluster"],"min"),
    #     ("mcp", "mcp", ["Cluster"],"max"),
    #     ("knapsack", "knapsack", ["Cluster"],"max"),
    #     ("max2sat", "max2sat", ["Cluster"],"max"),
    #     ("psp", "psp/instancesWith2items", ["Cluster"],"min"),
    #     ("lcs", "lcs",  ["Cluster"],"max"),
    #     ])
    
    # plot([
    #     ("talentsched", "talentsched", ["Gewoon","Cluster"],"min"),
    #     ("srflp", "srflp", ["Gewoon","Cluster"],"min"),
    #     ("tsptw", "tsptw/AFG",  ["Gewoon","Cluster"],"min"),
    #     ("misp", "misp", ["Gewoon","Cluster"],"max"),
    #     ("sop", "sop", ["Gewoon","Cluster"],"min"),
    #     ("mcp", "mcp", ["Gewoon","Cluster"],"max"),
    #     ("knapsack", "knapsack", ["Gewoon","Cluster"],"max"),
    #     ("max2sat", "max2sat", ["Gewoon","Cluster"],"max"),
    #     ("psp", "psp/instancesWith2items", ["Gewoon","Cluster"],"min"),
    #     ("lcs", "lcs",  ["Gewoon","Cluster"],"max"),
    #     ])


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