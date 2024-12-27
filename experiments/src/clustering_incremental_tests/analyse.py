import json,argparse, os
from pathlib import Path
import pandas as pd


def main(args):
    # Open and read the JSON file
    pathlist = Path(args.input).glob("*")
    outputfile = open(args.output,'a')
    for file in pathlist:
        filename = os.fsdecode(file)
        with open(filename, 'r') as f:
            results = json.load(f)
            outputfile.write(f"""{filename.split("/")[-1].split(".")[0]}, \
            {float(results["Lower Bnd"])}, \
            {float(results["Upper Bnd"])}, \
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
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Choose full or abridged verification.')
    parser.add_argument('--input',
                        '-i', 
                        type=Path,
                        help='input folder to analyse results from')
    parser.add_argument('--output',
                        '-o',  
                        type=Path,
                        default='/home/eaeigbe/Documents/PhD/ddo/experiments/summary.txt',
                        help='output file')

    args = parser.parse_args()
    main(args)