#!/bin/bash

SAVEIFS=$IFS
IFS=$(echo -en "\n\b")


function run {
    problemname=$1
    folder=$2
    problemtype=$3
	experimenttype=$4
	F="resources/$folder/*"
    runmdd="./target/release/examples/$problemname"


	#### make directories if they dont exist
	# Define the directory path
	DIR1="experiments/results/$experimenttype"
	DIR2="experiments/results/$experimenttype/$folder"

	# Check if the directory does not exist
	if [ ! -d "$DIR1" ]; then
		# Directory does not exist, so create it
		mkdir "$DIR1"
	else
		IFS='/' read -ra ADDR <<< "$folder"
		DIR2=$DIR1
		for i in "${ADDR[@]}"; do
			# process "$i"
			DIR2="$DIR2/$i"
			if [ ! -d "$DIR2" ]; then
				mkdir "$DIR2"
			fi
		done
		
	fi

	# #first run oracle
	# rm -r experiments/results/$experimenttype/$folder/TD_B_n_B
	# mkdir experiments/results/$experimenttype/$folder/TD_B_n_B
	# for f in $F; do
	# 	echo -e "\n"$f
	# 	$runmdd $f -j -s BB -d 120 -x "experiments/results/$experimenttype/$folder/TD_B_n_B/"
	# done

	# # # for t_width in {20,50,100,200,500,1000}; do
	for t_width in {20,50,100}; do

		rm -r experiments/results/$experimenttype/$folder/TD_w_$t_width
		mkdir experiments/results/$experimenttype/$folder/TD_w_$t_width		

		for f in $F; do
			if [[ $experimenttype == "Cluster" ]]; then
				# $runmdd  $f -s IR -w $t_width -c -j -x "experiments/results/$experimenttype/$folder/c_1_w1_10_w2_$t_width/"
				# $runmdd  $f -s IR -w $t_width -j -x "experiments/results/$experimenttype/$folder/c_0_w1_10_w2_$t_width/"
				$runmdd  $f -s TD -w $t_width -j -k -x "experiments/results/$experimenttype/$folder/TD_w_$t_width/"
			
			elif [[ $experimenttype == "Dominance" ]]; then
				
				$runmdd  $f -s TD -w $t_width -j --dominance -x "experiments/results/$experimenttype/$folder/TD_w_$t_width/"
			
			elif [[ $experimenttype == "RUB" ]]; then
				
				$runmdd  $f -s TD -w $t_width -j --rub -x "experiments/results/$experimenttype/$folder/TD_w_$t_width/"
			
			elif [[ $experimenttype == "VarOrd" ]]; then
				
				$runmdd  $f -s TD -w $t_width -j --variable-order -x "experiments/results/$experimenttype/$folder/TD_w_$t_width/"
			
			elif [[ $experimenttype == "Cluster+VarOrd" ]]; then
				
				$runmdd  $f -s TD -w $t_width -j -k --variable-order -x "experiments/results/$experimenttype/$folder/TD_w_$t_width/"
			
			elif [[ $experimenttype == "Dominance+VarOrd" ]]; then
				
				$runmdd  $f -s TD -w $t_width -j --dominance --variable-order -x "experiments/results/$experimenttype/$folder/TD_w_$t_width/"
			
			elif [[ $experimenttype == "RUB+VarOrd" ]]; then
				
				$runmdd  $f -s TD -w $t_width -j --rub --variable-order -x "experiments/results/$experimenttype/$folder/TD_w_$t_width/"
			
			elif [[ $experimenttype == "RUB+Cluster" ]]; then
				
				$runmdd  $f -s TD -w $t_width -j --rub -k -x "experiments/results/$experimenttype/$folder/TD_w_$t_width/"
			
			elif [[ $experimenttype == "RUB+Dominance" ]]; then
				
				$runmdd  $f -s TD -w $t_width -j --rub --dominance -x "experiments/results/$experimenttype/$folder/TD_w_$t_width/"
			
			elif [[ $experimenttype == "Dominance+Cluster" ]]; then
				
				$runmdd  $f -s TD -w $t_width -j --dominance -k -x "experiments/results/$experimenttype/$folder/TD_w_$t_width/"
			
			elif [[ $experimenttype == "All" ]]; then
				
				$runmdd  $f -s TD -w $t_width -j --dominance --variable-order -k --rub -x "experiments/results/$experimenttype/$folder/TD_w_$t_width/"
			
			elif [[ $experimenttype == "Gewoon" ]]; then
				
				$runmdd  $f -s TD -w $t_width -j -x "experiments/results/$experimenttype/$folder/TD_w_$t_width/"
			
			else
				echo "unknown experiment setup"
			fi
			
		done

		rm experiments/results/$experimenttype/$folder/summary_w1_10_w2_$t_width.csv
		echo Name,Lower,Upper,Duration,Aborted,RefineCluster,CompileCluster,Dominance,Binary,Solver,Width,Gap,Objective  > experiments/results/$experimenttype/$folder/summary_w1_10_w2_$t_width.csv
		
		python experiments/src/clustering_incremental_tests/analyse.py -i experiments/results/$experimenttype/$folder/TD_w_$t_width -o  experiments/results/$experimenttype/$folder/summary_w1_10_w2_$t_width.csv
		python experiments/src/clustering_incremental_tests/analyse.py -i experiments/results/$experimenttype/$folder/TD_B_n_B -o  experiments/results/$experimenttype/$folder/summary_w1_10_w2_$t_width.csv
	done

		# python experiments/src/clustering_incremental_tests/plot.py -i experiments/results/$experimenttype/$folder/summary_w1_10_w2_20.csv,experiments/results/$experimenttype/$folder/summary_w1_10_w2_50.csv,experiments/results/$experimenttype/$folder/summary_w1_10_w2_100.csv,experiments/results/$experimenttype/$folder/summary_w1_10_w2_200.csv,experiments/results/$experimenttype/$folder/summary_w1_10_w2_500.csv \
        # -t $problemtype

		# python experiments/src/clustering_incremental_tests/plot.py -n {$problemname}TD_c_no_c.png\
		# -i experiments/results/$experimenttype/$folder/summary_w1_10_w2_20.csv,experiments/results/$experimenttype/$folder/summary_w1_10_w2_50.csv,experiments/results/$experimenttype/$folder/summary_w1_10_w2_100.csv \
        # -t $problemtype

}


# ##############
# run "talentsched" "talentsched" "min" "All"
# run "srflp" "srflp" "min" "All"
# run "tsptw" "tsptw/AFG" "min" "All"
# run "misp" "misp" "max" "All"
# run "sop" "sop" "min" "All"
# run "mcp" "mcp" "max" "All"
# run "knapsack" "knapsack" "max" "All"
# run "max2sat" "max2sat" "max" "All"
# run "psp" "psp/instancesWith2items" "min" "All"
# run "lcs" "lcs" "max" "All"
# ################

# ##############
# run "talentsched" "talentsched" "min" "Gewoon"
# run "srflp" "srflp" "min" "Gewoon"
# run "tsptw" "tsptw/AFG" "min" "Gewoon"
# run "misp" "misp" "max" "Gewoon"
# run "sop" "sop" "min" "Gewoon"
# run "mcp" "mcp" "max" "Gewoon"
# run "knapsack" "knapsack" "max" "Gewoon"
# run "max2sat" "max2sat" "max" "Gewoon"
# run "psp" "psp/instancesWith2items" "min" "Gewoon"
# run "lcs" "lcs" "max" "Gewoon"
# ################

# ##############
# run "talentsched" "talentsched" "min" "Cluster"
# run "srflp" "srflp" "min" "Cluster"
# run "tsptw" "tsptw/AFG" "min" "Cluster"
# run "misp" "misp" "max" "Cluster"
# run "sop" "sop" "min" "Cluster"
# run "mcp" "mcp" "max" "Cluster"
# run "knapsack" "knapsack" "max" "Cluster"
# run "max2sat" "max2sat" "max" "Cluster"
# run "psp" "psp/instancesWith2items" "min" "Cluster"
# run "lcs" "lcs" "max" "Cluster"
# ################

# ##############
# run "knapsack" "knapsack" "max" "Dominance"
# run "lcs" "lcs" "max" "Dominance"
# run "tsptw" "tsptw/AFG" "min" "Dominance"
# ################

# ##############
# run "knapsack" "knapsack" "max" "RUB"
# run "talentsched" "talentsched" "min" "RUB"
# run "sop" "sop" "min" "RUB"
# run "misp" "misp" "max" "RUB"
# run "max2sat" "max2sat" "max" "RUB"
# run "tsptw" "tsptw/AFG" "min" "RUB"
# ################

# ##############
# run "knapsack" "knapsack" "max" "VarOrd"
# run "misp" "misp" "max" "VarOrd"
# run "max2sat" "max2sat" "max" "VarOrd"
# ################

# ##############
# # ############## "Cluster+VarOrd"
# run "knapsack" "knapsack" "max" "Cluster+VarOrd"
# run "misp" "misp" "max" "Cluster+VarOrd"
# run "max2sat" "max2sat" "max" "Cluster+VarOrd"
# ##############

# ##############
# # ############## "Dominance+VarOrd" 
# run "knapsack" "knapsack" "max" "Dominance+VarOrd" 
# ##############

# ############### 
# # ############## "RUB+VarOrd"	
# run "knapsack" "knapsack" "max" "RUB+VarOrd"	
# run "misp" "misp" "max" "RUB+VarOrd"	
# run "max2sat" "max2sat" "max" "RUB+VarOrd"	
# ##############


# ##############
# # ############## "RUB+Cluster" 
# run "knapsack" "knapsack" "max" "RUB+Cluster" 
# run "talentsched" "talentsched" "min" "RUB+Cluster" 
# run "sop" "sop" "min" "RUB+Cluster" 
# run "misp" "misp" "max" "RUB+Cluster" 
# run "max2sat" "max2sat" "max" "RUB+Cluster" 
# run "tsptw" "tsptw/AFG" "min" "RUB+Cluster" 
# ##############

# ##############
# # ############## "RUB+Dominance"
# run "knapsack" "knapsack" "max" "RUB+Dominance"
# run "tsptw" "tsptw/AFG" "min" "RUB+Dominance" 
# ##############

# ##############
# # ############## "Dominance+Cluster" 
# run "knapsack" "knapsack" "max" "Dominance+Cluster" 
# run "lcs" "lcs" "max" "Dominance+Cluster" 
run "tsptw" "tsptw/AFG" "min" "Dominance+Cluster" 
# ##############
			