#!/bin/bash

SAVEIFS=$IFS
IFS=$(echo -en "\n\b")

function runOracle {
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
		IFS='/' read -ra ADDR <<< "$folder"
		DIR2=$DIR1
		for i in "${ADDR[@]}"; do
			# process "$i"
			DIR2=$DIR2/$i
			if [ ! -d "$DIR2" ]; then
				mkdir "$DIR2"
			fi
		done
	else
		IFS='/' read -ra ADDR <<< "$folder"
		DIR2=$DIR1
		for i in "${ADDR[@]}"; do
			# process "$i"
			DIR2=$DIR2/$i
			if [ ! -d "$DIR2" ]; then
				mkdir "$DIR2"
			fi
		done
		
	fi

	#first run oracle
	rm -r experiments/results/BnB/$folder/TD_B_n_B
	mkdir experiments/results/BnB/$folder/TD_B_n_B
	for f in $F; do
		echo -e "\n"$f
		$runmdd $f -j -s BB -d 120 -x "experiments/results/BnB/$folder/TD_B_n_B/"
	done
}

function run {
    problemname=$1
    folder=$2
    problemtype=$3
	experimenttype=$4
	solvertype=$5
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
		IFS='/' read -ra ADDR <<< "$folder"
		DIR2=$DIR1
		for i in "${ADDR[@]}"; do
			# process "$i"
			DIR2=$DIR2/$i
			if [ ! -d "$DIR2" ]; then
				mkdir "$DIR2"
			fi
		done
	else
		IFS='/' read -ra ADDR <<< "$folder"
		DIR2=$DIR1
		for i in "${ADDR[@]}"; do
			# process "$i"
			DIR2=$DIR2/$i
			if [ ! -d "$DIR2" ]; then
				mkdir "$DIR2"
			fi
		done
		
	fi

	# # # for t_width in {20,50,100,200,500,1000}; do
	# for t_width in {20,50,100,200,500,1000}; do
	for t_width in {20,50,100,200,500,1000}; do

		rm -r "experiments/results/$experimenttype/$folder/${solvertype}_w_${t_width}"
		mkdir "experiments/results/$experimenttype/$folder/${solvertype}_w_${t_width}"	

		for f in $F; do
			if [[ $experimenttype == "Cluster" ]]; then
				# $runmdd  $f -s IR -w $t_width -c -j -x "experiments/results/$experimenttype/$folder/c_1_w1_10_w2_$t_width/"
				# $runmdd  $f -s IR -w $t_width -j -x "experiments/results/$experimenttype/$folder/c_0_w1_10_w2_$t_width/"
				$runmdd  $f -s $solvertype -w $t_width -j -k -c -x "experiments/results/$experimenttype/$folder/${solvertype}_w_${t_width}/"

			elif [[ $experimenttype == "BinaryGewoon" ]]; then
				# $runmdd  $f -s IR -w $t_width -c -j -x "experiments/results/$experimenttype/$folder/c_1_w1_10_w2_$t_width/"
				# $runmdd  $f -s IR -w $t_width -j -x "experiments/results/$experimenttype/$folder/c_0_w1_10_w2_$t_width/"
				$runmdd  $f -s $solvertype -w $t_width -j --binary-split --rub -x "experiments/results/$experimenttype/$folder/${solvertype}_w_${t_width}/"
			
			elif [[ $experimenttype == "Dominance" ]]; then
				
				$runmdd  $f -s $solvertype -w $t_width -j --dominance -x "experiments/results/$experimenttype/$folder/${solvertype}_w_${t_width}/"
			
			elif [[ $experimenttype == "RUB" ]]; then
				
				$runmdd  $f -s $solvertype -w $t_width -j --rub -x "experiments/results/$experimenttype/$folder/${solvertype}_w_${t_width}/"
			
			elif [[ $experimenttype == "VarOrd" ]]; then
				
				$runmdd  $f -s $solvertype -w $t_width -j --variable-order -x "experiments/results/$experimenttype/$folder/${solvertype}_w_${t_width}/"
			
			elif [[ $experimenttype == "Cluster+VarOrd" ]]; then
				
				$runmdd  $f -s $solvertype -w $t_width -j -k -c --variable-order -x "experiments/results/$experimenttype/$folder/${solvertype}_w_${t_width}/"
			
			elif [[ $experimenttype == "Dominance+VarOrd" ]]; then
				
				$runmdd  $f -s $solvertype -w $t_width -j --dominance --variable-order -x "experiments/results/$experimenttype/$folder/${solvertype}_w_${t_width}/"
			
			elif [[ $experimenttype == "RUB+VarOrd" ]]; then
				
				$runmdd  $f -s $solvertype -w $t_width -j --rub --variable-order -x "experiments/results/$experimenttype/$folder/${solvertype}_w_${t_width}/"
			
			elif [[ $experimenttype == "RUB+Cluster" ]]; then
				
				$runmdd  $f -s $solvertype -w $t_width -j --rub -k -c -x "experiments/results/$experimenttype/$folder/${solvertype}_w_${t_width}/"
			
			elif [[ $experimenttype == "RUB+Dominance" ]]; then
				
				$runmdd  $f -s $solvertype -w $t_width -j --rub --dominance -x "experiments/results/$experimenttype/$folder/${solvertype}_w_${t_width}/"
			
			elif [[ $experimenttype == "Dominance+Cluster" ]]; then
				
				$runmdd  $f -s $solvertype -w $t_width -j --dominance -k -c -x "experiments/results/$experimenttype/$folder/${solvertype}_w_${t_width}/"
			
			elif [[ $experimenttype == "All" ]]; then
				
				$runmdd  $f -s $solvertype -w $t_width -j --dominance --variable-order -k --rub -x "experiments/results/$experimenttype/$folder/${solvertype}_w_${t_width}/"
			
			elif [[ $experimenttype == "Gewoon" ]]; then
				
				$runmdd  $f -s $solvertype -w $t_width -j -x "experiments/results/$experimenttype/$folder/${solvertype}_w_${t_width}/"
			
			else
				echo "unknown experiment setup"
			fi
			
		done

		rm experiments/results/$experimenttype/$folder/summary_w1_10_w2_$t_width.csv
		echo Name,Lower,Upper,Duration,Aborted,RefineCluster,CompileCluster,Dominance,MergeQuality,Binary,Solver,Width,Gap,Objective  > experiments/results/$experimenttype/$folder/summary_w1_10_w2_$t_width.csv
		
		python3 experiments/src/clustering_incremental_tests/analyse.py -i "experiments/results/$experimenttype/$folder/${solvertype}_w_${t_width}" -o  experiments/results/$experimenttype/$folder/summary_w1_10_w2_$t_width.csv
		python3 experiments/src/clustering_incremental_tests/analyse.py -i experiments/results/BnB/$folder/TD_B_n_B -o  experiments/results/$experimenttype/$folder/summary_w1_10_w2_$t_width.csv
	done

}


function runConflictCount {
    problemname=$1
    folder=$2
    problemtype=$3
	experimenttype=$4
	solvertype=$5
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
		IFS='/' read -ra ADDR <<< "$folder"
		DIR2=$DIR1
		for i in "${ADDR[@]}"; do
			# process "$i"
			DIR2=$DIR2/$i
			if [ ! -d "$DIR2" ]; then
				mkdir "$DIR2"
			fi
		done
	else
		IFS='/' read -ra ADDR <<< "$folder"
		DIR2=$DIR1
		for i in "${ADDR[@]}"; do
			# process "$i"
			DIR2=$DIR2/$i
			if [ ! -d "$DIR2" ]; then
				mkdir "$DIR2"
			fi
		done
		
	fi

	# # # for t_width in {20,50,100,200,500,1000}; do
	# for t_width in {20,50,100,200,500,1000}; do
	for t_width in {20,50,100,200,500,1000}; do

		rm -r "experiments/results/$experimenttype/$folder/${solvertype}_w_${t_width}"
		mkdir "experiments/results/$experimenttype/$folder/${solvertype}_w_${t_width}"	

		for f in $F; do
			if [[ $experimenttype == "BinaryGewoon" ]]; then
				$runmdd  $f -s $solvertype -w $t_width -j --binary-split --rub -x "experiments/results/$experimenttype/$folder/${solvertype}_w_${t_width}/"
			
			elif [[ $experimenttype == "BinaryConflict" ]]; then
				
				$runmdd  $f -s $solvertype -w $t_width -j --binary-split --conflict-count --rub -x "experiments/results/$experimenttype/$folder/${solvertype}_w_${t_width}/"
			
			else
				echo "unknown experiment setup"
			fi
			
		done

		rm experiments/results/$experimenttype/$folder/summary_w1_10_w2_$t_width.csv
		echo Name,Lower,Upper,Duration,Aborted,RefineCluster,CompileCluster,Dominance,MergeQuality,Binary,Solver,Width,Gap,Objective  > experiments/results/$experimenttype/$folder/summary_w1_10_w2_$t_width.csv
		
		python3 experiments/src/clustering_incremental_tests/analyse.py -i "experiments/results/$experimenttype/$folder/${solvertype}_w_${t_width}" -o  experiments/results/$experimenttype/$folder/summary_w1_10_w2_$t_width.csv
		python3 experiments/src/clustering_incremental_tests/analyse.py -i experiments/results/BnB/$folder/TD_B_n_B -o  experiments/results/$experimenttype/$folder/summary_w1_10_w2_$t_width.csv
	done

}

################################################################################################# TD experiments
# ##############
# run "talentsched" "talentsched" "min" "All" "TD"
# run "srflp" "srflp" "min" "All" "TD"
# run "tsptw" "tsptw/AFG" "min" "All" "TD"
# run "misp" "misp" "max" "All" "TD"
# run "sop" "sop" "min" "All" "TD"
# run "mcp" "mcp" "max" "All" "TD"
# run "knapsack" "knapsack" "max" "All" "TD"
# run "max2sat" "max2sat" "max" "All" "TD"
# run "psp" "psp/instancesWith2items" "min" "All" "TD"
# run "lcs" "lcs" "max" "All" "TD"
# ################

##############
# run "talentsched" "talentsched" "min" "Gewoon" "TD"
# run "srflp" "srflp" "min" "Gewoon" "TD"
# run "tsptw" "tsptw/AFG" "min" "Gewoon" "TD"
# run "misp" "misp" "max" "Gewoon" "TD"
run "sop" "sop" "min" "Gewoon" "TD"
# run "mcp" "mcp" "max" "Gewoon" "TD"
run "knapsack" "knapsack" "max" "Gewoon" "TD"
# run "max2sat" "max2sat" "max" "Gewoon" "TD"
# run "psp" "psp/instancesWith2items" "min" "Gewoon" "TD"
run "lcs" "lcs" "max" "Gewoon" "TD"
run "alp" "alp" "min" "Gewoon" "TD"
################

# #############
# run "talentsched" "talentsched" "min" "Cluster" "TD"
# run "srflp" "srflp" "min" "Cluster" "TD"
# run "tsptw" "tsptw/AFG" "min" "Cluster" "TD"
# run "misp" "misp" "max" "Cluster" "TD"
run "sop" "sop" "min" "Cluster" "TD"
# run "mcp" "mcp" "max" "Cluster" "TD"
run "knapsack" "knapsack" "max" "Cluster" "TD"
# run "max2sat" "max2sat" "max" "Cluster" "TD"
# run "psp" "psp/instancesWith2items" "min" "Cluster" "TD"
run "lcs" "lcs" "max" "Cluster" "TD"
run "alp" "alp" "min" "Cluster" "TD"
# ###############

##############
# run "knapsack" "knapsack" "max" "Dominance" "TD"
run "lcs" "lcs" "max" "Dominance" "TD"
# run "tsptw" "tsptw/AFG" "min" "Dominance" "TD"
# run "sop" "sop" "min" "Dominance" "TD"
# run "alp" "alp" "min" "Dominance" "TD"
# run "misp" "misp" "max" "Dominance" "TD"
################

##############
# run "knapsack" "knapsack" "max" "RUB" "TD"
# # run "talentsched" "talentsched" "min" "RUB" "TD"
# # run "sop" "sop" "min" "RUB" "TD"
# run "misp" "misp" "max" "RUB" "TD"
# run "max2sat" "max2sat" "max" "RUB" "TD"
# # run "tsptw" "tsptw/AFG" "min" "RUB" "TD"
################

##############
# run "knapsack" "knapsack" "max" "VarOrd" "TD"
# run "misp" "misp" "max" "VarOrd" "TD"
# run "max2sat" "max2sat" "max" "VarOrd" "TD"
################

##############
# ############## "Cluster+VarOrd"
# run "knapsack" "knapsack" "max" "Cluster+VarOrd" "TD"
# run "misp" "misp" "max" "Cluster+VarOrd" "TD"
# run "max2sat" "max2sat" "max" "Cluster+VarOrd" "TD"
##############

##############
# ############## "Dominance+VarOrd" 
# run "knapsack" "knapsack" "max" "Dominance+VarOrd" "TD"
# run "misp" "misp" "max" "Dominance+VarOrd" "TD"
##############

############### 
# ############## "RUB+VarOrd"	
# run "knapsack" "knapsack" "max" "RUB+VarOrd" "TD"
# run "misp" "misp" "max" "RUB+VarOrd" "TD"	 
# run "max2sat" "max2sat" "max" "RUB+VarOrd" "TD"	
##############


##############
# ############## "RUB+Cluster" 
# run "knapsack" "knapsack" "max" "RUB+Cluster" "TD" 
# # run "talentsched" "talentsched" "min" "RUB+Cluster" "TD" 
# # run "sop" "sop" "min" "RUB+Cluster" "TD"
# run "misp" "misp" "max" "RUB+Cluster" "TD"
# run "max2sat" "max2sat" "max" "RUB+Cluster" "TD"
# # run "tsptw" "tsptw/AFG" "min" "RUB+Cluster" "TD"
##############

# ##############
# # ############## "RUB+Dominance"
# run "knapsack" "knapsack" "max" "RUB+Dominance" "TD"
# run "tsptw" "tsptw/AFG" "min" "RUB+Dominance" "TD"
# ##############

# ##############
# # ############## "Dominance+Cluster" 
# run "knapsack" "knapsack" "max" "Dominance+Cluster" "TD"
run "lcs" "lcs" "max" "Dominance+Cluster" "TD"
# run "tsptw" "tsptw/AFG" "min" "Dominance+Cluster" "TD"
# run "sop" "sop" "min" "Dominance+Cluster" "TD"
# run "alp" "alp" "min" "Dominance+Cluster" "TD"
# ##############
################################################################################################# TD experiments




################################################################################################# IR experiments
# #############
# run "tsptw" "tsptw/AFG" "min" "Gewoon" "IR"
# run "misp" "misp" "max" "Gewoon" "IR"
# run "sop" "sop" "min" "Gewoon" "IR"
# run "knapsack" "knapsack" "max" "Gewoon" "IR"
# ###############

# #############
# run "tsptw" "tsptw/AFG" "min" "BinaryGewoon" "IR"
# run "misp" "misp" "max" "BinaryGewoon" "IR"
# run "sop" "sop" "min" "BinaryGewoon" "IR"
# run "knapsack" "knapsack" "max" "BinaryGewoon" "IR"
# ###############

# #############
# run "tsptw" "tsptw/AFG" "min" "Cluster" "IR"
# run "misp" "misp" "max" "Cluster" "IR"
# run "sop" "sop" "min" "Cluster" "IR"
# run "knapsack" "knapsack" "max" "Cluster" "IR"
# ###############

# #############
# run "tsptw" "tsptw/AFG" "min" "Dominance" "IR"
# run "sop" "sop" "min" "Dominance" "IR"
# run "knapsack" "knapsack" "max" "Dominance" "IR"
# ###############

# ##############
# # ############## "Dominance+Cluster" 
# run "tsptw" "tsptw/AFG" "min" "Dominance+Cluster"  "IR"
# run "sop" "sop" "min" "Dominance+Cluster"  "IR"
# run "knapsack" "knapsack" "max" "Dominance+Cluster"  "IR"
# ##############
################################################################################################# IR experiments




##############
# runOracle "talentsched" "talentsched" "min" "All"
# runOracle "srflp" "srflp" "min" "All"
# runOracle "tsptw" "tsptw/AFG" "min" "All"
# runOracle "misp" "misp" "max" "All"
# runOracle "sop" "sop" "min" "All"
# runOracle "mcp" "mcp" "max" "All"
# runOracle "knapsack" "knapsack" "max" "All"
# runOracle "max2sat" "max2sat" "max" "All"
# runOracle "psp" "psp/instancesWith2items" "min" "All"
runOracle "lcs" "lcs" "max" "All"
################

# runConflictCount "knapsack" "knapsack_subset" "max" "BinaryGewoon" "IR"
# runConflictCount "knapsack" "knapsack_subset" "max" "BinaryConflict" "IR"