#!/bin/bash

SAVEIFS=$IFS
IFS=$(echo -en "\n\b")

function knapsack {
	F="/home/eaeigbe/Documents/PhD/ddo/resources/knapsack_subset/*"

	# #first run oracle
	# rm -r /home/eaeigbe/Documents/PhD/ddo/experiments/results/knapsack/TD_B_n_B
	# mkdir /home/eaeigbe/Documents/PhD/ddo/experiments/results/knapsack/TD_B_n_B
	# for f in $F; do
	# 	echo -e "\n"$f
	# 	./target/release/examples/knapsack $f -j -s BB -d 30 -x "/home/eaeigbe/Documents/PhD/ddo/experiments/results/knapsack/TD_B_n_B/"
	# done

	for t_width in {20,50,100,200,500,1000}; do
	# for t_width in {200,500}; do
		rm -r /home/eaeigbe/Documents/PhD/ddo/experiments/results/knapsack/c_1_w1_10_w2_$t_width
		rm -r /home/eaeigbe/Documents/PhD/ddo/experiments/results/knapsack/c_1_b_1_w1_10_w2_$t_width
		rm -r /home/eaeigbe/Documents/PhD/ddo/experiments/results/knapsack/c_0_w1_10_w2_$t_width
		rm -r /home/eaeigbe/Documents/PhD/ddo/experiments/results/knapsack/TD_w_$t_width
		rm -r /home/eaeigbe/Documents/PhD/ddo/experiments/results/knapsack/TD_c_w_$t_width
		

		mkdir /home/eaeigbe/Documents/PhD/ddo/experiments/results/knapsack/c_1_w1_10_w2_$t_width
		mkdir /home/eaeigbe/Documents/PhD/ddo/experiments/results/knapsack/c_1_b_1_w1_10_w2_$t_width
		mkdir /home/eaeigbe/Documents/PhD/ddo/experiments/results/knapsack/c_0_w1_10_w2_$t_width
		mkdir /home/eaeigbe/Documents/PhD/ddo/experiments/results/knapsack/TD_w_$t_width
		mkdir /home/eaeigbe/Documents/PhD/ddo/experiments/results/knapsack/TD_c_w_$t_width
		

		for f in $F; do
			echo -e "\n"$f
			./target/release/examples/knapsack $f -s IR -w $t_width -c -j -x "/home/eaeigbe/Documents/PhD/ddo/experiments/results/knapsack/c_1_w1_10_w2_$t_width/"
			./target/release/examples/knapsack $f -s IR -w $t_width -c -b -j -x "/home/eaeigbe/Documents/PhD/ddo/experiments/results/knapsack/c_1_b_1_w1_10_w2_$t_width/"
			./target/release/examples/knapsack $f -s IR -w $t_width -j -x "/home/eaeigbe/Documents/PhD/ddo/experiments/results/knapsack/c_0_w1_10_w2_$t_width/"
			./target/release/examples/knapsack $f -s TD -w $t_width -j -x "/home/eaeigbe/Documents/PhD/ddo/experiments/results/knapsack/TD_w_$t_width/"
			./target/release/examples/knapsack $f -s TD -w $t_width -j -k -x "/home/eaeigbe/Documents/PhD/ddo/experiments/results/knapsack/TD_c_w_$t_width/"
					
		done

		rm /home/eaeigbe/Documents/PhD/ddo/experiments/results/knapsack/summary_w1_10_w2_$t_width.csv
		echo Name,Lower,Upper,Duration,Aborted,RefineCluster,CompileCluster,Binary,Solver,Width,Gap,Objective  > /home/eaeigbe/Documents/PhD/ddo/experiments/results/knapsack/summary_w1_10_w2_$t_width.csv
		
		python /home/eaeigbe/Documents/PhD/ddo/experiments/src/clustering_incremental_tests/analyse.py -i /home/eaeigbe/Documents/PhD/ddo/experiments/results/knapsack/c_0_w1_10_w2_$t_width -o  /home/eaeigbe/Documents/PhD/ddo/experiments/results/knapsack/summary_w1_10_w2_$t_width.csv
		python /home/eaeigbe/Documents/PhD/ddo/experiments/src/clustering_incremental_tests/analyse.py -i /home/eaeigbe/Documents/PhD/ddo/experiments/results/knapsack/c_1_w1_10_w2_$t_width -o  /home/eaeigbe/Documents/PhD/ddo/experiments/results/knapsack/summary_w1_10_w2_$t_width.csv
		python /home/eaeigbe/Documents/PhD/ddo/experiments/src/clustering_incremental_tests/analyse.py -i /home/eaeigbe/Documents/PhD/ddo/experiments/results/knapsack/c_1_b_1_w1_10_w2_$t_width -o  /home/eaeigbe/Documents/PhD/ddo/experiments/results/knapsack/summary_w1_10_w2_$t_width.csv
		python /home/eaeigbe/Documents/PhD/ddo/experiments/src/clustering_incremental_tests/analyse.py -i /home/eaeigbe/Documents/PhD/ddo/experiments/results/knapsack/TD_w_$t_width -o  /home/eaeigbe/Documents/PhD/ddo/experiments/results/knapsack/summary_w1_10_w2_$t_width.csv
		python /home/eaeigbe/Documents/PhD/ddo/experiments/src/clustering_incremental_tests/analyse.py -i /home/eaeigbe/Documents/PhD/ddo/experiments/results/knapsack/TD_c_w_$t_width -o  /home/eaeigbe/Documents/PhD/ddo/experiments/results/knapsack/summary_w1_10_w2_$t_width.csv
		python /home/eaeigbe/Documents/PhD/ddo/experiments/src/clustering_incremental_tests/analyse.py -i /home/eaeigbe/Documents/PhD/ddo/experiments/results/knapsack/TD_B_n_B -o  /home/eaeigbe/Documents/PhD/ddo/experiments/results/knapsack/summary_w1_10_w2_$t_width.csv
	done

	python /home/eaeigbe/Documents/PhD/ddo/experiments/src/clustering_incremental_tests/plot.py -i /home/eaeigbe/Documents/PhD/ddo/experiments/results/knapsack/summary_w1_10_w2_20.csv,/home/eaeigbe/Documents/PhD/ddo/experiments/results/knapsack/summary_w1_10_w2_50.csv,/home/eaeigbe/Documents/PhD/ddo/experiments/results/knapsack/summary_w1_10_w2_100.csv,/home/eaeigbe/Documents/PhD/ddo/experiments/results/knapsack/summary_w1_10_w2_200.csv -t max

}

function sop {
	F="/home/eaeigbe/Documents/PhD/ddo/resources/sop/*"

	#first run oracle
	rm -r /home/eaeigbe/Documents/PhD/ddo/experiments/results/sop/TD_B_n_B
	mkdir /home/eaeigbe/Documents/PhD/ddo/experiments/results/sop/TD_B_n_B
	for f in $F; do
		echo -e "\n"$f
		./target/release/examples/sop $f -j -s BB -d 30 -x "/home/eaeigbe/Documents/PhD/ddo/experiments/results/sop/TD_B_n_B/"
	done

	# for t_width in {20,50,100,200,500,1000}; do
	for t_width in {20,50,100,200}; do
		# rm -r /home/eaeigbe/Documents/PhD/ddo/experiments/results/sop/c_1_w1_10_w2_$t_width
		# rm -r /home/eaeigbe/Documents/PhD/ddo/experiments/results/sop/c_0_w1_10_w2_$t_width
		rm -r /home/eaeigbe/Documents/PhD/ddo/experiments/results/sop/TD_w_$t_width
		rm -r /home/eaeigbe/Documents/PhD/ddo/experiments/results/sop/TD_c_w_$t_width
		

		# mkdir /home/eaeigbe/Documents/PhD/ddo/experiments/results/sop/c_1_w1_10_w2_$t_width
		# mkdir /home/eaeigbe/Documents/PhD/ddo/experiments/results/sop/c_0_w1_10_w2_$t_width
		mkdir /home/eaeigbe/Documents/PhD/ddo/experiments/results/sop/TD_w_$t_width
		mkdir /home/eaeigbe/Documents/PhD/ddo/experiments/results/sop/TD_c_w_$t_width
		

		for f in $F; do
			echo -e "\n"$f
			# ./target/release/examples/sop $f -s IR -w $t_width -c -j -x "/home/eaeigbe/Documents/PhD/ddo/experiments/results/sop/c_1_w1_10_w2_$t_width/"
			# ./target/release/examples/sop $f -s IR -w $t_width -j -x "/home/eaeigbe/Documents/PhD/ddo/experiments/results/sop/c_0_w1_10_w2_$t_width/"
			./target/release/examples/sop $f -s TD -w $t_width -j -x "/home/eaeigbe/Documents/PhD/ddo/experiments/results/sop/TD_w_$t_width/"
			./target/release/examples/sop $f -s TD -w $t_width -j -k -x "/home/eaeigbe/Documents/PhD/ddo/experiments/results/sop/TD_c_w_$t_width/"
		done

		rm /home/eaeigbe/Documents/PhD/ddo/experiments/results/sop/summary_w1_10_w2_$t_width.csv
		echo Name,Lower,Upper,Duration,Aborted,RefineCluster,CompileCluster,Binary,Solver,Width,Gap,Objective  > /home/eaeigbe/Documents/PhD/ddo/experiments/results/sop/summary_w1_10_w2_$t_width.csv
		
		# python /home/eaeigbe/Documents/PhD/ddo/experiments/src/clustering_incremental_tests/analyse.py -i /home/eaeigbe/Documents/PhD/ddo/experiments/results/sop/c_0_w1_10_w2_$t_width -o  /home/eaeigbe/Documents/PhD/ddo/experiments/results/sop/summary_w1_10_w2_$t_width.csv
		# python /home/eaeigbe/Documents/PhD/ddo/experiments/src/clustering_incremental_tests/analyse.py -i /home/eaeigbe/Documents/PhD/ddo/experiments/results/sop/c_1_w1_10_w2_$t_width -o  /home/eaeigbe/Documents/PhD/ddo/experiments/results/sop/summary_w1_10_w2_$t_width.csv
		python /home/eaeigbe/Documents/PhD/ddo/experiments/src/clustering_incremental_tests/analyse.py -i /home/eaeigbe/Documents/PhD/ddo/experiments/results/sop/TD_w_$t_width -o  /home/eaeigbe/Documents/PhD/ddo/experiments/results/sop/summary_w1_10_w2_$t_width.csv
		python /home/eaeigbe/Documents/PhD/ddo/experiments/src/clustering_incremental_tests/analyse.py -i /home/eaeigbe/Documents/PhD/ddo/experiments/results/sop/TD_c_w_$t_width -o  /home/eaeigbe/Documents/PhD/ddo/experiments/results/sop/summary_w1_10_w2_$t_width.csv
		python /home/eaeigbe/Documents/PhD/ddo/experiments/src/clustering_incremental_tests/analyse.py -i /home/eaeigbe/Documents/PhD/ddo/experiments/results/sop/TD_B_n_B -o  /home/eaeigbe/Documents/PhD/ddo/experiments/results/sop/summary_w1_10_w2_$t_width.csv
	done

		python /home/eaeigbe/Documents/PhD/ddo/experiments/src/clustering_incremental_tests/plot.py -i /home/eaeigbe/Documents/PhD/ddo/experiments/results/sop/summary_w1_10_w2_20.csv,/home/eaeigbe/Documents/PhD/ddo/experiments/results/sop/summary_w1_10_w2_50.csv,/home/eaeigbe/Documents/PhD/ddo/experiments/results/sop/summary_w1_10_w2_100.csv,/home/eaeigbe/Documents/PhD/ddo/experiments/results/sop/summary_w1_10_w2_200.csv -t min

}

function tsptw {
	F="/home/eaeigbe/Documents/PhD/ddo/resources/tsptw/AFG/*"

	# #first run oracle
	# rm -r /home/eaeigbe/Documents/PhD/ddo/experiments/results/tsptw/AFG/TD_B_n_B
	# mkdir /home/eaeigbe/Documents/PhD/ddo/experiments/results/tsptw/AFG/TD_B_n_B
	# for f in $F; do
	# 	echo -e "\n"$f
	# 	./target/release/examples/tsptw $f -j -s BB -d 30 -x "/home/eaeigbe/Documents/PhD/ddo/experiments/results/tsptw/AFG/TD_B_n_B/"
	# done

	# for t_width in {20,50,100,200,500,1000}; do
	for t_width in {20,50,100,200}; do
		rm -r /home/eaeigbe/Documents/PhD/ddo/experiments/results/tsptw/AFG/c_1_w1_10_w2_$t_width
		rm -r /home/eaeigbe/Documents/PhD/ddo/experiments/results/tsptw/AFG/c_0_w1_10_w2_$t_width
		rm -r /home/eaeigbe/Documents/PhD/ddo/experiments/results/tsptw/AFG/TD_w_$t_width
		rm -r /home/eaeigbe/Documents/PhD/ddo/experiments/results/tsptw/AFG/TD_c_w_$t_width
		

		mkdir /home/eaeigbe/Documents/PhD/ddo/experiments/results/tsptw/AFG/c_1_w1_10_w2_$t_width
		mkdir /home/eaeigbe/Documents/PhD/ddo/experiments/results/tsptw/AFG/c_0_w1_10_w2_$t_width
		mkdir /home/eaeigbe/Documents/PhD/ddo/experiments/results/tsptw/AFG/TD_w_$t_width
		mkdir /home/eaeigbe/Documents/PhD/ddo/experiments/results/tsptw/AFG/TD_c_w_$t_width
		

		for f in $F; do
			echo -e "\n"$f
			./target/release/examples/tsptw $f -s IR -w $t_width -c -j -x "/home/eaeigbe/Documents/PhD/ddo/experiments/results/tsptw/AFG/c_1_w1_10_w2_$t_width/"
			./target/release/examples/tsptw $f -s IR -w $t_width -j -x "/home/eaeigbe/Documents/PhD/ddo/experiments/results/tsptw/AFG/c_0_w1_10_w2_$t_width/"
			./target/release/examples/tsptw $f -s TD -w $t_width -j -x "/home/eaeigbe/Documents/PhD/ddo/experiments/results/tsptw/AFG/TD_w_$t_width/"
			./target/release/examples/tsptw $f -s TD -w $t_width -j -k -x "/home/eaeigbe/Documents/PhD/ddo/experiments/results/tsptw/AFG/TD_c_w_$t_width/"
		done

		rm /home/eaeigbe/Documents/PhD/ddo/experiments/results/tsptw/AFG/summary_w1_10_w2_$t_width.csv
		echo Name,Lower,Upper,Duration,Aborted,RefineCluster,CompileCluster,Binary,Solver,Width,Gap,Objective  > /home/eaeigbe/Documents/PhD/ddo/experiments/results/tsptw/AFG/summary_w1_10_w2_$t_width.csv
		
		python /home/eaeigbe/Documents/PhD/ddo/experiments/src/clustering_incremental_tests/analyse.py -i /home/eaeigbe/Documents/PhD/ddo/experiments/results/tsptw/AFG/c_0_w1_10_w2_$t_width -o  /home/eaeigbe/Documents/PhD/ddo/experiments/results/tsptw/AFG/summary_w1_10_w2_$t_width.csv
		python /home/eaeigbe/Documents/PhD/ddo/experiments/src/clustering_incremental_tests/analyse.py -i /home/eaeigbe/Documents/PhD/ddo/experiments/results/tsptw/AFG/c_1_w1_10_w2_$t_width -o  /home/eaeigbe/Documents/PhD/ddo/experiments/results/tsptw/AFG/summary_w1_10_w2_$t_width.csv
		python /home/eaeigbe/Documents/PhD/ddo/experiments/src/clustering_incremental_tests/analyse.py -i /home/eaeigbe/Documents/PhD/ddo/experiments/results/tsptw/AFG/TD_w_$t_width -o  /home/eaeigbe/Documents/PhD/ddo/experiments/results/tsptw/AFG/summary_w1_10_w2_$t_width.csv
		python /home/eaeigbe/Documents/PhD/ddo/experiments/src/clustering_incremental_tests/analyse.py -i /home/eaeigbe/Documents/PhD/ddo/experiments/results/tsptw/AFG/TD_c_w_$t_width -o  /home/eaeigbe/Documents/PhD/ddo/experiments/results/tsptw/AFG/summary_w1_10_w2_$t_width.csv
		python /home/eaeigbe/Documents/PhD/ddo/experiments/src/clustering_incremental_tests/analyse.py -i /home/eaeigbe/Documents/PhD/ddo/experiments/results/tsptw/AFG/TD_B_n_B -o  /home/eaeigbe/Documents/PhD/ddo/experiments/results/tsptw/AFG/summary_w1_10_w2_$t_width.csv
	done

		python /home/eaeigbe/Documents/PhD/ddo/experiments/src/clustering_incremental_tests/plot.py -i /home/eaeigbe/Documents/PhD/ddo/experiments/results/tsptw/AFG/summary_w1_10_w2_20.csv,/home/eaeigbe/Documents/PhD/ddo/experiments/results/tsptw/AFG/summary_w1_10_w2_50.csv,/home/eaeigbe/Documents/PhD/ddo/experiments/results/tsptw/AFG/summary_w1_10_w2_100.csv,/home/eaeigbe/Documents/PhD/ddo/experiments/results/tsptw/AFG/summary_w1_10_w2_200.csv -t min

}

function misp {
	F="/home/eaeigbe/Documents/PhD/ddo/resources/misp/*"

	# #first run oracle
	# rm -r /home/eaeigbe/Documents/PhD/ddo/experiments/results/misp/TD_B_n_B
	# mkdir /home/eaeigbe/Documents/PhD/ddo/experiments/results/misp/TD_B_n_B
	# for f in $F; do
	# 	echo -e "\n"$f
	# 	./target/release/examples/misp $f -j -s BB -d 30 -x "/home/eaeigbe/Documents/PhD/ddo/experiments/results/misp/TD_B_n_B/"
	# done

	# for t_width in {20,50,100,200}; do
	for t_width in {20,50,100,200}; do
		rm -r /home/eaeigbe/Documents/PhD/ddo/experiments/results/misp/c_1_w1_10_w2_$t_width
		rm -r /home/eaeigbe/Documents/PhD/ddo/experiments/results/misp/c_1_b_1_w1_10_w2_$t_width
		rm -r /home/eaeigbe/Documents/PhD/ddo/experiments/results/misp/c_0_w1_10_w2_$t_width
		rm -r /home/eaeigbe/Documents/PhD/ddo/experiments/results/misp/TD_w_$t_width
		rm -r /home/eaeigbe/Documents/PhD/ddo/experiments/results/misp/TD_c_w_$t_width
		

		mkdir /home/eaeigbe/Documents/PhD/ddo/experiments/results/misp/c_1_w1_10_w2_$t_width
		mkdir /home/eaeigbe/Documents/PhD/ddo/experiments/results/misp/c_1_b_1_w1_10_w2_$t_width
		mkdir /home/eaeigbe/Documents/PhD/ddo/experiments/results/misp/c_0_w1_10_w2_$t_width
		mkdir /home/eaeigbe/Documents/PhD/ddo/experiments/results/misp/TD_w_$t_width
		mkdir /home/eaeigbe/Documents/PhD/ddo/experiments/results/misp/TD_c_w_$t_width
		

		for f in $F; do
			echo -e "\n"$f
			./target/release/examples/misp $f -s IR -w $t_width -c -j -x "/home/eaeigbe/Documents/PhD/ddo/experiments/results/misp/c_1_w1_10_w2_$t_width/"
			./target/release/examples/misp $f -s IR -w $t_width -c -b -j -x "/home/eaeigbe/Documents/PhD/ddo/experiments/results/misp/c_1_b_1_w1_10_w2_$t_width/"
			./target/release/examples/misp $f -s IR -w $t_width -j -x "/home/eaeigbe/Documents/PhD/ddo/experiments/results/misp/c_0_w1_10_w2_$t_width/"
			./target/release/examples/misp $f -s TD -w $t_width -j -x "/home/eaeigbe/Documents/PhD/ddo/experiments/results/misp/TD_w_$t_width/"
			./target/release/examples/misp $f -s TD -w $t_width -j -k -x "/home/eaeigbe/Documents/PhD/ddo/experiments/results/misp/TD_c_w_$t_width/"
		done

		rm /home/eaeigbe/Documents/PhD/ddo/experiments/results/misp/summary_w1_10_w2_$t_width.csv
		echo Name,Lower,Upper,Duration,Aborted,RefineCluster,CompileCluster,Binary,Solver,Width,Gap,Objective > /home/eaeigbe/Documents/PhD/ddo/experiments/results/misp/summary_w1_10_w2_$t_width.csv
		
		python /home/eaeigbe/Documents/PhD/ddo/experiments/src/clustering_incremental_tests/analyse.py -i /home/eaeigbe/Documents/PhD/ddo/experiments/results/misp/c_0_w1_10_w2_$t_width -o  /home/eaeigbe/Documents/PhD/ddo/experiments/results/misp/summary_w1_10_w2_$t_width.csv
		python /home/eaeigbe/Documents/PhD/ddo/experiments/src/clustering_incremental_tests/analyse.py -i /home/eaeigbe/Documents/PhD/ddo/experiments/results/misp/c_1_w1_10_w2_$t_width -o  /home/eaeigbe/Documents/PhD/ddo/experiments/results/misp/summary_w1_10_w2_$t_width.csv
		python /home/eaeigbe/Documents/PhD/ddo/experiments/src/clustering_incremental_tests/analyse.py -i /home/eaeigbe/Documents/PhD/ddo/experiments/results/misp/c_1_b_1_w1_10_w2_$t_width -o  /home/eaeigbe/Documents/PhD/ddo/experiments/results/misp/summary_w1_10_w2_$t_width.csv
		python /home/eaeigbe/Documents/PhD/ddo/experiments/src/clustering_incremental_tests/analyse.py -i /home/eaeigbe/Documents/PhD/ddo/experiments/results/misp/TD_w_$t_width -o  /home/eaeigbe/Documents/PhD/ddo/experiments/results/misp/summary_w1_10_w2_$t_width.csv
		python /home/eaeigbe/Documents/PhD/ddo/experiments/src/clustering_incremental_tests/analyse.py -i /home/eaeigbe/Documents/PhD/ddo/experiments/results/misp/TD_c_w_$t_width -o  /home/eaeigbe/Documents/PhD/ddo/experiments/results/misp/summary_w1_10_w2_$t_width.csv
		python /home/eaeigbe/Documents/PhD/ddo/experiments/src/clustering_incremental_tests/analyse.py -i /home/eaeigbe/Documents/PhD/ddo/experiments/results/misp/TD_B_n_B -o  /home/eaeigbe/Documents/PhD/ddo/experiments/results/misp/summary_w1_10_w2_$t_width.csv
	done

	# python /home/eaeigbe/Documents/PhD/ddo/experiments/src/clustering_incremental_tests/plot.py -i /home/eaeigbe/Documents/PhD/ddo/experiments/results/misp/summary_w1_10_w2_20.csv,/home/eaeigbe/Documents/PhD/ddo/experiments/results/misp/summary_w1_10_w2_50.csv,/home/eaeigbe/Documents/PhD/ddo/experiments/results/misp/summary_w1_10_w2_100.csv,/home/eaeigbe/Documents/PhD/ddo/experiments/results/misp/summary_w1_10_w2_200.csv -t max


}

# sop
# knapsack
tsptw
# misp



