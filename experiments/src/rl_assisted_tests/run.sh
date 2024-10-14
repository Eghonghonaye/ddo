#!/bin/bash

SAVEIFS=$IFS
IFS=$(echo -en "\n\b")



function jssp {
	F="/home/eaeigbe/Documents/PhD/ddo/resources/jssp/*"
		for f in $F; do
			echo $f
			./target/release/examples/jssp $f -d 30 -w 1 -j -x "/home/eaeigbe/Documents/PhD/ddo/experiments/results/jssp/w_1_d_10_dd/"
			./target/release/examples/jssp $f -d 30 -w 1 -j -x "/home/eaeigbe/Documents/PhD/ddo/experiments/results/jssp/w_1_d_10_dd_rl/" -m "/home/eaeigbe/Documents/PhD/trainedModels/job_shop/base_50_20.tar"
		done
}

jssp


