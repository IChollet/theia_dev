#!/bin/bash

expected=true
for dir in interpolation algebra cubature algorithms ; do
    echo -e "\033[0;1;35mExplore $dir\033[0m"
    for file in ./"$dir"/test_*.cpp ; do
	name=$(basename "$file" .cpp)
	executable="./$dir/$name"
	trsh=$(make "./$dir/$name")
	actual=$("$executable")
	if [[ "$actual" == "true" ]]; then
	    echo -e "\t\033[32m[PASS]\033[0m $name"
	else
	    echo -e "\t\033[31m[FAIL]\033[0m $name (output: $actual)"
	    echo "$trsh"
	fi
    done
done
