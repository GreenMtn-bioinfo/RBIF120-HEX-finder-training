#! /bin/bash


# NOTE: This script will not work unless the root folder is empty of the directories these three scripts make
# in other words it expects a blank slate, see the README if you are trying to recreate the large NPY of training/testing profiles


# TODO: add user-confirmed delete here

# 1) Run script 1
script="preparation_1.py"
echo "Running $script..."
python "./$script"
exstat=$?

# 2) If previous worked, run script 2
if [ $exstat -ne 0 ]; then
	echo "Bash: script $script failed to execute properly, stopping."
	exit 1
else
	script="get_boundary_seqs_2.py"
	echo "Running $script..."
	python "./$script"
	exstat=$?
fi

# 3) If previous worked, run script 3
if [ $exstat -ne 0 ]; then
	echo "Bash: script $script failed to execute properly, stopping."
	exit 1
else
	script="profile_generator_3.py"
	echo "Running $script..."
	python "./$script"
	exstat=$?
fi

# 4) Report if final script ran successfully
if [ $exstat -ne 0 ]; then
	echo "Bash: script $script failed to execute properly, stopping."
	exit 1
else
	echo "Success! All scripts ran without issue."
fi