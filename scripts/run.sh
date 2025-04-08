#!/bin/sh
#This script runs the synthesizer on all files in the specified input dir.
inputs_dir="./tests/synth/test/"
#All outputs will be placed in the outputs_dir folder
outputs_dir="./outputs/"
#Runs in foreground or background
foreground=0


#Synthesizer related arguments
bitwidth=4
num_programs=20
num_iters=10
total_rounds=40
program_length=20
solution_size=0
random_seed=2333
condition_length=6
num_abd_procs=5
INIT_COST=1
INV_TEMP=200

for entry in "$inputs_dir"/*
do
  echo "$entry"
  filename=$(basename ${entry})
  file_extension="${filename##*.}"
  file_base_name="${filename%.*}"
  if [ "$foreground" -eq 1 ]; then
    echo synth-transfer ${entry} -num_programs ${num_programs} -total_rounds ${total_rounds} -num_iters ${num_iters} -condition_length \
     ${condition_length} -solution_size ${solution_size} -num_abd_procs ${num_abd_procs} -weighted_dsl -random_seed ${random_seed}\
      -bitwidth ${bitwidth} -program_length ${program_length} -outputs_folder ${outputs_dir}${file_base_name}
    synth-transfer ${entry} -num_programs ${num_programs} -total_rounds ${total_rounds} -num_iters ${num_iters} -condition_length \
     ${condition_length} -solution_size ${solution_size} -num_abd_procs ${num_abd_procs} -weighted_dsl -random_seed ${random_seed}\
      -bitwidth ${bitwidth} -program_length ${program_length} -outputs_folder ${outputs_dir}${file_base_name}
  else
    echo synth-transfer ${entry} -num_programs ${num_programs} -total_rounds ${total_rounds} -num_iters ${num_iters} -condition_length \
     ${condition_length} -solution_size ${solution_size} -num_abd_procs ${num_abd_procs} -weighted_dsl -random_seed ${random_seed}\
      -bitwidth ${bitwidth} -program_length ${program_length} -outputs_folder ${outputs_dir}${file_base_name} &
    synth-transfer ${entry} -num_programs ${num_programs} -total_rounds ${total_rounds} -num_iters ${num_iters} -condition_length \
     ${condition_length} -solution_size ${solution_size} -num_abd_procs ${num_abd_procs} -weighted_dsl -random_seed ${random_seed}\
      -bitwidth ${bitwidth} -program_length ${program_length}  -outputs_folder ${outputs_dir}${file_base_name} &
  fi
done
