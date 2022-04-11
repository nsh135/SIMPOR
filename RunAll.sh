#! /bin/bash
while read -r dataset;
do
   sbatch job.sh $dataset ;
done < datasets.txt