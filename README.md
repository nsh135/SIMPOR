1. Set up the environment using conda following requirments.txt

** If run all experiement at once using Slurm
### 2.a If using Slum on multiple datasets in datasets.txt
bash RunAll.sh

** 2.bIf run each dataset separately. Run main file E.g., 
python main.py --dataset moon --cuda True --gridSearch True --IR 3 -n_runs 2
python main.py --dataset moon --cuda False --n_threads 30  --IR 3 --n_runs 1

3. Generate and save result figures to Figures directory. This command was already run automatically in step 2. Only run this if you need regenerate figures.  
python plotResult.py EXP_DIR_NAME

4. Collecting all experiments and generate table reuslts. For report purpose only.
python ResultCollecting.py 


