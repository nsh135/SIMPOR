This is the code for reproducing experiments in paper SIMPOR. 

1. Set up the environment using conda following requirments.txt

2. Run main E.g., 
python main.py --dataset moon --cuda True --gridSearch True --IR 3 -n_runs 2

3. compute and save result figures (in Figures directory) 
python plotResult.py

** If you could not achieve the expected results, pls adjust parameters. 
E.g., Neural network related parameters are located in main.py from line 82-102 
Optimization gradient ascent parameters are in simpor.py line 317-320

** The lastest version was tested with GPU option.
Have not tested the multi-threads CPU option (though it worked a while before submited day). 
If you find any problem, pls make a request. Due to the limit of hardware, it would run for a while. 
