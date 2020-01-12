#creates a bash file which executes our Perceptron with all possible comnbinaitons
#of our 8 features
#bash file also executes Evaluation and writes scores for all combinations in file output.txt

import itertools as it

nr_feats = 8
tuples = []

for i in range(nr_feats+2):
	if i > 0:
		comb = it.combinations(range(nr_feats+1), i)
		for j in comb:
			if 0 not in j:
				tuples.append(j) #this is the tuple with different feature combinations


with open("execute.txt", "wt") as file:
	file.write("#!/bin/bash" + "\n\n")
	for flags in tuples:
		file.write("python3 Perceptron.py train.col dev.col " + "".join(map(str, flags)) + "\n")
		file.write("python3 Evaluation.py dev_stripped.txt prediction.txt >> output.txt" + "\n")



#1
# write in your command line: python3 createbashfile.py
# make sure these files are in your directory: Perceptron.py, train.col, dev.col, Evaluation.py, dev_stripped.py
# all available on bitbucket


#2 
# when step 1 is completed you should have a file execute.txt
# run this file on your command line: bash execute.txt

#3
# when step 2 is completed (will take very long) you should have a file called output.txt
# this file contains all the scores for each possible feature combination

