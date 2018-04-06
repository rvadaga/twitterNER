notes for 01_crf_suite_wnut
---------------------------
- this was downloaded from https://github.com/aritter/twitter_nlp
- link was obtained from wnut 2016 website
- given directory is in twitter_nlp/data/annotated/wnut16
- baseline.sh was provided in the above dir; they used train_data='train',
test_data='dev'.
- not sure if that's correct, so I modified the script 
to generate other scripts baseline_1.sh, etc.
- USAGE:
	./baseline.sh <folder_name_to_store_results>
- results are stored in ../../results/01_crf_suite_wnut/<given_folder_name>
- in order to run this package, you will need crfsuite installed.
- for successful installation, please adhere to 
http://www.chokkan.org/software/crfsuite/manual.html
- you will need to install liblbfgs before installing crfsuite


02_tagger_lample
----------------
- you may need to give execute permissions to conlleval inside
evaluations folder
- by default, for any given run, all outputs/scores inside temp
folder inside evaluations directory
- DO NOT forget to give embedding locations while running train.py
in case you have them

04_lasagne_nlp_wnut
-------------------
- use theano==0.8.2 to run the program

