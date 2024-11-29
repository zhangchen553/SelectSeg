# SelectSeg
selective crack image segmentation

Each of the main.py has it corresponding config.py  
E.g: e_main.py —> a_config.py  
All main.py can run independently.

## First stage:
Run e_main.py several time to train multiple models with different random seeds.  
(3 models should be enough)

## Second stage: 
 c_ranking_metric.py is use to calculate the ranking score for each image.  
c_data_ranking.py then will output different ratio of image list according to the ranking score.

## Third stage (optional):
e_main_partial_fully.py only use high ranking image to train new model.  
This script can help to decide the rankings threshold.

## Fourth stage (optional):
e_cpa_main.py is semi supervised learning to train model.  
It use both high ranking image list and the corresponding low ranking image list.
