## Recommender  
A simple news recommender implemented follow algorithm:  
* content based knn
* collaboration filter nmf

#### Environment & packages
* linux
* python2  
* json, numpy, sklearn, jieba  

#### How to run
* `$cd path/to/RecommendationSystem`
* `$cp path/to/user_click_data.txt ./data`
* `$export PYTHONPATH=./:$PATHONPATH`
* `$python preprocess/processor.py`
* `$python recommender/recommender.py`
* `$python test/test.py`

#### File tree
```
RecommendationSystem:
|--algorithm:  
   |--__init__.py, cf.py,  knn.py   
     
|--data:  
   |-- user_click_data.txt,  stop_words.txt  
    
|--preprocess:  
   |-- __init__.py,  processor.py   
    
|--recommender:  
   |-- __init__.py,  recommender.py  
    
|--test:  
   |-- __init__.py,  test.py    
    
|--tools:  
   |-- __init__.py,  utils.py 
```
