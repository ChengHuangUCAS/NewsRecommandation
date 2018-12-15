#recommender

#### environment & packages
* linux
* python2  
* json, numpy, sklearn, jieba  

#### how to run
* `$cd path/to/RecommendationSystem`
* `$cp path/to/user_click_data.txt ./data`
* `$export PYTHONPATH=./:$PATHONPATH`
* `$python preprocess/processor.py`
* `$python recommender/recommender.py`
* `$python test/test.py`

#### file tree
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
