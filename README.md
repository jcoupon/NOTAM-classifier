# Final Project DSWD-2018-07: NOTAM-classifier

Capstone project for Propulsion. Machine learning classifier for Notices to Airmen (NOTAM).


## Documentation
### Python

* [maps in python](https://uoftcoders.github.io/studyGroup/lessons/python/cartography/lesson/)

### Notam decode guides

* [wikipedia.org](https://en.wikipedia.org/wiki/NOTAM)
* [theairlinepilots.com](https://www.theairlinepilots.com/flightplanningforairlinepilots/notamdecode.php)
* [thinkaviation.net](http://thinkaviation.net/notams-decoded/)
* [flightcrewguide.com](http://flightcrewguide.com/wiki/preflight/notam-format/)
* [www.drorpilot.com](http://www.drorpilot.com/English/notam.htm) (automated)
* [snowflakesoftware.com](https://snowflakesoftware.com/news/top-7-things-need-know-notams/)


## workflow

### cleaning


```
cd Notebooks

../python/main.py clean '../Data/23-08-2018/Export.txt' -sep ';' -path_out test.csv
```

### clustering


```
../python/main.py train ../Data/training/train.csv         -vectorize_method word2vec -n_dim 50         -cluster_method  hierarch_euclid_ward        -path_model ../Data/training/train_model_vectorize_word2vec.pickle,../Data/training/train_model_cluster_hierarch_euclid_ward.pickle
../python/main.py predict ../Data/training/train.csv         -vectorize_method word2vec         -cluster_method  hierarch_euclid_ward        -path_model ../Data/training/train_model_vectorize_word2vec.pickle,../Data/training/train_model_cluster_hierarch_euclid_ward.pickle         -cluster_dist test         -path_out ../Data/training/train_word2vec_hierarch_euclid_ward_cluster_purity.csv
../python/main.py predict ../Data/training/test.csv         -vectorize_method word2vec         -cluster_method  hierarch_euclid_ward        -path_model ../Data/training/train_model_vectorize_word2vec.pickle,../Data/training/train_model_cluster_hierarch_euclid_ward.pickle         -cluster_dist test         -path_out ../Data/training/test_word2vec_hierarch_euclid_ward_cluster_purity.csv

```

### classification

```
../python/main.py train_classifier ../Data/training/train_predict.csv -path_model ../Data/training/train_model_classifier.pickle

../python/main.py predict_classifier ../Data/training/test_predict.csv -path_model ../Data/training/train_model_classifier.pickle -path_out ../Data/training/test_predict_class.csv


```
