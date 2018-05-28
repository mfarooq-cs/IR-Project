# IR-Project
Course project of Information Retrieval

Steps to run the project
1. Install python 2.7
```
sudo apt-get update
sudo apt-get install python2.7
```
2. Install pip
```
sudo apt-get install python-pip
```
3. Install project specific python packages
```
pip install -r requirements.txt
```
4. Now, you can run any classifier(naive-bayseian, rocchio and knn) by using
```
python classifier.py dataset
```
5. For Naive-Baysiean classifier
```
python NaiveBayes.py Datasets/legalcases/
```
6. For Rocchio classifier
```
python Rocchio.py Datasets/sentensedata/
```
7. For KNN classifier
```
python knn.py Datasets/emails/
```
