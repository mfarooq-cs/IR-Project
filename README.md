# IR-Project
Course project of Information Retrieval

**Project Setup**
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

**Step-wise Implementation**

1. Analyze data and find classes from data

2. Split data in 75:25 ratio

3. Text Preprocessing

4. Create dictionary of tokens

5. Classifier Training
```
	a. Calculate Class Probabilities for Naive Bayesian
	b. Calculate centroids for Rocchio
	c. Calclulate K nearest neighbours for KNN
```
6. Test a single unseen doc

7. Test a pool of 25% unseen docs

8. Calculate Evaluation Metrics

9. Calculate Precision, Recall & Accuracy

**Single Doc Classification**
```
python classifier.py dataset doc_to_classify
```
1. For Naive-Baysiean classifier
```
python NaiveBayes.py Datasets/legalcases/ test_data/doc
```
2. For Rocchio classifier
```
python Rocchio.py Datasets/sentensedata/ test_data/doc
```