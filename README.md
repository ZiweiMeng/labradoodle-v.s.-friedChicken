# Project: Labradoodle or Fried Chicken? In Blakc and White. 
![image](figs/poodleKFC.jpg)

### [Full Project Description](doc/project3_desc.html)

Term: Spring 2017

+ Team # 3
+ Team members
	+ Ziwei Meng (Presenter)
	+ Bowen Huang
	+ Nanjun Wang
	+ Jingru Xue
	+ Chengcheng Yuan

+ Project summary: In this project, we created a classification engine for images of poodles versus images of fried chicken in black and white thus focusing on texture instead of color. We added CNN features extracted from pre-trained Inception V3 network which decrease our test error by a lot. At the same time, we tested several classifiers on our data: GBM, SVM, Random Forest and Logistic Regression. Among these models, Logistic Regression and SVM yielded the most accurate results. We ensembled them to get the final model, reached a 95.5% accuracy compared to 72.4% accuracy of baseline model.

**Contribution statement**:<br/>
Ziwei tuned the baseline model, was responsible for the extraction of the CNN features, formatted the main code according to the guidelines and organized the presentation.<br/>
Jingru tuned advanced model with logistic regression on both SIFT and CNN features, selected best lambda using grid search and cross-validation.<br/>
Nanjun tuned advanced model with SVM using linear kernel on both SIFT and CNN features, selected best cost using grid search and cross-validation.<br/>
Chengcheng tuned advanced model with random forest on both SIFT and CNN features, selected best n.var using grid search and cross-validation.<br/>
Bowen tuned advanced model with GBM on both SIFT and CNN features, selected best n.trees using grid search and cross-validation.<br/>

**Necessary libraries/packages:** <br/>
1. data.table (R)
  * install.packages("data.table")
2. keras (python)
  * $pip install keras==1.2.1
3. gbm (R)
  * install.packages("gbm")
4. glmnet (R)
  * install.packages("glmnet")

Following [suggestions](http://nicercode.github.io/blog/2013-04-05-projects/) by [RICH FITZJOHN](http://nicercode.github.io/about/#Team) (@richfitz). This folder is orgarnized as follows.

```
proj/
├── lib/
├── data/
├── doc/
├── figs/
└── output/
```

Please see each subfolder for a README file.
