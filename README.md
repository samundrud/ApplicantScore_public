# ApplicantScore
___Fast Track Top Talent___

This consulting project was completed as part of the Insight Data Science Fellowship program (New York, YN, Summer 2020) by Sarah Amundrud


# Table of Contents
1. [Introduction](README.md#introduction)
2. [Data](README.md#data)
3. [Approach](README.md#approach)
4. [Requirements](README.md#requirements)
5. [Repository Contents](README.md#repository-contents)
 
 
 
# Introduction
My client, a tech company that hires people across several domains, recieves more than 12.000 applications a year for less than 1000 open positions across seven sectors.
Processing and ranking applications manually not only results in a significant cost in terms of time, but it also means that it can take up to several weeks to respond to applicants. 
In order to reduce the processing time of applications, I used machine learning and natural language processing (NLP) to develop an algorithm that scores and ranks applications, thus allowing my client to prioritize high quality applications for further processing along with significantly cutting down on time spent ranking the applications manually. 

# Data
I obtained 3600 job applications that were labeled as either no (the application was rejected during the first round of review) or Yes+ (the application went on to the next stage). What made this task challenging is that the applications consisted almost entirely of unstructured data; namely, the applicants answers to a variety of application questions, ranging from describing their education and professional background, domain knowledge, industry specific skills, along what motivated them to apply for the position. 

 
# Approach
I trained five logistic regression machine learning models that ranks unstructured text based job applications.
Because of the compact and often technical nature of the answers, I decided on employing an approach to feature engineering that was largely based on domain knowledge and targeted searches for certain keywords. After discussions with several hiring managers, I came up with close to 200 keywords that were potentially important in predicting whether or not an application was making it to the next round (namely the interview). This process resulted in a dataset of almost 200 features that included keywords relevant to relevant background, required skills, and domain knowledge. 






# Requirements

The following languages and packages were used in this project:

* python 3.7.4
* numpy 1.16.5
* pandas 0.25.1
* matplotlib 3.1.1
* seaborn 0.9.0
* scikit-learn 0.23.1 
* nltk 3.4.5


As this project contains confidential information (i.e., applications to the Insight fellowship programs), the data is not published in this repository.


# Repository Contents

* __0_AppProcessor.py__ - Processes unstructured text based answers and creates structured data set with relevant features

* __0_AppScorer.py__ - Creates applicant scores for each domain

* __1_Domain_Model_DS.ipynb__ - Model for Data Science

* __2_Domain_Model_HD.ipynb__ - Model for Health Data Science

* __3_Domain_Model_AI.ipynb__ - Model for Artificial Intelligence

* __4_Domain_Model_DE.ipynb__ - Model for Data Engineering

* __5_Domain_Model_DO.ipynb__ - Model for DevOps



