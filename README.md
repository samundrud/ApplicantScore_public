# ApplicantScore
___Fast Track Top Talent___

This consulting project was completed as part of the Insight Data Science Fellowship program (New York, YN, Summer 2020) by Sarah Amundrud.

This is the public repository of the project, which does not include any data.


# Table of Contents
1. [Introduction](README.md#introduction)
2. [Data](README.md#data)
3. [Approach](README.md#approach)
4. [Requirements](README.md#requirements)
5. [Repository Contents](README.md#repository-contents)
 
 
 
# Introduction
My client, a tech company that hires people across several tech domains (e.g., data science, data engineering, etc.), receives more than 12,000 applications a year for less than 1,000 open positions.
Processing and ranking applications manually can take several weeks, resulting in a slow response rate to applicants. 
To help my client reduce their application processing time and respond to top applicants in days rather than weeks (i.e., for interview requests), I used machine learning and natural language processing (NLP) to develop an algorithm that scores and ranks applications, thus allowing my client to prioritize and fast track high quality applications. In addition to significantly speeding up the response time, my product saves my client roughly 2000h/year in manual processing and ranking of applications. 

# Data
I obtained 3,600 job applications that were labeled as either no (the application was rejected during the first round of review) or Yes+ (the application went on to the next stage). The applications consisted almost entirely of unstructured data; namely, the applicants answers to a variety of application questions, ranging from describing their education and professional background, domain knowledge, industry specific skills, along what motivated them to apply for the position. 

 
# Approach
I trained five logistic regression models (one for each of five tech domains my client is involved in) that ranks unstructured text based job applications.
From the text based answers to application questions, I engineered features using a targeted keyword extraction approach that was based on detailed discussions with hiring managers who have deep domain knowledge. The resulting features were related to an applicant's relevant background, required skills, as well as domain knowledge. 



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

* __Models__ - Folder containing models and features for the various domains

