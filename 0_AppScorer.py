## Script that takes in processed applications (i.e., output from AppProcessor.py script)
## creates scores for each applicant for program suitability for DS, HD, AI, DE, DO, and SEC


import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import pickle

pd.options.mode.chained_assignment = None  # default='warn'

# import processed applications
Apps2019Processed = pd.read_csv('Data/Apps2019Processed.csv')
Apps2020Processed = pd.read_csv('Data/Apps2020Processed.csv')


## chose applications to be scored, e.g., 
data = Apps2019Processed # or pick Apps2020Processed




## Import models

# Data Science
DS_feats = pd.read_csv("Models/DS_features.csv")
DS_LGmodel = pickle.load(open('Models/DS_LGmodel.sav', 'rb'))

# Health Data Science
HD_feats = pd.read_csv("Models/HD_features.csv")
HD_LGmodel = pickle.load(open('Models/HD_LGmodel.sav', 'rb'))

## Artificial Intelligence
AI_feats = pd.read_csv("Models/AI_features.csv")
AI_LGmodel = pickle.load(open('Models/AI_LGmodel.sav', 'rb'))

## Data Engineering
DE_feats = pd.read_csv("Models/DE_features.csv")
DE_LGmodel = pickle.load(open('Models/DE_LGmodel.sav', 'rb'))

## DevOps
DO_feats = pd.read_csv("Models/DO_features.csv")
DO_LGmodel = pickle.load(open('Models/DO_LGmodel.sav', 'rb'))

## Security
SEC_feats = pd.read_csv("Models/SEC_features.csv")
SEC_LGmodel = pickle.load(open('Models/SEC_LGmodel.sav', 'rb'))



## Applicant Scoring Function

def score_function(features, model, data, name):
    """
    takes in features: csv file with relevant features for model
    model: logistic regression model (DS, HD, AI, DE, DO, or SEC)
    data: processed data frame: dfMaster
    name: string of program name
    """
    
    # save feature names in list
    feats = list(features['features'])

    # extract columns with applicant info from processed data and save in score sheet
    Score_Sheet = data[['hashed_application_id', 'hashed_candidate_id', 'program_short', 'location_short',
                              'year', 'session']]

    # extract columns with relevant applicant features
    applicant_feats = data[feats]
    applicant_feats = applicant_feats.fillna(0)

    # add score to score sheet
    Score_Sheet[str(name) + '_score'] = model.predict_proba(applicant_feats)[:,1]

    # join features with score sheet
    Score_Sheet = Score_Sheet.join(applicant_feats)
    
    return Score_Sheet



## Create applicant scores for each of the six programs


data = df_2019 # or pick df_2020

# process data
dfMaster = app_processor(df_2019, keywords)


## Generate Score Sheets for each program
### score sheets also include relevant features (ordered by feature importance)
DS_Score_Sheet = score_function(DS_feats, DS_LGmodel, dfMaster, 'DS')
HD_Score_Sheet = score_function(HD_feats, HD_LGmodel, dfMaster, 'HD')
AI_Score_Sheet = score_function(AI_feats, AI_LGmodel, dfMaster, 'AI')
DE_Score_Sheet = score_function(DE_feats, DE_LGmodel, dfMaster, 'DE')
DO_Score_Sheet = score_function(DO_feats, DO_LGmodel, dfMaster, 'DO')
SEC_Score_Sheet = score_function(SEC_feats, SEC_LGmodel, dfMaster, 'SEC')


##  add scores to dfMaster and create Applicant_Scores sheet
dfMaster['DS_score'] = DS_Score_Sheet['DS_score']
dfMaster['HD_score'] = HD_Score_Sheet['HD_score']
dfMaster['AI_score'] = AI_Score_Sheet['AI_score']
dfMaster['DE_score'] = DE_Score_Sheet['DE_score']
dfMaster['DO_score'] = DO_Score_Sheet['DO_score']
dfMaster['SEC_score'] = SEC_Score_Sheet['SEC_score']


Applicant_Scores = dfMaster[['hashed_application_id', 'hashed_candidate_id', 'DS_score',
                             'HD_score', 'AI_score', 'DE_score', 'DO_score', 'SEC_score']]




## save score sheets (including relevant features) for each applicant and each program as csv
#DS_Score_Sheet.to_csv('Data/DS_Score_Sheet.csv')
#HD_Score_Sheet.to_csv('Data/HD_Score_Sheet.csv')
#AI_Score_Sheet.to_csv('Data/AI_Score_Sheet.csv')
#DE_Score_Sheet.to_csv('Data/DE_Score_Sheet.csv')
#DO_Score_Sheet.to_csv('Data/DO_Score_Sheet.csv')
#SEC_Score_Sheet.to_csv('Data/SEC_Score_Sheet.csv')



## save all applicant scores (without features) as csv
#Applicant_Scores.to_csv('Data/Applicant_Scores.csv')





