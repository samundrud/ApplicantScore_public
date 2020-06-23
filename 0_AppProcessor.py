## This script takes in raw applications and processes them so for further analysis / input into models


# import libraries
import pandas as pd
import numpy as np
from nltk.corpus import stopwords

# supress chained assignement warning
pd.options.mode.chained_assignment = None  # default='warn'

# import raw applications
df_2019 = pd.read_csv("../ApplicantScore_Data/Applications_2019.csv")
df_2020 = pd.read_csv("../ApplicantScore_Data/Applications_2020.csv")

# import keywords
keywords = pd.read_csv('../ApplicantScore_Data/Keywords.csv')



# Application Processor Function
## transforms raw applicatins to structured data that can be run through models

def app_processor(df, keywords):
    
    """
    inserts and runs functions for text feature extraction, text pre-processing, and keyword extraction
    feature engineering: converts categories to binary (or scaled betw. 0-1)
    generates dfMaster, a dataframe with all relevant features used to build models (or apply existing models
    """
    
    ### TEXT PROCESSING AND KEYWORD EXTRACTION

    # extract text questions and answers from raw data
    dftext = df[['hashed_application_id', 'question_short', 'answer_text']].copy()
    # remove duplicates
    dftext = dftext.drop_duplicates(['hashed_application_id', 'question_short'],keep='first')

    
    

    # Text feature extraction function
    
    def basicFE(text_frame):
        """
        Extracts basic features from text (answer_text) and adds results to data frame:
        count words
        count number of characters
        average word length
        count number of stopwords    
        """
        # word count: word_count
        text_frame['word_count'] = dftext['answer_text'].dropna().apply(lambda x: len(str(x).split(" ")))

        # number of characters: char_count
        text_frame['char_count'] = dftext['answer_text'].str.len() ## this also includes spaces

        # average word length: avg_word
        def avg_word(sentence):
            words = str(sentence).split()
            if len(words) == 0:
                pass
            else:
                return (sum(len(word) for word in words)/len(words))
        dftext['avg_word'] = dftext['answer_text'].dropna().apply(lambda x: avg_word(x))

        # number of stopwords
        stop = stopwords.words('english')
        dftext['stopwords'] = dftext['answer_text'].dropna().apply(lambda x: len([x for x in str(x).split() if x in stop]))

        return text_frame
 
    basicFE(dftext)

    
    
    # Text pre-processing function
    def textProc(text_frame):
        """
        Processes the applicant responses (answer_text) and adds to new column in data frame:
        lower case
        remove punctuation
        remove stop words    
        """
        # makes everything lower case
        text_frame['answer_text_proc'] = dftext['answer_text'].apply(lambda x: " ".join(x.lower() for x in str(x).split()))

        # removing punctuation and adds space ()
        dftext['answer_text_proc'] = dftext['answer_text_proc'].str.replace('[^\w\s]','')

        # removal of stop words
        stop = stopwords.words('english')
        dftext['answer_text_proc'] = dftext['answer_text_proc'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

        return dftext
    
    textProc(dftext)

    # remove string nan
    dftext['answer_text_proc'] = dftext['answer_text_proc'].replace('nan', '')

    
    
    # run kw_extractor functions: extract keywords from processed and raw text
    ### Keyword extraction functions
    ### general keywords
    def kw_extractor(dftext):
        keys = list(keywords['PLs'][keywords['PLs'].notna()]) +  list(keywords['Stats'][keywords['Stats'].notna()]) + list(keywords['Tools'][keywords['Tools'].notna()]) + list(keywords['General'][keywords['General'].notna()])
        # extract keywords from processed text
        for key in keys:
            dftext[key] = np.where(dftext['answer_text_proc'].str.contains(key), 1, 0) 
        return dftext
    
    kw_extractor(dftext)


    ### short and special character keywords
    def kw_extractor_special(dftext):
        # need to look for short words (e.g. R and C) as seperate words
        # (because these letters are often in words and thus get picked up by the above function)
        # look for these words in unprocessed text
        keys1 = list(keywords['PLs_short'][keywords['PLs_short'].notna()])
        for key in keys1:
            bin = []
            for text in dftext['answer_text_proc']:
                if key in str(text).lower().split():
                    bin.append(1)
                else:
                    bin.append(0)
            dftext[key] = bin

        keys2 = list(keywords['General_short'][keywords['General_short'].notna()])
        for key in keys2:
            bin = []
            for text in dftext['answer_text_proc']:
                if key in str(text).lower().split():
                    bin.append(1)
                else:
                    bin.append(0)
            dftext[key] = bin

        # for C++ and c#, look in raw text (because + removed in processed text)
        keys3 = ['c++', 'c#']
        for key in keys3:
            bin = []
            for text in dftext['answer_text']:
                if key in str(text).lower():
                    bin.append(1)
                else:
                    bin.append(0)
            dftext[key] = bin

        return dftext

    kw_extractor_special(dftext)







    ### FEATURE ENGINEERING

    # remove columns with text data and make one row per applicant
    df = df.drop(['question_short', 'question_text', 'answer_text'], axis=1)
    df.drop_duplicates(subset='hashed_application_id', keep='first', inplace=True)
    df = df.reset_index(drop=True)


    # create a column for program-location (but remove section)
    df['program_loc'] = df['plys_string'].apply(lambda x: x.split('-')[0])

    # convert yes/no columns to binary
    df['linkedin_available'] = df['linkedin_available'].map(dict(yes=1, no=0))
    df['github_available'] = df['github_available'].map(dict(yes=1, no=0))
    df['website_available'] = df['website_available'].map(dict(yes=1, no=0))

    # make dummy variables for season (session) and location
    df = pd.concat([df, pd.get_dummies(df['session'], prefix='session_')], axis=1)
    df = pd.concat([df, pd.get_dummies(df['location_short'], prefix='location_')], axis=1)

    # extract applicant IDs
    IDs = list(df['hashed_application_id'])





    ### PROCESS TEXT ANSWERS

    # create new data sheet with processed text answers
    text2 = dftext[dftext['hashed_application_id'].isin(IDs)]

    # add location to text2
    text2 = pd.merge(text2, df[['hashed_application_id', 'program_loc']], how="left", on='hashed_application_id')


    ## KEYWORD SEARCH IN TEXT ANSWERS
    # create empty dataframe with applicant IDs
    features = pd.DataFrame(df['hashed_application_id'], columns = ['hashed_application_id']) 

    # add text featurs to features
    # education level and study area
    feats = ['education_level', 'study_area']
    for feat in feats:
        temp = text2[['hashed_application_id', 'answer_text']][text2['question_short'] == feat]
        temp = temp.rename(columns = {'answer_text':feat})
        features = pd.merge(features, temp, how="left", on='hashed_application_id')

    # length of responses to "essay" questions
    feats = ['research_description', 'side_projects', 'coursework', 'industry_motivation',
            'toughest_challenge', 'pipeline_experience', 'domain_experience', 'exciting_innovation', 'codebase_size',
            'team_size']
    for feat in feats:
        temp = text2[['hashed_application_id', 'char_count']][text2['question_short'] == feat]
        temp = temp.rename(columns = {'char_count':(feat+'_char')})
        features = pd.merge(features, temp, how="left", on='hashed_application_id')

    # Programming languages and tools
    feats = ['python', 'javascript', 'fortran', 'golang', 'cobol', 'wasm', 'matlab', 'solidity', 'scala', 'pytorch', 'tensorflow',
           'bash', 'sql', 'c++', 'c#', 'r', 'ruby', 'rust', 'perl', 'java']
    for feat in feats:
        temp = text2[['hashed_application_id', feat]][text2['question_short'] == 'programming_languages']
        temp = temp.rename(columns = {feat:('PL_'+feat)})
        features = pd.merge(features, temp, how="left", on='hashed_application_id')

    # stats / ML skills
    feats = ['cluster', 'decision tree', 'deep learning', 'generalized linear model', 'glm', 'k nearest neighb',
             'linear regression', 'logistic regression', 'multiple regression', 'neural network', 'pca', 
             'principle component analys', 'random forest','supervised learning', 'support vector machine', 'svm',
             'unsupervised learning', 'regression', 'timeseries', 'time series']

    # need to look in several answers (e.g, research_description, side_projects, coursework, programming_languages)
    for feat in feats:
        temp = text2[['hashed_application_id', feat]][text2['question_short'] == 'programming_languages']
        temp = temp.rename(columns = {feat:('ML_skills_'+feat)})
        features = pd.merge(features, temp, how="left", on='hashed_application_id')
    for feat in feats:
        temp = text2[['hashed_application_id', feat]][text2['question_short'] == 'research_description']
        temp = temp.rename(columns = {feat:('ML_research_'+feat)})
        features = pd.merge(features, temp, how="left", on='hashed_application_id')
    for feat in feats:
        temp = text2[['hashed_application_id', feat]][text2['question_short'] == 'side_projects']
        temp = temp.rename(columns = {feat:('ML_project_'+feat)})
        features = pd.merge(features, temp, how="left", on='hashed_application_id')
    for feat in feats:
        temp = text2[['hashed_application_id', feat]][text2['question_short'] == 'coursework']
        temp = temp.rename(columns = {feat:('ML_coursework_'+feat)})
        features = pd.merge(features, temp, how="left", on='hashed_application_id')


    # extract and add other keywords
    feats = ['openvas', 'nlp', 'github', 'linux', 'gcp', 'nikto', 'node',
           'wireshark', 'terraform', 'sklearn', 'scikit', 'object oriented prog',
           'math', 'encryp', 'distributed system', 'cryptograph', 'cybersecurity',
           'leetcode', 'appsec', 'deployment', 'infrastructure',
           'machine learning', 'network', 'netsec', 'docker', 'full stack',
           'engineer', 'end to end', 'hyperledger', 'network security',
           'packet sniff', 'penetration', 'pipeline', 'postdoc', 'product',
           'rsa encrypt', 'scalable systems', 'software engineer', 'threat analys',
           'version control', 'oop', 'go', 'aws']


    # need to look in several answers (e.g, PLs (skills), research_description, side_projects, coursework,
    # pipeline_experience, domain_experience, systems_experience)
    for feat in feats:
        temp = text2[['hashed_application_id', feat]][text2['question_short'] == 'programming_languages']
        temp = temp.rename(columns = {feat:('KW_Skills_'+feat)})
        features = pd.merge(features, temp, how="left", on='hashed_application_id')
    for feat in feats:
        temp = text2[['hashed_application_id', feat]][text2['question_short'] == 'research_description']
        temp = temp.rename(columns = {feat:('KW_Research_'+feat)})
        features = pd.merge(features, temp, how="left", on='hashed_application_id')
    for feat in feats:
        temp = text2[['hashed_application_id', feat]][text2['question_short'] == 'side_projects']
        temp = temp.rename(columns = {feat:('KW_Project_'+feat)})
        features = pd.merge(features, temp, how="left", on='hashed_application_id')
    for feat in feats:
        temp = text2[['hashed_application_id', feat]][text2['question_short'] == 'coursework']
        temp = temp.rename(columns = {feat:('KW_Coursework_'+feat)})
        features = pd.merge(features, temp, how="left", on='hashed_application_id')
    for feat in feats:
        temp = text2[['hashed_application_id', feat]][text2['question_short'] == 'pipeline_experience']
        temp = temp.rename(columns = {feat:('KW_Pipeline_'+feat)})
        features = pd.merge(features, temp, how="left", on='hashed_application_id')
    for feat in feats:
        temp = text2[['hashed_application_id', feat]][text2['question_short'] == 'domain_experience']
        temp = temp.rename(columns = {feat:('KW_Domain_'+feat)})
        features = pd.merge(features, temp, how="left", on='hashed_application_id')
    for feat in feats:
        temp = text2[['hashed_application_id', feat]][text2['question_short'] == 'systems_experience']
        temp = temp.rename(columns = {feat:('KW_Systems_'+feat)})
        features = pd.merge(features, temp, how="left", on='hashed_application_id')





    #### SUMMARIZE AND SCALE FEATURES 

    ## sum up programming languages
    features['PL_all'] = features.filter(regex='^PL',axis=1).sum(axis=1)
    # scale, so values are between 0 and 1 (set max cut-off is 8)
    features['PL_all'][features['PL_all'] >= 8] = 8
    features['PL_all'] = features['PL_all']/8


    # create summary columns for keywords that could be mentioned in several answers
    # select all ML columns
    ML_all = features.filter(regex='^ML',axis=1)
    # sum up the ML techniques over the possible answers
    ML_group = ML_all.T.groupby([s.split('_')[2] for s in ML_all.T.index.values]).sum().T
    ML_group = ML_group.add_prefix('ML_')
    # if ML techniques was mentioned in 2 or more answers, only count once
    ML_group[ML_group > 0] = 1
    # make column that summarized all ML techniques mentioned for each applicant
    ML_group['ML_all'] = ML_group.sum(axis = 1, skipna = True)
    # scale, so values are between 0 and 1 (set max cut-off to 6)
    ML_group['ML_all'][ML_group['ML_all'] >= 6] = 6
    ML_group['ML_all'] = ML_group['ML_all']/6

    # merge back to features and remove original ML columns (for the seperate questions)
    features = features.join(ML_group)
    features = features.drop(ML_all.columns, axis=1)



    ### now do the same for the other keywords
    # select all KW columns
    KW_all = features.filter(regex='^KW',axis=1)
    # sum up the KWs over the possible answers
    KW_group = KW_all.T.groupby([s.split('_')[2] for s in KW_all.T.index.values]).sum().T
    # if KW was mentioned in 2 or more answers, only count once
    KW_group[KW_group > 0] = 1

    # merge back to features and remove original KW columns (for the seperate questions)
    features = features.join(KW_group)
    features = features.drop(KW_all.columns, axis=1)







    #### MERGE EXTRACTED FEATURES WITH MAIN DATA AND CREATE MASTER DATA FRAME
    dfMaster = pd.merge(df, features, how="left", on='hashed_application_id')

    # add MD to PhD and create two groups (Phd yes and no)
    dfMaster['PhD'] = 0
    dfMaster.loc[dfMaster['education_level'] == 'PhD', 'PhD'] = 1
    dfMaster.loc[dfMaster['education_level'] == 'MD', 'PhD'] = 1



    ## Determine "preferred degrees" for various programs

    ## Data Science
    # study area keywords
    keys = ['physics', 'astronomy', 'astrophysics', 'engineering', 'biology', 'ecology', 'chemistry',
             'data science', 'artificial intelligence', 'neuroscience', 'operations research', 'mathematics',
             'economics', 'finance', 'computer science']
    pref = []
    for study in dfMaster['study_area']:
        if any(s in str(study).lower() for s in keys):
            pref.append(1)
        else:
            pref.append(0)
    dfMaster['preferred_study_area_DS'] = pref


    ## Security
    # study area keywords
    keys = ['engineer', 'computer science', 'cybersecur']
    pref = []
    for study in dfMaster['study_area']:
        if any(s in str(study).lower() for s in keys):
            pref.append(1)
        else:
            pref.append(0)
    dfMaster['preferred_study_area_SEC'] = pref


    ## Decentralized Consensus (DC)
    # study area keywords
    keys = ['computer science', 'network', 'information security', 'cyber security']
    pref = []
    for study in dfMaster['study_area']:
        if any(s in str(study).lower() for s in keys):
            pref.append(1)
        else:
            pref.append(0)
    dfMaster['preferred_study_area_DC'] = pref



    ## Data Engineering (DE)
    # study area keywords
    keys = ['computer science', 'data science', 'analytics', 'computer engineer', 'software engineer']
    pref = []
    for study in dfMaster['study_area']:
        if any(s in str(study).lower() for s in keys):
            pref.append(1)
        else:
            pref.append(0)
    dfMaster['preferred_study_area_DE'] = pref


    ## Dev Ops (DO)
    # study area keywords
    keys = ['engineer']
    pref = []
    for study in dfMaster['study_area']:
        if any(s in str(study).lower() for s in keys):
            pref.append(1)
        else:
            pref.append(0)
    # add features to Master data
    dfMaster['preferred_study_area_DO'] = pref




    #### The scale of the length of the questions are way out of proportion
    ##scale them so they are all between 0 and 1

    # create list with all questions that were used in analysis
    questions = ['side_projects_char', 'coursework_char', 'industry_motivation_char', 'research_description_char',
                 'toughest_challenge_char', 'pipeline_experience_char', 'domain_experience_char', 'exciting_innovation_char',
                 'codebase_size_char', 'team_size_char']

    # assing zero to missing values
    for quest in questions:
        dfMaster[quest] = dfMaster[quest].fillna(0)
        dfMaster[quest] = dfMaster[quest].fillna(0)

    # make maximum allowable character length 1000 (because this was the allowed limit for a lot or applicants)
    # also divide by 1000 to scale length of characters
    for quest in questions:
        dfMaster[quest] = np.where((dfMaster[quest] > 1000), 1000, dfMaster[quest])
        dfMaster[quest] = dfMaster[quest] / 1000
    
    return dfMaster



## run function to process raw applications

# process data
dfMaster_2019 = app_processor(df_2019, keywords)
dfMaster_2020 = app_processor(df_2020, keywords)

# save as csv
#dfMaster_2019.to_csv('Data/Apps2019Processed.csv')
#dfMaster_2020.to_csv('Data/Apps2020Processed.csv')


