import numpy as np
# helper functions for analysis of the data set

def get_strongly_correlated_features(df, threshold=0.7):
    # get strongly correlated features from a DataFrame above a certain threshold
    # returns two lists of tuples containing pairs of strongly correlated features
    corr_matrix = df.corr()
    # get the upper triangle to prevent going through the same data twice
    corr_matrix_up = corr_matrix.where(np.triu(np.ones(corr_matrix.shape)).astype(bool))
    positively_correlated_features = []
    negatively_correlated_features = []

    for i in range(len(corr_matrix_up.columns)):
        for j in range(i):
            if (corr_matrix_up.iloc[j, i] > threshold):
                positively_correlated_features.append((corr_matrix_up.columns[j], corr_matrix_up.columns[i]))
            elif (corr_matrix_up.iloc[j, i] < -threshold):
                negatively_correlated_features.append((corr_matrix_up.columns[j], corr_matrix_up.columns[i]))    

    return positively_correlated_features, negatively_correlated_features

# scoring dass-21 questionnaire and appending it to the dataframe
def append_dass21_scores(df, question_df):
    # dass-21 questionnaire has 3 subsections: depression, anxiety and stress
    # get question_codes for each
    anxiety_qs, depression_qs, stress_qs = question_df.groupby('area')['question_code'].apply(list)
    # scores for anxiety, depression, and stress
    df['dass21_depression_score'] = 2 * df[depression_qs].sum(axis=1)
    df['dass21_anxiety_score'] = 2 * df[anxiety_qs].sum(axis=1)
    df['dass21_stress_score'] = 2 * df[stress_qs].sum(axis=1)

    # total DASS-21 score
    df['dass21_score'] = (df['dass21_anxiety_score'] +
                        df['dass21_depression_score'] +
                        df['dass21_stress_score'])
    pass

def get_depression_dass21_label(score):
    if score <= 9:
        return 'Normal'
    elif score <= 13:
        return 'Mild'
    elif score <= 20:
        return 'Moderate'
    elif score <= 27:
        return 'Severe'
    else:
        return 'Extremely Severe'
    
def get_anxiety_dass21_label(score):
    if score <= 7:
        return 'Normal'
    elif score <= 9:
        return 'Mild'
    elif score <= 14:
        return 'Moderate'
    elif score <= 19:
        return 'Severe'
    else:
        return 'Extremely Severe'
    
def get_stress_dass21_label(score):
    if score <= 14:
        return 'Normal'
    elif score <= 18:
        return 'Mild'
    elif score <= 25:
        return 'Moderate'
    elif score <= 33:
        return 'Severe'
    else:
        return 'Extremely Severe'
    
def get_dass21_label(score):
    # summed up upper ranges for each area
    if score <= 30:
        return 'Normal'
    elif score <= 40:
        return 'Mild'
    elif score <= 59:
        return 'Moderate'
    elif score <= 79:
        return 'Severe'
    else:
        return 'Extremely Severe'

def append_dass21_severity_labels(df):
    # based on the symptom severity table above we can build labels to better understand the scores
    df['dass21_depression_label'] = df['dass21_depression_score'].apply(get_depression_dass21_label)
    df['dass21_anxiety_label'] = df['dass21_anxiety_score'].apply(get_anxiety_dass21_label)
    df['dass21_stress_label'] = df['dass21_stress_score'].apply(get_stress_dass21_label)
    df['dass21_label'] = df['dass21_score'].apply(get_dass21_label)