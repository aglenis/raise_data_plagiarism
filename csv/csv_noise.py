#### INSERT NOISE
#### COSINE SIMILARITY & JACCARD SIMILARITY


import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity

def classify_dataset(df):
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = df.select_dtypes(include=['object', 'category']).columns

    if len(num_cols) > 0 and len(cat_cols) == 0:
        return 'numerical'
    elif len(cat_cols) > 0 and len(num_cols) == 0:
        return 'categorical'
    else:
        return 'mixed'

def check_plagiarism(df1, df2):
    df_type = classify_dataset(df1)

    if df_type == 'numerical':
        num_cols = df1.select_dtypes(include=['int64', 'float64']).columns
        scaler = StandardScaler()

        df1_num = df1[num_cols].dropna()
        df2_num = df2[num_cols].dropna()

        df1_num_scaled = scaler.fit_transform(df1_num)
        df2_num_scaled = scaler.transform(df2_num)

        cosine_sim = cosine_similarity(df1_num_scaled, df2_num_scaled).diagonal()
        return np.mean(cosine_sim)

    elif df_type == 'categorical':
        cat_cols = df1.select_dtypes(include=['object', 'category']).columns
        encoder = OneHotEncoder(handle_unknown='ignore')

        df1_cat = df1[cat_cols].dropna()
        df2_cat = df2[cat_cols].dropna()

        df1_cat_encoded = encoder.fit_transform(df1_cat).toarray()
        df2_cat_encoded = encoder.transform(df2_cat).toarray()

        jaccard_sim = np.array([np.sum(np.logical_and(df1_cat, df2_cat)) / np.sum(np.logical_or(df1_cat, df2_cat)) 
                                for df1_cat, df2_cat in zip(df1_cat_encoded, df2_cat_encoded)])
        return np.mean(jaccard_sim)

    elif df_type == 'mixed':
        num_cols = df1.select_dtypes(include=['int64', 'float64']).columns
        cat_cols = df1.select_dtypes(include=['object', 'category']).columns

        common_rows = df1.dropna(subset=num_cols.tolist() + cat_cols.tolist()).index.intersection(
                      df2.dropna(subset=num_cols.tolist() + cat_cols.tolist()).index)

        df1_num = df1.loc[common_rows, num_cols]
        df2_num = df2.loc[common_rows, num_cols]
        scaler = StandardScaler()
        df1_num_scaled = scaler.fit_transform(df1_num)
        df2_num_scaled = scaler.transform(df2_num)
        cosine_sim = cosine_similarity(df1_num_scaled, df2_num_scaled).diagonal()

        df1_cat = df1.loc[common_rows, cat_cols]
        df2_cat = df2.loc[common_rows, cat_cols]
        encoder = OneHotEncoder(handle_unknown='ignore')
        df1_cat_encoded = encoder.fit_transform(df1_cat).toarray()
        df2_cat_encoded = encoder.transform(df2_cat).toarray()
        jaccard_sim = np.array([np.sum(np.logical_and(df1_cat, df2_cat)) / np.sum(np.logical_or(df1_cat, df2_cat)) 
                                for df1_cat, df2_cat in zip(df1_cat_encoded, df2_cat_encoded)])

        combined_sim = (np.mean(cosine_sim) + np.mean(jaccard_sim)) / 2
        return combined_sim

# TEST
plagiarism_score = check_plagiarism(data, noisy_data)
print("Plagiarism Score:", plagiarism_score)






#### ROW ADDITION
#### Earth Mover's Distance & Cramér's V 


from scipy.stats import wasserstein_distance, chi2_contingency
import pandas as pd
import numpy as np

def emd_between_datasets(df1, df2, num_cols):
    emd_scores = []
    for col in num_cols:
        valid_df1 = df1[col].dropna()
        valid_df2 = df2[col].dropna()

        if not valid_df1.empty and not valid_df2.empty:
            emd_score = wasserstein_distance(valid_df1, valid_df2)
            emd_score_normalized = 1 / (1 + emd_score)
            emd_scores.append(emd_score_normalized)

    return np.mean(emd_scores) if emd_scores else 0

def cramers_v(col1, col2):
    contingency_table = pd.crosstab(col1, col2)
    chi2, _, _, _ = chi2_contingency(contingency_table)
    n = contingency_table.sum().sum()
    phi2 = chi2 / n
    r, k = contingency_table.shape
    phi2_corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))    
    r_corr = r - ((r-1)**2)/(n-1)
    k_corr = k - ((k-1)**2)/(n-1)
    denominator = min((k_corr-1), (r_corr-1))
    if denominator <= 0:
        return 0  
    return np.sqrt(phi2_corr / denominator)

def average_cramers_v(df1, df2, cat_cols):
    scores = []
    for col in cat_cols:
        valid_col1 = df1[col].dropna()
        valid_col2 = df2[col].dropna()

        if not valid_col1.empty and not valid_col2.empty:
            score = cramers_v(valid_col1, valid_col2)
            if not np.isnan(score):
                scores.append(score)

    return np.mean(scores) if scores else 0

def calculate_plagiarism_score(df1, df2):
    num_cols = df1.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = df1.select_dtypes(include=['object', 'category']).columns

    emd_score = emd_between_datasets(df1, df2, num_cols)
    cramers_v_score = average_cramers_v(df1, df2, cat_cols)
    
    return (emd_score + cramers_v_score) / 2


# TEST
plagiarism_score = calculate_plagiarism_score(data, noisy_data)
print("Similarity Score:", plagiarism_score)








#### ROW REMOVING
#### Earth Mover's Distance & Cramér's V 


from scipy.stats import wasserstein_distance

def emd_similarity(df1, df2, num_cols):
    emd_scores = []
    for col in num_cols:
        if col in df2:
            valid_df1 = df1[col].dropna()
            valid_df2 = df2[col].dropna()

            common_indices = valid_df1.index.intersection(valid_df2.index)
            if not common_indices.empty:
                emd = wasserstein_distance(valid_df1[common_indices], valid_df2[common_indices])
                emd_scores.append(1 / (1 + emd))  
    return np.mean(emd_scores) if emd_scores else 1


def average_cramers_v(df1, df2, cat_cols):
    scores = []
    for col in cat_cols:
        if col in df2:
            valid_df1 = df1[col].dropna()
            valid_df2 = df2[col].dropna()

            if not valid_df1.empty and not valid_df2.empty:
                score = cramers_v(valid_df1, valid_df2)
                scores.append(score)
    return np.mean(scores) if scores else 0

def calculate_plagiarism_score(df1, df2):
    num_cols = df1.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = df1.select_dtypes(include=['object', 'category']).columns

    emd_score = emd_similarity(df1, df2, num_cols) if len(num_cols) > 0 else 1
    cramers_v_score = average_cramers_v(df1, df2, cat_cols) if len(cat_cols) > 0 else 1

    return (emd_score + cramers_v_score) / 2


# TEST
plagiarism_score = calculate_plagiarism_score(data, noisy_data)
print("plagiarism Score:", plagiarism_score)
