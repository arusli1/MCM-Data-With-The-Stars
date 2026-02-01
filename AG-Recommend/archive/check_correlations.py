import pandas as pd
import numpy as np
from run_analysis import load_and_preprocess, feature_engineering

def check_directions():
    df = load_and_preprocess()
    df = feature_engineering(df)
    
    # 1. Age Correlations
    print("\n--- Age Correlations ---")
    print(df[['age', 'avg_judge_score', 'avg_fan_share', 'placement']].corr()['age'])
    
    # 2. Partner Impact (Derek Hough vs Rest)
    print("\n--- Derek Hough Impact ---")
    df['is_derek'] = df['ballroom_partner'] == 'Derek Hough'
    print(df.groupby('is_derek')[['avg_judge_score', 'avg_fan_share', 'placement']].mean())
    
    # 3. Industry Impact (Mean by Industry)
    print("\n--- Industry Impact ---")
    print(df.groupby('industry_clean')[['avg_judge_score', 'avg_fan_share', 'placement']].mean().sort_values('avg_judge_score', ascending=False))

if __name__ == "__main__":
    check_directions()
