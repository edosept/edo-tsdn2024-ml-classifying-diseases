import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

def create_dummy_data(n_samples=20000, target_prevalence=0.29):
    """
    Create dummy data emphasizing how poor lifestyle in young people
    can lead to hypertension risk
    """
    np.random.seed(42)
    
    # 1. Generate base population with more young people
    age_distribution = np.concatenate([
        np.random.normal(25, 5, n_samples // 2),  # 50% young adults (20-30)
        np.random.normal(50, 15, n_samples // 2)  # 50% older adults
    ])
    
    data = pd.DataFrame({
        'id': np.arange(1, n_samples + 1),
        'age': age_distribution.clip(15, 90).astype(int),
        'gender': np.random.choice(['Male', 'Female'], n_samples)
    })
    
    # 2. Generate height based on gender
    data['height_cm'] = np.where(
        data['gender'] == 'Male',
        np.random.normal(175, 7, n_samples),
        np.random.normal(162, 6, n_samples)
    )
    
    # 3. Generate weight with higher obesity rates in young people
    def generate_weight(height, age):
        if age < 35:
            # Young people more likely to be overweight
            base_bmi = np.random.normal(28, 5)  # Higher average BMI for young
        else:
            base_bmi = np.random.normal(25, 4)  # More normal BMI for older
        
        return (base_bmi * (height/100)**2)
    
    data['weight_kg'] = data.apply(lambda row: 
        generate_weight(row['height_cm'], row['age']), axis=1)
    
    # 4. Generate belly circumference with higher rates in young
    data['belly_circumference_cm'] = data.apply(lambda row:
        row['weight_kg'] * 0.5 + (60 if row['age'] < 35 else 50) + 
        np.random.normal(0, 5), axis=1)
    
    # 5. Generate lifestyle factors - younger people have worse habits
    def get_lifestyle_probs(age):
        if age < 35:
            # Young people more likely to have poor habits
            smoking_probs = [0.4, 0.1, 0.5]  # Higher smoking rate
            exercise_probs = [0.7, 0.3]      # More sedentary
            salt_probs = [0.3, 0.7]          # Higher salt intake
            sugar_probs = [0.3, 0.7]         # Higher sugar intake
        else:
            # Older people have better habits
            smoking_probs = [0.6, 0.3, 0.1]
            exercise_probs = [0.4, 0.6]
            salt_probs = [0.6, 0.4]
            sugar_probs = [0.6, 0.4]
        return smoking_probs, exercise_probs, salt_probs, sugar_probs
    
    data['smoking_status'] = data.apply(lambda row:
        np.random.choice(['never', 'quit', 'smoker'], 
            p=get_lifestyle_probs(row['age'])[0]), axis=1)
    
    data['exercise_frequency'] = data.apply(lambda row:
        np.random.choice(['low', 'high'],
            p=get_lifestyle_probs(row['age'])[1]), axis=1)
    
    data['salt_consumption'] = data.apply(lambda row:
        np.random.choice(['low', 'high'],
            p=get_lifestyle_probs(row['age'])[2]), axis=1)
    
    data['sugar_consumption'] = data.apply(lambda row:
        np.random.choice(['low', 'high'],
            p=get_lifestyle_probs(row['age'])[3]), axis=1)
    
    # 6. Generate binary columns
    data['self_emotional'] = np.random.binomial(1, 0.5, n_samples)
    data['family_history'] = np.random.binomial(1, 0.3, n_samples)
    
    # 7. Calculate hypertension risk with stronger relationships
    def calculate_risk(row):
        """Calculate hypertension risk with emphasis on lifestyle impact for young people"""
        
        # Calculate BMI
        bmi = row['weight_kg'] / ((row['height_cm']/100) ** 2)
        
        # Initialize risk score components
        genetic_score = 0.0
        lifestyle_score = 0.0
        health_score = 0.0
        
        # 1. Genetic Score
        if row['family_history'] == 1:
            genetic_score += 1.5
            
        # Age scoring with reduced direct impact for young
        if row['age'] >= 60:
            genetic_score += 1.2
        elif row['age'] >= 45:
            genetic_score += 0.9
        elif row['age'] >= 35:
            genetic_score += 0.6
            
        # Gender impact (minor)
        if row['gender'] == 'Male':
            genetic_score += 0.2
        
        # 2. Lifestyle Score (higher impact for young)
        is_young = row['age'] < 35
        
        # Salt consumption (very important for young)
        if row['salt_consumption'] == 'high':
            lifestyle_score += 1.5 if is_young else 1.2
            
        # Sugar consumption
        if row['sugar_consumption'] == 'high':
            lifestyle_score += 1.3 if is_young else 1.0
            
        # Exercise (very important for young)
        if row['exercise_frequency'] == 'low':
            lifestyle_score += 1.4 if is_young else 1.0
            
        # Smoking (higher impact on young)
        if row['smoking_status'] == 'smoker':
            lifestyle_score += 1.2 if is_young else 0.8
        elif row['smoking_status'] == 'quit':
            lifestyle_score += 0.6 if is_young else 0.4
        
        # 3. Health Score (obesity-related)
        # BMI scoring (higher impact on young)
        if bmi >= 35:
            health_score += 1.3 if is_young else 1.1
        elif bmi >= 30:
            health_score += 1.0 if is_young else 0.8
        elif bmi >= 25:
            health_score += 0.7 if is_young else 0.5
            
        # Belly circumference scoring
        waist_threshold = 88 if row['gender'] == 'Female' else 102
        waist_ratio = row['belly_circumference_cm'] / waist_threshold
        if waist_ratio >= 1.2:
            health_score += 1.2 if is_young else 1.0
        elif waist_ratio >= 1.0:
            health_score += 0.9 if is_young else 0.7
        
        # Normalize scores
        genetic_score = np.clip(genetic_score / 3.0, 0, 1)
        lifestyle_score = np.clip(lifestyle_score / (5.0 if is_young else 4.0), 0, 1)  # Higher max for young
        health_score = np.clip(health_score / (2.5 if is_young else 2.1), 0, 1)
        
        # Calculate final risk with age-dependent weights
        if is_young:
            # Young people: lifestyle matters most
            risk = (
                0.25 * genetic_score +   
                0.45 * lifestyle_score + 
                0.30 * health_score
            )
            
            # Strong impact of combined bad habits in young
            if lifestyle_score > 0.6 and health_score > 0.6:
                risk += 0.2  # Big penalty for combined bad lifestyle and obesity in young
                
            if row['family_history'] == 1 and lifestyle_score > 0.6:
                risk += 0.15  # Additional risk for bad lifestyle with family history
                
        else:
            # Older people: more balanced
            risk = (
                0.35 * genetic_score +  
                0.33 * lifestyle_score +  
                0.32 * health_score
            )
            
            # More moderate impact of combinations for older
            if lifestyle_score > 0.7 and health_score > 0.7:
                risk += 0.1
                
            if row['family_history'] == 1 and lifestyle_score > 0.7:
                risk += 0.1
        
        # Small random component
        risk += np.random.normal(0, 0.02)
        
        return np.clip(risk, 0, 1)

    # Calculate probabilities with threshold effects
    probabilities = data.apply(calculate_risk, axis=1)

    # Create even clearer separation with sigmoid transformation
    probabilities = 1 / (1 + np.exp(-12 * (probabilities - 0.5)))

    data['label_hypertension'] = np.random.binomial(1, probabilities)
    
    # 8. Add outliers (2%)
    n_outliers = int(n_samples * 0.02)
    outlier_indices = np.random.choice(n_samples, n_outliers, replace=False)
    
    data.loc[outlier_indices[:n_outliers//4], 'weight_kg'] = \
        np.random.choice([1, 2, 800, 1000], n_outliers//4)
    data.loc[outlier_indices[n_outliers//4:n_outliers//2], 'height_cm'] = \
        np.random.choice([10, 15, 350, 400], n_outliers//4)
    data.loc[outlier_indices[n_outliers//2:3*n_outliers//4], 'belly_circumference_cm'] = \
        np.random.choice([10, 15, 400, 500], n_outliers//4)
    data.loc[outlier_indices[3*n_outliers//4:], 'age'] = \
        np.random.choice([1, 2, 3, 120, 130, 140], n_outliers//4)
    
    # 9. Add timestamps
    start_date = datetime(2023, 1, 1)
    random_days = np.random.randint(0, 365, n_samples).tolist()
    data['input_time'] = [start_date + timedelta(days=x) for x in random_days]
    
    # 10. Add missing values
    missing_probs = {
        'self_emotional': 0.90,
        'sugar_consumption': 0.05,
        'salt_consumption': 0.05,
        'exercise_frequency': 0.07,
        'smoking_status': 0.03,
        'gender': 0.02,
        'weight_kg': 0.03,
        'height_cm': 0.03,
        'belly_circumference_cm': 0.03,
        'family_history': 0.03
    }
    
    for col, prob in missing_probs.items():
        mask = np.random.random(len(data)) < prob
        data.loc[mask, col] = np.nan
        
    # 11. Adjust prevalence if needed
    while abs(data['label_hypertension'].mean() - target_prevalence) > 0.01:
        if data['label_hypertension'].mean() > target_prevalence:
            idx = data[data['label_hypertension'] == 1].sample(1).index[0]
            data.loc[idx, 'label_hypertension'] = 0
        else:
            idx = data[data['label_hypertension'] == 0].sample(1).index[0]
            data.loc[idx, 'label_hypertension'] = 1
    
    return data

def main():
    # Create dummy dataset
    data = create_dummy_data(n_samples=20000, target_prevalence=0.29)
    
    print("\nData Summary:")
    print(f"Total samples: {len(data)}")
    print(f"Hypertension Prevalence: {(data['label_hypertension'].mean()*100):.1f}%")
    
    # Print age group analysis
    print("\nAge Group Analysis:")
    young = data[data['age'] < 35]
    old = data[data['age'] >= 35]
    print(f"Young (<35) hypertension rate: {(young['label_hypertension'].mean()*100):.1f}%")
    print(f"Older (>=35) hypertension rate: {(old['label_hypertension'].mean()*100):.1f}%")
    
    # Print lifestyle impact in young people
    young_healthy = young[
        (young['smoking_status'] == 'never') &
        (young['exercise_frequency'] == 'high') &
        (young['salt_consumption'] == 'low')
    ]
    young_unhealthy = young[
        (young['smoking_status'] == 'smoker') &
        (young['exercise_frequency'] == 'low') &
        (young['salt_consumption'] == 'high')
    ]
    print("\nLifestyle Impact in Young People:")
    print(f"Healthy lifestyle hypertension rate: {(young_healthy['label_hypertension'].mean()*100):.1f}%")
    print(f"Unhealthy lifestyle hypertension rate: {(young_unhealthy['label_hypertension'].mean()*100):.1f}%")
    
    # Save the data
    os.makedirs('dataset', exist_ok=True)
    data.to_csv('dataset/dummy_data.csv', index=False)
    print("\nFile saved: 'dataset/dummy_data.csv'")

if __name__ == "__main__":
    main()