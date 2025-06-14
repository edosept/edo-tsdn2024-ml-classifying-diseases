# Machine Learning Project Sharing Session to TSDN 2024

## Overview
This project implements a machine learning model using LightGBM to predict hypertension risk based on various health indicators. The model provides risk probability, risk level classification, and personalized health recommendations.

## ⚠️ Disclaimer
This model is trained on synthetic/dummy data for educational and demonstration purposes only. The predictions and recommendations should not be used as a substitute for professional medical advice. Always consult with healthcare professionals for medical decisions.

## Key Findings
Our model highlights a significant correlation between lifestyle choices and hypertension risk, particularly among young adults:
- Young individuals with poor lifestyle habits (high salt/sugar intake, smoking, low exercise) can have high risk levels
- Family history combined with unhealthy lifestyle choices significantly increases risk
- Regular exercise and healthy dietary habits are crucial preventive factors
- Early intervention and lifestyle modifications are essential for risk reduction

## Features
- Risk probability prediction (0-1 scale)
- Risk level classification (Very Low to Very High)
- Personalized health recommendations
- Detailed prediction reports

## Model Input Parameters
- `family_history` (0 or 1) ~ means no or yes for family history hypertension
- `salt_consumption` (low, high)
- `exercise_frequency` (low, high)
- `belly_circumference_cm` (50-220 cm)
- `sugar_consumption` (low, high)
- `smoking_status` (never, quit, smoker)
- `weight_kg` (50-200 kg)
- `height_cm` (100-220 cm)

## Prediction Output Format
```
📊 Prediction Results:
--------------------------------------------------
Risk Probability: [0-1 scale]
Risk Level: [Very Low/Low/Moderate/High/Very High]
Classification: [Low Risk/High Risk]

💡 Recommendation:
--------------------------------------------------
[Personalized health recommendation based on risk level]

📋 Input Values:
--------------------------------------------------
[List of input parameters and their values]
```

## Risk Level Categories
- Very Low (< 0.2): Maintain healthy lifestyle
- Low (0.2 - 0.4): Continue good habits, regular check-ups
- Moderate (0.4 - 0.6): Consider lifestyle changes, consult healthcare provider
- High (0.6 - 0.8): Immediate lifestyle changes, regular medical monitoring
- Very High (> 0.8): Urgent medical consultation recommended

## Usage in Google Colab
Make predictions by providing your health parameters to the model. Example:
```python
input_data = {
    "family_history": 0,
    "salt_consumption": "high",
    "exercise_frequency": "low",
    "belly_circumference_cm": 99,
    "sugar_consumption": "high",
    "smoking_status": "never",
    "weight_kg": 200.0,
    "height_cm": 165.0
}

results = predict_hypertension_risk("path_to_model.joblib", input_data)
```

## Dependencies
- pandas
- numpy
- scikit-learn
- lightgbm
- joblib

## License  
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for terms.

**Keep the copyright notice if you use this code. Attribution appreciated but not required.**

![MIT License](https://img.shields.io/badge/License-MIT-blue.svg)
