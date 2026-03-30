"""
Contextual preambles for ablation study 2.

Adds domain knowledge context before the feature-effect question.
"""

PREAMBLES = {
    'stroke': {
        'age': "Stroke risk increases significantly with age due to arterial stiffening and accumulated vascular damage. Based on this medical knowledge,",
        'hypertension': "Hypertension is a major risk factor for stroke as high blood pressure damages blood vessel walls. Based on this medical knowledge,",
        'heart_disease': "Heart disease often indicates systemic cardiovascular problems that elevate stroke risk. Based on this medical knowledge,",
        'avg_glucose_level': "Elevated glucose levels damage blood vessels over time, increasing stroke susceptibility. Based on this medical knowledge,",
        'bmi': "Higher BMI is associated with metabolic syndrome and vascular dysfunction that raise stroke risk. Based on this medical knowledge,",
        'smoking_status_smokes': "Smoking accelerates atherosclerosis and increases blood clotting, substantially raising stroke risk. Based on this medical knowledge,"
    },
    'blood': {
        'recency': "People who have donated blood recently are typically more engaged and committed donors. Based on this behavioral pattern,",
        'frequency': "Frequent donors demonstrate a pattern of sustained commitment to blood donation. Based on this behavioral pattern,",
        'monetary': "Donors who have contributed more total blood volume show stronger donation commitment. Based on this behavioral pattern,",
        'time': "Donors with longer donation histories have established habits and are more likely to continue donating. Based on this behavioral pattern,"
    },
    'heart_failure': {
        'age': "Heart failure risk increases with age due to cumulative cardiac stress and declining function. Based on this cardiac physiology,",
        'anaemia': "Anemia reduces oxygen delivery to tissues and can worsen heart failure outcomes. Based on this cardiac physiology,",
        'ejection_fraction': "Higher ejection fraction indicates better cardiac pump function and improved prognosis. Based on this cardiac physiology,",
        'serum_creatinine': "Elevated creatinine signals kidney dysfunction, which commonly accompanies severe heart failure. Based on this cardiac physiology,",
        'diabetes': "Diabetes damages blood vessels and the heart muscle, worsening heart failure prognosis. Based on this cardiac physiology,"
    },
    'adult': {
        'age': "Income typically increases with age due to career advancement and accumulated experience. Based on this economic pattern,",
        'education-num': "Higher education is strongly associated with increased earning potential across occupations. Based on this economic pattern,",
        'hours-per-week': "Working more hours per week generally correlates with higher total income. Based on this economic pattern,",
        'capital-gain': "Capital gains indicate investment income, which is more common among higher earners. Based on this economic pattern,",
        'occupation_Exec-managerial': "Executive and managerial positions typically command higher salaries than other occupations. Based on this economic pattern,"
    }
}

def get_preamble(dataset, feature):
    """
    Get contextual preamble for a dataset and feature.
    Returns empty string if no preamble defined for this feature.
    """
    if dataset in PREAMBLES and feature in PREAMBLES[dataset]:
        return PREAMBLES[dataset][feature]
    return ""
