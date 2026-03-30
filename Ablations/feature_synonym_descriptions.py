"""
Three sets of grammatically correct synonym feature descriptions for ablation study.

Variant 1: Concise/terse (current style - "has X", "is X")
Variant 2: Descriptive/verbose ("has a higher level of X", "is characterized by X")
Variant 3: Clinical/formal ("exhibits X", "demonstrates X", "presents with X")
"""

FEATURE_DESCRIPTIONS = {
    'stroke': {
        'variant1': {  # Current/baseline
            'age': 'is older',
            'hypertension': 'has hypertension',
            'heart_disease': 'has heart disease',
            'avg_glucose_level': 'has higher average glucose level in blood',
            'bmi': 'has higher body mass index',
            'gender_Female': 'is female',
            'gender_Male': 'is male',
            'gender_Other': 'has other gender',
            'ever_married_No': 'has never been married',
            'ever_married_Yes': 'has been married',
            'work_type_Govt_job': 'works in government',
            'work_type_Never_worked': 'has never worked',
            'work_type_Private': 'works in private sector',
            'work_type_Self-employed': 'is self-employed',
            'work_type_children': 'is a child',
            'Residence_type_Rural': 'lives in rural area',
            'Residence_type_Urban': 'lives in urban area',
            'smoking_status_Unknown': 'has unknown smoking status',
            'smoking_status_formerly smoked': 'formerly smoked',
            'smoking_status_never smoked': 'never smoked',
            'smoking_status_smokes': 'currently smokes'
        },
        'variant2': {  # Descriptive/verbose
            'age': 'is of older age',
            'hypertension': 'has a diagnosis of hypertension',
            'heart_disease': 'has a history of heart disease',
            'avg_glucose_level': 'has an elevated average glucose level in blood',
            'bmi': 'has an increased body mass index',
            'gender_Female': 'is a female patient',
            'gender_Male': 'is a male patient',
            'gender_Other': 'has a gender identity other than male or female',
            'ever_married_No': 'has never been married before',
            'ever_married_Yes': 'has been married at some point',
            'work_type_Govt_job': 'is employed by the government',
            'work_type_Never_worked': 'has never held employment',
            'work_type_Private': 'is employed in the private sector',
            'work_type_Self-employed': 'is a self-employed individual',
            'work_type_children': 'is a child without employment',
            'Residence_type_Rural': 'resides in a rural area',
            'Residence_type_Urban': 'resides in an urban area',
            'smoking_status_Unknown': 'has an unknown smoking history',
            'smoking_status_formerly smoked': 'has a history of smoking but quit',
            'smoking_status_never smoked': 'has never smoked tobacco',
            'smoking_status_smokes': 'is an active smoker'
        },
        'variant3': {  # Clinical/formal
            'age': 'exhibits advanced age',
            'hypertension': 'presents with hypertension',
            'heart_disease': 'demonstrates cardiovascular disease',
            'avg_glucose_level': 'presents with elevated blood glucose levels',
            'bmi': 'demonstrates elevated body mass index',
            'gender_Female': 'is of female sex',
            'gender_Male': 'is of male sex',
            'gender_Other': 'is of non-binary sex',
            'ever_married_No': 'reports never being married',
            'ever_married_Yes': 'reports marriage history',
            'work_type_Govt_job': 'reports government employment',
            'work_type_Never_worked': 'reports no employment history',
            'work_type_Private': 'reports private sector employment',
            'work_type_Self-employed': 'reports self-employment status',
            'work_type_children': 'is of pediatric age',
            'Residence_type_Rural': 'reports rural residence',
            'Residence_type_Urban': 'reports urban residence',
            'smoking_status_Unknown': 'has undocumented smoking status',
            'smoking_status_formerly smoked': 'reports previous tobacco use',
            'smoking_status_never smoked': 'reports no tobacco use history',
            'smoking_status_smokes': 'reports current tobacco use'
        }
    },
    'blood': {
        'variant1': {  # Current/baseline
            'recency': 'has fewer months since last donation',
            'frequency': 'has higher total number of donations',
            'monetary': 'has donated more blood in total',
            'time': 'has more months since first donation'
        },
        'variant2': {  # Descriptive/verbose
            'recency': 'has donated blood more recently',
            'frequency': 'has donated blood on more occasions',
            'monetary': 'has contributed a greater total volume of blood',
            'time': 'has been donating blood for a longer period of time'
        },
        'variant3': {  # Clinical/formal
            'recency': 'demonstrates recent donation activity',
            'frequency': 'exhibits frequent donation behavior',
            'monetary': 'demonstrates substantial cumulative donation volume',
            'time': 'reports extended donation history duration'
        }
    },
    'heart_failure': {
        'variant1': {  # Current/baseline
            'age': 'is older',
            'anaemia': 'has lower hemoglobin levels',
            'creatinine_phosphokinase': 'has higher CPK enzyme levels',
            'diabetes': 'has elevated blood glucose',
            'ejection_fraction': 'has higher ejection fraction',
            'high_blood_pressure': 'has elevated blood pressure',
            'platelets': 'has higher platelet count',
            'serum_creatinine': 'has higher serum creatinine',
            'serum_sodium': 'has higher serum sodium',
            'sex': 'is male',
            'smoking': 'is a smoker',
            'time': 'has been monitored for longer'
        },
        'variant2': {  # Descriptive/verbose
            'age': 'is of older age',
            'anaemia': 'has a reduced hemoglobin concentration',
            'creatinine_phosphokinase': 'has an elevated CPK enzyme level in blood',
            'diabetes': 'has a diagnosis of diabetes mellitus',
            'ejection_fraction': 'has a greater cardiac ejection fraction',
            'high_blood_pressure': 'has a diagnosis of high blood pressure',
            'platelets': 'has an increased platelet count in blood',
            'serum_creatinine': 'has an elevated serum creatinine level',
            'serum_sodium': 'has an elevated serum sodium concentration',
            'sex': 'is a male patient',
            'smoking': 'has a history of smoking',
            'time': 'has been in the study for a longer duration'
        },
        'variant3': {  # Clinical/formal
            'age': 'exhibits advanced age',
            'anaemia': 'presents with anemia',
            'creatinine_phosphokinase': 'demonstrates elevated creatinine phosphokinase',
            'diabetes': 'presents with diabetes mellitus',
            'ejection_fraction': 'demonstrates preserved ejection fraction',
            'high_blood_pressure': 'presents with hypertension',
            'platelets': 'demonstrates thrombocytosis',
            'serum_creatinine': 'presents with elevated serum creatinine',
            'serum_sodium': 'demonstrates hypernatremia',
            'sex': 'is of male sex',
            'smoking': 'reports tobacco use',
            'time': 'exhibits extended follow-up duration'
        }
    },
    'adult': {
        'variant1': {  # Current/baseline
            'age': 'age in years',
            'fnlwgt': 'final weight',
            'education-num': 'years of education',
            'capital-gain': 'capital gains',
            'capital-loss': 'capital losses',
            'hours-per-week': 'hours worked per week',
            'workclass_Private': 'work class being private sector',
            'workclass_Self-emp-not-inc': 'work class being self-employed not incorporated',
            'education_Bachelors': 'education level being bachelors degree',
            'marital-status_Married-civ-spouse': 'marital status being married to civilian spouse',
            'marital-status_Never-married': 'marital status being never married',
            'occupation_Exec-managerial': 'occupation being executive or managerial',
            'occupation_Prof-specialty': 'occupation being professional specialty',
            'relationship_Husband': 'relationship being husband',
            'sex_Male': 'sex being male'
        },
        'variant2': {  # Descriptive/verbose
            'age': 'is of a certain age in years',
            'fnlwgt': 'has a particular census final weight',
            'education-num': 'has completed a certain number of years of education',
            'capital-gain': 'has received capital gains in income',
            'capital-loss': 'has experienced capital losses',
            'hours-per-week': 'works a certain number of hours per week',
            'workclass_Private': 'is employed in the private sector',
            'workclass_Self-emp-not-inc': 'is self-employed without incorporation',
            'education_Bachelors': 'holds a bachelor\'s degree',
            'marital-status_Married-civ-spouse': 'is married to a civilian spouse',
            'marital-status_Never-married': 'has never been married',
            'occupation_Exec-managerial': 'works in an executive or managerial role',
            'occupation_Prof-specialty': 'works in a professional specialty occupation',
            'relationship_Husband': 'is a husband in the household',
            'sex_Male': 'is a male individual'
        },
        'variant3': {  # Clinical/formal
            'age': 'reports age measured in years',
            'fnlwgt': 'demonstrates census final weight value',
            'education-num': 'reports educational attainment in years',
            'capital-gain': 'reports investment income gains',
            'capital-loss': 'reports investment income losses',
            'hours-per-week': 'reports weekly work hours',
            'workclass_Private': 'reports private sector employment',
            'workclass_Self-emp-not-inc': 'reports unincorporated self-employment',
            'education_Bachelors': 'reports bachelor\'s degree attainment',
            'marital-status_Married-civ-spouse': 'reports married status with civilian spouse',
            'marital-status_Never-married': 'reports unmarried status',
            'occupation_Exec-managerial': 'reports executive or managerial occupation',
            'occupation_Prof-specialty': 'reports professional specialty occupation',
            'relationship_Husband': 'reports husband relationship role',
            'sex_Male': 'reports male sex'
        }
    }
}


def get_feature_description(dataset, feature, variant='variant1'):
    """
    Get feature description for given dataset, feature, and variant.

    Args:
        dataset: Dataset name (stroke, blood, heart_failure, adult)
        feature: Feature name
        variant: One of 'variant1', 'variant2', 'variant3'

    Returns:
        Feature description string
    """
    if dataset not in FEATURE_DESCRIPTIONS:
        raise ValueError(f"Unknown dataset: {dataset}")
    if variant not in FEATURE_DESCRIPTIONS[dataset]:
        raise ValueError(f"Unknown variant: {variant}")
    if feature not in FEATURE_DESCRIPTIONS[dataset][variant]:
        raise ValueError(f"Unknown feature {feature} for dataset {dataset}")

    return FEATURE_DESCRIPTIONS[dataset][variant][feature]
