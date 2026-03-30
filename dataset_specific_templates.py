"""
Expanded 20 templates using the original 2-placeholder format.

Key: Uses template.format(feature_desc, task_description)
     NOT template.format(feature_desc) like v2 did.

This preserves the original question-based format that worked better.
"""

def get_templates_and_tokens(dataset_name, task_description):
    """
    Returns 20 templates using the original 2-placeholder format.

    All templates use {} twice: first for feature description, second for task.
    This creates prompts like: "Consider a person who {features}. Is this person {task}? Answer:"

    Args:
        dataset_name: Name of the dataset
        task_description: Task description string (e.g., "have chronic kidney disease")

    Returns:
        templates: List of 20 template strings with 2 placeholders each
        tokens: Tuple of ("Yes", "No")
    """

    # Determine dataset type
    regression_datasets = ['abalone', 'airfoil_noise', 'bike_sharing', 'combined_cycle_power',
                          'communities_crime', 'concrete_strength', 'energy_efficiency', 'forest_fires']
    community_datasets = ['communities_crime']
    object_datasets = ['airfoil_noise', 'concrete_strength', 'combined_cycle_power', 'bike_sharing']

    # Check if regression dataset (takes priority over community/object categorization)
    if dataset_name in regression_datasets:
        # For regression: format(feature_desc, target_desc)
        # Question asks if feature affects target; LLM answers: "increase" or "decrease"
        if dataset_name in community_datasets:
            # Community regression templates
            templates = [
                "In a community where {}, how does this affect {}? Answer:",
                "Consider a community that {}. How does this affect {}? Answer:",
                "Given a community with {}, how does this affect {}? Answer:",
                "A community has {}. How does this affect {}? Answer:",
                "Imagine a community with {}. How does this affect {}? Answer:",
                "A community where {}. How does this feature affect {}? Answer:",
                "Consider a community where {}. How does this affect {}? Answer:",
                "Given that a community {}, how does this affect {}? Answer:",
                "A community that {}. How does this feature affect {}? Answer:",
                "Think about a community with {}. How does this affect {}? Answer:",
            ]
        else:
            # Object regression templates (for airfoil, concrete, bike, power, etc.)
            templates = [
                "If {} increases, does {} increase or decrease? Answer:",
                "When {} increases, does {} increase or decrease? Answer:",
                "As {} increases, does {} increase or decrease? Answer:",
                "With higher {}, does {} increase or decrease? Answer:",
                "If {} goes up, does {} increase or decrease? Answer:",
                "When {} is higher, does {} increase or decrease? Answer:",
                "As {} goes up, does {} increase or decrease? Answer:",
                "With increasing {}, does {} increase or decrease? Answer:",
                "If {} rises, does {} increase or decrease? Answer:",
                "When {} becomes larger, does {} increase or decrease? Answer:",
            ]
        tokens = ("increase", "decrease")

    # Classification datasets (only if not regression)
    elif dataset_name in community_datasets:
        # Community templates (20 variations) - classification
        templates = [
            "Consider a community that {}. Is this community {}? Answer:",
            "A community that {}. Question: Is this community {}? Answer:",
            "Given that a community {}, is this community {}? Answer:",
            "Suppose a community has {}. Does this community {}? Answer:",
            "Imagine a community with {}. Is this community {}? Answer:",
            "A community has {}. Question: Does this community {}? Answer:",
            "Consider a community with {}. Is this community {}? Answer:",
            "Given a community that {}, does this community {}? Answer:",
            "A community where {}. Is this community {}? Answer:",
            "Suppose a community that {}. Does this community {}? Answer:",
            "Think about a community with {}. Is this community {}? Answer:",
            "A community characterized by {}. Is this community {}? Answer:",
            "Consider a community where {}. Does this community {}? Answer:",
            "Given a community with {}, is this community {}? Answer:",
            "A community that has {}. Question: Is this community {}? Answer:",
            "Imagine a community that {}. Does this community {}? Answer:",
            "A community exhibits {}. Is this community {}? Answer:",
            "Consider that a community {}. Is this community {}? Answer:",
            "A community shows {}. Question: Does this community {}? Answer:",
            "Given that a community has {}, is this community {}? Answer:",
        ]
        tokens = ("Yes", "No")

    elif dataset_name in object_datasets:
        # Object templates (20 variations) - classification
        templates = [
            "Consider an object that {}. Does this object {}? Answer:",
            "An object that {}. Question: Does this object {}? Answer:",
            "Given that an object {}, does this object {}? Answer:",
            "Suppose an object has {}. Does this object {}? Answer:",
            "Imagine an object with {}. Does this object {}? Answer:",
            "An object has {}. Question: Does this object {}? Answer:",
            "Consider an object with {}. Does this object {}? Answer:",
            "Given an object that {}, does this object {}? Answer:",
            "An object where {}. Does this object {}? Answer:",
            "Suppose an object that {}. Does this object {}? Answer:",
            "Think about an object with {}. Does this object {}? Answer:",
            "An object characterized by {}. Does this object {}? Answer:",
            "Consider an object where {}. Does this object {}? Answer:",
            "Given an object with {}, does this object {}? Answer:",
            "An object that has {}. Question: Does this object {}? Answer:",
            "Imagine an object that {}. Does this object {}? Answer:",
            "An object exhibits {}. Does this object {}? Answer:",
            "Consider that an object {}. Does this object {}? Answer:",
            "An object shows {}. Question: Does this object {}? Answer:",
            "Given that an object has {}, does this object {}? Answer:",
        ]
        tokens = ("Yes", "No")

    else:
        # Person templates (20 variations)
        templates = [
            "Consider a person who {}. Is this person {}? Answer:",
            "A person who {}. Question: Is this person {}? Answer:",
            "Given that a person {}, is this person {}? Answer:",
            "Suppose a person has {}. Is this person {}? Answer:",
            "Imagine a person with {}. Is this person {}? Answer:",
            "A person has {}. Question: Is this person {}? Answer:",
            "Consider a person with {}. Is this person {}? Answer:",
            "Given a person who {}, is this person {}? Answer:",
            "A person where {}. Is this person {}? Answer:",
            "Suppose a person who {}. Is this person {}? Answer:",
            "Think about a person with {}. Is this person {}? Answer:",
            "A person characterized by {}. Is this person {}? Answer:",
            "Consider a person where {}. Is this person {}? Answer:",
            "Given a person with {}, is this person {}? Answer:",
            "A person that has {}. Question: Is this person {}? Answer:",
            "Imagine a person who {}. Is this person {}? Answer:",
            "A person exhibits {}. Is this person {}? Answer:",
            "Consider that a person {}. Is this person {}? Answer:",
            "A person shows {}. Question: Is this person {}? Answer:",
            "Given that a person has {}, is this person {}? Answer:",
        ]
        tokens = ("Yes", "No")

    return templates, tokens
