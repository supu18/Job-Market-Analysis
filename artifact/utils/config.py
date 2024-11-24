# utils/config.py
"""This file contains the configuration settings for the project. It defines experience level keywords and abbreviations used in the project."""
from utils.imports import *

# Define experience level keywords
experience_levels = {
        "Senior": ["Senior", "Sr."],
        "Lead": ["Lead"],
        "Team Lead": ["Team Lead"],
        "Tech Lead": ["Tech Lead", "Technical Lead"],
        "Manager": ["Manager", "Mgr"],
        "Principle Associate": ["Principle Associate", "Pr Ass.", "Principal Associate"],
        "Associate": ["Associate", "Ass."]
}

exp_level_abbr = {
                "Senior": "Sr.",
                "Lead": "Lead",
                "Team Lead": "TL",
                "Tech Lead": "TL",
                "Manager": "Mgr",
                "Principle Associate": "Pr Ass.",
                "Associate": "Ass.",
                "Entry": "Entry"
}
