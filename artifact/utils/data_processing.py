"""This file contains functions to load and preprocess data from a CSV file. The data is then filtered based on columns of interest and missing values are filled. The experience level of each job posting is extracted from the job title using a predefined set of keywords. The preprocessed data is then saved to a new CSV file."""

from utils.imports import *  # Import necessary libraries
from utils.config import experience_levels, exp_level_abbr  # Import predefined configurations

def load_data_from_url(url, filename):
    """
    Downloads data from the given URL and saves it to the specified filename.

    Args:
        url (str): The URL of the data to be downloaded.
        filename (str): The name of the file to save the downloaded data.

    Returns:
        None
    """

    if not os.path.exists(filename):  # Check if file already exists
        response = requests.get(url)  # Send HTTP GET request to download data
        with open(filename, 'wb') as f:  # Open file in binary write mode
            f.write(response.content)  # Write downloaded content to file

def extract_data(filename):
    """
    Extracts relevant columns from a CSV file and returns a filtered DataFrame.

    Args:
        filename (str): The path to the CSV file.

    Returns:
        pandas.DataFrame: A DataFrame containing the filtered data.
    """
    # Load data into DataFrame
    jobs_data = pd.read_csv(filename)

    # Columns of interest
    columns_of_interest = [
        "job_title_short", "job_location", "job_via", "job_schedule_type",
        "job_work_from_home", "job_posted_date", "job_skills", "job_country", "search_location", "company_name",
        "job_title",'salary_year_avg' ,'job_no_degree_mention','job_health_insurance'
    ]

    # Filter DataFrame based on columns of interest
    jobs_data_filtered = jobs_data[columns_of_interest]

    return jobs_data_filtered

def preprocess_data(jobs_data):
    """
    Preprocesses the given jobs_data DataFrame by filling missing values, converting data types,
    extracting experience levels from job titles, and creating an abbreviated job title.

    Args:
        jobs_data (pandas.DataFrame): The DataFrame containing job data.

    Returns:
        pandas.DataFrame: The preprocessed jobs_data DataFrame.
    """
    # Fill missing values
    jobs_data['job_work_from_home'] = jobs_data['job_work_from_home'].fillna(True)
    jobs_data['salary_year_avg'] = jobs_data['salary_year_avg'].fillna(0)
    jobs_data['job_posted_date'] = pd.to_datetime(jobs_data['job_posted_date']).dt.date
    # Replace NaN values with an empty string
    jobs_data['job_skills'] = jobs_data['job_skills'].fillna('')
    
    # Ensure all entries in the skills column are strings
    jobs_data['job_skills'] = jobs_data['job_skills'].apply(lambda x: str(x))
    jobs_data['job_title'] = jobs_data['job_title'].fillna('').astype(str)

    # Function to extract experience level from job title
    def get_experience_level(title):
        for level, keywords in experience_levels.items():
            for keyword in keywords:
                if keyword.lower() in title.lower():
                    return level
        return "Entry"  # Default to Entry level if no keywords found

    # Apply the function to extract experience level
    jobs_data['experience_level'] = jobs_data['job_title'].apply(get_experience_level)

    def create_abbreviated_job_title(row):
        """
        Creates an abbreviated job title based on the experience level.

        Args:
            row (pandas.Series): A row of the DataFrame.

        Returns:
            str: The abbreviated job title.
        """
        exp_level = row['experience_level']
        if exp_level in exp_level_abbr:
            return f"{row['job_title_short']} ({exp_level_abbr[exp_level]})"
        return row['job_title_short']

    jobs_data['abbreviated_job_title'] = jobs_data.apply(create_abbreviated_job_title, axis=1)
    print(jobs_data['abbreviated_job_title'])
    print(jobs_data['job_title'])

    return jobs_data
