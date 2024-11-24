"""
    Author: Supriya Jayarj
    Date: 2024-07-19

    This script loads job data from a URL, preprocesses it, generates visualizations. 
"""
# Import necessary modules
from utils.imports import *
from utils.config import *
from utils.data_processing import load_data_from_url, extract_data, preprocess_data
from utils.visualization import plot_job_via_distribution, plot_search_location_vs_job_location, plot_job_schedule_type, plot_health_insurance_and_salary, plot_bubble_chart, plot_calendar_heatmap, plot_waffle_chart, plot_companies_with_most_postings, plot_combined_skills_required, detect_seasonality_in_job_postings, analyze_regional_job_demand, plot_radar_chart, plot_3d_surface, plot_world_map_with_filters, plot_3d_stacked_bar


if __name__ == "__main__":
    # Create a folder to store results if it doesn't exist
    results_folder = "results"
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    print(f"Results will be saved in {results_folder}")
    # URL to download job data
    url = "https://huggingface.co/datasets/lukebarousse/data_jobs/resolve/main/data_jobs.csv?download=true"

    # Filepath to save the downloaded data
    filename = os.path.join(results_folder, "data_jobs.csv")

    # Check if data file exists, if not, download it
    if not os.path.exists(filename):
        load_data_from_url(url, filename)

    # Check if preprocessed data file exists
    preprocessed_filename = os.path.join(results_folder, "data_jobs_preprocessed.csv")
    if os.path.exists(preprocessed_filename):
        print(f"Preprocessed data found as {preprocessed_filename}. Loading...")
        jobs_data_preprocessed = pd.read_csv(preprocessed_filename)
    else:
        # Extract and preprocess data
        jobs_data_filtered = extract_data(filename)
        jobs_data_preprocessed = preprocess_data(jobs_data_filtered)
        jobs_data_preprocessed.to_csv(preprocessed_filename, index=False)
        print(f"Preprocessed data saved as {preprocessed_filename}")

    # A subset of preprocessed data
    jobs_data_sampled = jobs_data_preprocessed.sample(n=20000, random_state=1)

    # Generate visualizations
    print("Generating visualizations...")
    plot_job_via_distribution(jobs_data_sampled)
    plot_search_location_vs_job_location(jobs_data_sampled)
    plot_job_schedule_type(jobs_data_sampled)
    plot_health_insurance_and_salary(jobs_data_sampled)
    plot_bubble_chart(jobs_data_sampled)
    plot_calendar_heatmap(jobs_data_sampled)
    plot_waffle_chart(jobs_data_sampled)
    plot_companies_with_most_postings(jobs_data_sampled)
    plot_combined_skills_required(jobs_data_sampled)
    detect_seasonality_in_job_postings(jobs_data_sampled)
    analyze_regional_job_demand(jobs_data_sampled)
    plot_radar_chart(jobs_data_sampled)
    plot_3d_surface(jobs_data_sampled)
    plot_world_map_with_filters(jobs_data_sampled)
    plot_3d_stacked_bar(jobs_data_sampled)
    print("Visualizations generated successfully!")
