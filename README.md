[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/V3NiVcOe)


## Job Market Analysis 

The project includes a comprehensive data analysis and visalization. 

- Data analysis: Jupyter notebook found in ```artifact/job_market.ipynb```
- Data Visualisation: Found in ```artifact/main.py```


## Data Analysis:

The main analysis function performs the following:
- Computes descriptive statistics.
- Analyzes salary trends by location.
- Counts job postings by company.
- Trains a Random Forest Regressor for salary prediction.
## Data Quality Report:

- An interactive widget is provided for generating a detailed data quality report.
- The report includes information on missing values, data types, outliers, and value ranges.

## Interactive Widget:

- Interactive widgets allow users to explore job data based on selected skills and experience levels.
- The user can select multiple skills by using ```CMD + skills ``` on Mac or ```Ctrl + skills ```  on Windows 
- The user can also explore salary predicted based on the experince level
- A button to generate and display the full analysis report is provided.

### Output

The analysis function returns a comprehensive report including:
- Descriptive statistics of the dataset
- Average salaries by location
- Job counts by company 
- Experience Level Analysis

This analysis provides valuable insights into the job market, offers a recommendation of Jobs based on skills and  predictive model for estimating salaries based on job characteristics.



## Setup

1. Clone the repository:

    ```bash
    git clone https://github.com/freiburg-missing-semester-course/project-supu18-1.git
    cd artifact
    ```

2. Create and activate a virtual environment:

    ```bash
    python3 -m venv env
    source env/bin/activate  # On Windows use `env\Scripts\activate`
    ```

3. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```
    or 

   ```bash
    ./start.sh
    ```

     ```bash
    pip install bs4 
    ```
    
## Usage

1. Run the main script:

    ```bash
    python main.py
    ```

This will load the data from URL, preprocess the data and generate visualizations. The data collection from the URL might take few minutes. Due to some reason if loading does not happen then extract and copy the ```data_jobs.csv``` & ```data_jobs_preprocessed.csv``` files from the ```artifact/dataset/Job_data.zip``` and paste into the results folder. These data are necessary and used for further analysis and plotting

## Project Components

### Configuration

The `artifact/utils/config.py` script contains configuration settings, allowing for easy adjustments.

### Data Processing

The `artifact/utils/data_processing.py` script handles preprocessing tasks such as handling missing values, encoding categorical variables, and converting columns to appropriate data types.

### Visualizations

The `artifact/utils/visualization.py` script generates various visualizations to analyze job types, experience levels, and geographical distributions. Generated plots are saved in the `artifact/images/` folder.

### Jupyter Notebook

The project includes a Jupyter notebook for in-depth analysis of job data, statistics, and model training. To use the notebook:

1. Install Jupyter:

    ```bash
    pip install jupyter
    ```

2. Launch Jupyter Notebook:

    ```bash
    jupyter notebook
    ```

3. Open the notebook ```job_market.ipynb``` job file in your VScode or Colab and run the cells.

### Recommended : 
You can directly extract ```data_jobs_preprocessed_analysis.csv``` from the ```artifact/dataset/data_jobs_preprocessed_analysis.zip``` and paste it in results folder to skip computation of feature addition of data before you run the notebook.I recommend this step as preprocessing for analysis of data may take lot of time.

Note : If using VSCode to run notebook downgrade Jupyter version to below specified version 
``` 
Name: Jupyter
Id: ms-toolsai.jupyter
Description: Jupyter notebook support, interactive programming and computing that supports Intellisense, debugging and more.
Version: 2023.7.1002162226
Publisher: Microsoft
VS Marketplace Link: https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter
```

### Results 

The `artifact/results` folder contains the CSV files and the HTML and JSON reports generated after analysis.

### Docker
1.Pull the Docker image:
    ```bash
        docker pull supu18/project-supu18-1
    ```
2.Note:

If you face any issues while running docker image, please directly run the application with the python commands mentioned.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contributions

You are welcome to contribute to this project by opening an issue regarding any problems that you encounter or suggestions for improvement. Pull requests are also welcome.

## Acknowledgements

This project has been influenced by various tutorials, online resources, and open-source projects related to job data analysis and machine learning. Special thanks go out to the authors and contributors of these resources.
