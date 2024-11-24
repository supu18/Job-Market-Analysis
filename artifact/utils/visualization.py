"""
This module contains functions to visualize the job data 
using various plots and charts.
"""
from numpy import save
from utils.imports import * 

def save_plot(filename):
    """Save the plot to the images directory."""
    if not os.path.exists("images"):
        os.makedirs("images")
    plt.savefig(f"images/{filename}", bbox_inches="tight")
    plt.show()
    plt.close()

def plot_job_via_distribution(jobs_data):
    """
    Plots a pie chart to visualize the distribution of job postings by source.

    Parameters:
    - jobs_data (DataFrame): The DataFrame containing job data.

    Returns:
    None
    """
    job_counts = jobs_data['job_via'].value_counts().sort_values(ascending=False).head(10)
    fig, ax = plt.subplots(figsize = (15, 9), subplot_kw=dict(aspect = "equal"))
    ax.pie(job_counts, labels = job_counts.index, autopct = '%1.1f%%')
    ax.set_title("Top 10 Distribution of Job Postings by Source", fontsize=16)
    ax.axis('equal')

    plt.rcParams.update({'font.size': 14})
    
    save_plot("job_via_distribution.png")
    plt.show()

def plot_search_location_vs_job_location(jobs_data):
    """
    Plot the job location distribution with degree mention indicator on the world map. 
    Args:
        jobs_data (pd.DataFrame): The job data containing 'job_location' and 'job_no_degree_mention' columns.
    """
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    world = world[(world.pop_est > 0) & (world.name != "Antarctica")]
    job_counts = jobs_data.groupby('job_location')['job_no_degree_mention'].mean().reset_index()
    job_counts['degree_mention'] = job_counts['job_no_degree_mention'].apply(lambda x: 'Yes' if x < 0.5 else 'No')
    world = world.merge(job_counts, how = 'left', left_on = 'name', right_on = 'job_location')
    world['degree_mention'] = world['degree_mention'].fillna('No data')
    color_map = {'Yes': 'red', 'No': 'blue', 'No data': 'lightgrey'}
    world['color'] = world['degree_mention'].map(color_map)
    fig, ax = plt.subplots(1, 1, figsize = (15, 9)) 
    world.plot(ax=ax, color=world['color'], edgecolor='black', legend=True)
    plt.rcParams.update({'font.size': 14})
    legend_elements = [Patch(facecolor='red', edgecolor='black', label='Degree Mentioned'),
                       Patch(facecolor='blue', edgecolor='black', label='No Degree Mentioned'),
                       Patch(facecolor='lightgrey', edgecolor='black', label='No Data')]
    ax.legend(handles=legend_elements, loc='lower left', fontsize=12)   
    plt.title('Degree Mention Indicator by Job Location', fontsize=20, pad=20)
    plt.axis('off')  
    plt.tight_layout()
    save_plot('search_location_vs_job_location.png')
    plt.show()

def plot_job_schedule_type(jobs_data):
    """
    Plots a squarify chart to visualize the distribution of top 10 job schedule types.

    Parameters:
    - jobs_data (pandas.DataFrame): DataFrame containing job data.

    Returns:
    None
    """
    schedule_counts = jobs_data['job_schedule_type'].value_counts().nlargest(10)
    total_jobs = sum(schedule_counts.values)
    fig, ax = plt.subplots(figsize = (15, 9))
    # Plot squarify chart
    squarify.plot(sizes=schedule_counts.values, 
            label=[f"{type}\n{count} ({count/total_jobs:.1%})" 
                for type, count in schedule_counts.items()],
            alpha=.8,
            color=plt.cm.tab20c.colors,
            text_kwargs={'fontsize': 10, 'color': 'black', 'wrap': True, 'verticalalignment': 'center'})
    plt.title("Distribution of Top 10 Job Schedule Types", fontsize = 16)
    plt.axis('off') 
    plt.text(0.5, 0.01, f'Total Jobs in Top 10 Categories: {total_jobs}',
          horizontalalignment='center',
          verticalalignment='bottom',
          transform=plt.gca().transAxes,
          fontsize=8) 
    plt.tight_layout()
    save_plot("job_schedule_type.png") 
    plt.show()

def plot_health_insurance_and_salary(jobs_data):
    """
    Plot the salary distribution by health insurance availability with larger fonts.
    
    Args:
        jobs_data (pd.DataFrame): The job data containing 'job_health_insurance' and 'salary_year_avg' columns.
    """
    plt.figure(figsize=(15, 9)) 
    plt.rcParams.update({'font.size': 14})  
    sns.violinplot(x="job_health_insurance", y="salary_year_avg", data=jobs_data)
    plt.title("Salary Distribution by Health Insurance Availability", fontsize=18, pad=20)
    plt.xlabel("Health Insurance", fontsize = 16, labelpad = 15)
    plt.ylabel("Average Annual Salary", fontsize = 16, labelpad = 15)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize = 14)
    plt.text(0.05, 0.95, "False: Health Insurance\nnot available", 
             transform=plt.gca().transAxes, 
             verticalalignment='top', fontsize=12)
    plt.text(0.95, 0.95, "True: Health Insurance\navailable", 
             transform=plt.gca().transAxes, 
             verticalalignment='top', horizontalalignment='right', fontsize=12)
    plt.tight_layout()
    save_plot("health_insurance_and_salary.png")
    plt.show()

def plot_bubble_chart(data):
    """
    Plot a bubble chart of top companies by job postings and average salary with improved readability.
    
    Args:
        data (pd.DataFrame): The job data containing 'company_name' and 'salary_year_avg' columns.
    """
    top_companies = data["company_name"].value_counts().nlargest(10)
    company_names = top_companies.index
    num_postings = top_companies.values

    avg_salary = data[data["company_name"].isin(company_names)].groupby("company_name")["salary_year_avg"].mean().reindex(company_names)
    plt.figure(figsize = (15, 9)) 
    plt.rcParams.update({'font.size': 12})
    scatter = plt.scatter(avg_salary, num_postings, s=num_postings * 15, alpha=0.6, 
                          c=np.random.rand(len(company_names)), cmap='viridis', marker='o')
    plt.xlabel("Average Salary (in $1000)", fontsize=14, labelpad=10)
    plt.ylabel("Number of Postings", fontsize=14, labelpad=10)
    plt.title("Bubble Chart of Top Companies by Job Postings and Average Salary", fontsize=16, pad=20)
    cbar = plt.colorbar(scatter, label='Random Color Indicator')
    cbar.ax.set_ylabel('Random Color Indicator', fontsize=12, rotation=270, labelpad=20)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    for i, company in enumerate(company_names):
        x = avg_salary[i]
        y = num_postings[i]
        plt.annotate(company, (x, y), fontsize=10, ha='center', va='bottom', 
                     xytext=(0, 5), textcoords='offset points',
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.7))
    plt.tight_layout()
    save_plot("bubble_chart.png")
    plt.show()

def plot_calendar_heatmap(data):
    """
    Plot a calendar heatmap of job postings with improved readability.
    
    Args:
        data (pd.DataFrame): The job data containing 'job_posted_date' column.
    """
    data['job_posted_date'] = pd.to_datetime(data['job_posted_date'])
    data['year'] = data['job_posted_date'].dt.year
    data['month'] = data['job_posted_date'].dt.month
    data['day'] = data['job_posted_date'].dt.day
    job_trends = data.groupby(['year', 'month', 'day']).size().reset_index(name = 'counts')
    fig, axes = plt.subplots(4, 3, figsize = (15, 9), sharex = True, sharey = True)
    fig.suptitle('Calendar Heatmap of Job Postings Over Months', y = 1.02, fontsize = 24)

    # Set global font sizes
    plt.rcParams.update({'font.size': 10})

    for (year, month), month_data in job_trends.groupby(['year', 'month']):
        days_in_month = monthrange(year, month)[1]

        month_data = month_data.set_index('day').reindex(range(1, days_in_month + 1), fill_value=0)
        flattened_counts = month_data['counts'].values.flatten()
        num_days = len(flattened_counts)
        num_weeks = int(np.ceil(num_days / 7))
        total_slots = num_weeks * 7
        if num_days < total_slots:
            flattened_counts = np.pad(flattened_counts, (0, total_slots - num_days), 'constant')
        heatmap_data = flattened_counts.reshape(num_weeks, 7)
        row_idx, col_idx = divmod(month - 1, 3)  
        ax = axes[row_idx, col_idx] 
        sns.heatmap(heatmap_data, annot=True, fmt="d", cmap='YlGnBu', cbar=False, 
                    linewidths=.5, linecolor='gray', ax=ax, annot_kws={'size': 8})
        ax.set_title(f"{month_name[month]} {year}", fontsize=14, pad=10)
        ax.set_yticks([])
        ax.set_xticks(range(7))
        ax.set_xticklabels(list(day_abbr)[:days_in_month], rotation=0, fontsize=10)
    # Added a colorbar to the entire figure
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    sm = plt.cm.ScalarMappable(cmap='YlGnBu', norm=plt.Normalize(vmin=0, vmax=job_trends['counts'].max()))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Number of Job Postings', fontsize=14, labelpad=15)
    cbar.ax.tick_params(labelsize=12)
    # Adjusted layout to prevent overlap of titles
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    save_plot("calendar_heatmap.png")
    plt.show()

def plot_waffle_chart(data):
    """
    Plot a waffle chart of top 10 job titles by count.
    
    Args:
        data (pd.DataFrame): The job data containing 'job_title_short' column.
    """
    job_counts = data["job_title_short"].value_counts().nlargest(10)
    data_dict = job_counts.to_dict()

    rows = 20
    cols = 20

    fig = plt.figure(
        FigureClass=Waffle,
        rows=rows,
        columns=cols,
        values=data_dict,
        figsize=(15, 9),
        colors=sns.color_palette("colorblind", len(data_dict))
    )

    plt.suptitle('Top 10 Job Titles by Count', fontsize=16)
    plt.title('Each square represents one job posting', fontsize=12)
    plt.text(0.5, -0.05, f'Total jobs shown: {sum(data_dict.values())}', 
             horizontalalignment='center', verticalalignment='center', 
             transform = plt.gca().transAxes)
    cmap = ListedColormap(sns.color_palette("colorblind", len(data_dict)))
    sm = plt.cm.ScalarMappable(cmap = cmap, norm = plt.Normalize(0, len(data_dict)))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax = fig.gca(), fraction = 0.03, pad=0.04, ticks = range(len(data_dict)))
    cbar.set_ticklabels(list(data_dict.keys()))
    plt.tight_layout() 
    save_plot("waffle_chart.png")
    plt.show()

def plot_companies_with_most_postings(data):
    """
    Plot the top 10 companies with the most job postings.
    
    Args:
        data (pd.DataFrame): The job data containing 'company_name' column.
    """
    company_counts = data['company_name'].value_counts().head(10)
    plt.figure(figsize = (15, 9))
    sns.barplot(x=company_counts.values, y = company_counts.index, palette = 'viridis')
    plt.title("Top 10 Companies with Most Job Postings")
    plt.xlabel("Number of Postings")
    plt.ylabel("Company Name")
    save_plot("companies_with_most_postings.png")
    plt.show()

def plot_combined_skills_required(data):
    """
    Plots a bar chart showing the top skills required for different jobs based on the given data.

    Parameters:
    data (DataFrame): The data containing the job skills information.

    Returns:
    None
    """
    data['job_skills'] = data['job_skills'].str.strip("[]").str.replace("'", "").str.split(", ")
    skills = data['job_skills'].explode().value_counts().nlargest(10)
    
    plt.figure(figsize = (15, 9))
    skills.plot(kind='barh', color='orange')
    plt.title('Top Skills Required for Different Jobs')
    plt.xlabel('Number of Jobs')
    plt.ylabel('Skill')
    plt.tight_layout()
    save_plot('combined_skills_required.png')
    plt.show()

def detect_seasonality_in_job_postings(data):
    """
    Detect seasonality in job postings over time.
    
    Args:
        data (pd.DataFrame): The job data containing 'job_posted_date' column.
    """
    data['job_posted_date'] = pd.to_datetime(data['job_posted_date'])
    data['month'] = data['job_posted_date'].dt.month
    data['year'] = data['job_posted_date'].dt.year
    monthly_counts = data.groupby(['year', 'month']).size().unstack(fill_value=0)
    
    plt.figure(figsize = (15, 9))
    sns.heatmap(monthly_counts, cmap='coolwarm', annot=True, fmt="d")
    plt.title("Seasonality in Job Postings Over Time")
    plt.xlabel("Month")
    plt.ylabel("Year")
    save_plot("seasonality_in_job_postings.png")
    plt.show()

def analyze_regional_job_demand(data):
    """
    Analyzes the regional job demand based on the provided data.

    Args:
        data (pandas.DataFrame): The data containing job locations.

    Returns:
        None
    """
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    world = world[(world.pop_est > 0) & (world.name != "Antarctica")]
    
    # Merge job data with world data
    job_counts = data['job_location'].value_counts()
    job_counts = job_counts[job_counts > 0]
    job_counts = job_counts.reset_index()
    job_counts.columns = ['name', 'job_count']
    
    world = world.merge(job_counts, how='left', left_on='name', right_on='name')
    world['job_count'] = world['job_count'].fillna(0)
    
    fig, ax = plt.subplots(1, 1, figsize=(15, 9))
    world.boundary.plot(ax=ax, linewidth=0.5, color='black')
    world.plot(column='job_count', ax=ax, legend=True, 
               legend_kwds={'label': "Number of Job Postings", 'orientation': "horizontal", 'shrink': 0.6, 'aspect': 20, 'pad': 0.08},
               cmap='YlOrRd', missing_kwds={'color': 'lightgrey'})
    
    plt.title('Regional Job Demand', fontsize=24, pad=20)
    ax.axis('off')
    # Annotate the map with job counts
    for idx, row in world.iterrows():
        if row['job_count'] > 0:
            ax.annotate(f"{row['name']}: {int(row['job_count'])}", 
                        xy=row['geometry'].centroid.coords[0], 
                        ha='center', va='center', fontsize=8,
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7))
    
    plt.tight_layout()
    save_plot('regional_job_demand.png')
    plt.show()

def plot_3d_surface(data):
    """
    Plots a 3D surface plot of the average salary by country and month.

    Args:
        data (DataFrame): The input data containing job information.

    Returns:
        None
    """
    data['posted_month'] = pd.to_datetime(data['job_posted_date']).dt.to_period('M')
    top_countries = data['job_country'].value_counts().nlargest(10).index
    filtered_data = data[data['job_country'].isin(top_countries)]

    grouped_data = filtered_data.groupby(['job_country', 'posted_month']).agg({
        'salary_year_avg': 'mean',
        'job_no_degree_mention': lambda x: (x == 'Yes').mean() * 100
    }).reset_index()

    pivot_salary = grouped_data.pivot(index='job_country', columns='posted_month', values='salary_year_avg')
    # pivot_degree = grouped_data.pivot(index='job_country', columns='posted_month', values='job_no_degree_mention')

    fig = plt.figure(figsize = (15, 9)) 
    ax = fig.add_subplot(111, projection='3d')

    X, Y = np.meshgrid(np.arange(pivot_salary.shape[1]), np.arange(pivot_salary.shape[0]))
    surf = ax.plot_surface(X, Y, pivot_salary.values, cmap='viridis', 
                           linewidth=0, antialiased=False, alpha=0.8)
    
    plt.rcParams.update({'font.size': 12})

    ax.set_xlabel('Month', fontsize=14, labelpad=30)  
    ax.set_ylabel('Country', fontsize=14, labelpad=30)
    ax.set_zlabel('Average Salary', fontsize=14, labelpad=30)
    
    ax.set_xticks(np.arange(0, pivot_salary.shape[1], 3))
    ax.set_xticklabels(pivot_salary.columns[::3].strftime('%Y-%m'), rotation=45, ha='right', fontsize=10)
    ax.set_yticks(np.arange(pivot_salary.shape[0]))
    ax.set_yticklabels(pivot_salary.index, fontsize=10)

    ax.tick_params(axis='x', which='major', pad=8)
    ax.tick_params(axis='y', which='major', pad=8)
    ax.tick_params(axis='z', which='major', pad=8)

    ax.view_init(elev=20, azim=45)

    cbar = fig.colorbar(surf, shrink=0.6, aspect=10, pad=0.1, label='Average Salary')
    cbar.ax.set_ylabel('Average Salary', fontsize=12, labelpad=20)

    ax.set_title('3D Surface Plot of Average Salary by Country and Month', fontsize=16, pad=20)

    plt.tight_layout()
    save_plot("3d_surface.png")
    plt.show()

def plot_world_map_with_filters(jobs_data):
    """
    Plots a world map with filters for different types of jobs.

    Args:
        jobs_data (DataFrame): The jobs data containing job locations and other information.

    Returns:
        None
    """
    
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    merged = world.set_index('name').join(jobs_data.set_index('job_location'))
    fig, ax = plt.subplots(figsize=(15, 9))
    rax = plt.axes([0.05, 0.3, 0.12, 0.3])
    
    # Create CheckButtons 
    check = CheckButtons(rax, ('Remote Jobs', 'Onsite Jobs'), (True, True))
    
    # Adjust font size and check box size
    for label in check.labels:
        label.set_fontsize(14)
    for rect in check.rectangles:
        rect.set_width(0.06)
        rect.set_height(0.06)

    # Customize the checkmark size
    for lines in check.lines:
        for line in lines:
            line.set_linewidth(1.5)  # Reduce line width
            
    def update_checkmarks(checkbox):
        for rect, lines in zip(checkbox.rectangles, checkbox.lines):
            x, y = rect.get_xy()
            w, h = rect.get_width(), rect.get_height()
            
            # Adjust these values to fine-tune the checkmark position and size
            lines[0].set_xdata([x + 0.5 * w, x + 0.9 * w])
            lines[1].set_xdata([x + 0.8 * w, x + 0.2 * w])
            lines[0].set_ydata([y + 0.5 * h, y + 0.9 * h])
            lines[1].set_ydata([y + 0.8 * h, y + 0.2 * h])
            
            lines[0].set_visible(checkbox.get_status()[checkbox.lines.index(lines)])
            lines[1].set_visible(checkbox.get_status()[checkbox.lines.index(lines)])

    update_checkmarks(check)

    def plot_jobs():
        ax.clear()
        world.plot(ax=ax, color='lightgrey')
        
        show_remote, show_onsite = check.get_status()
        
        if show_remote:
            remote_jobs = merged[merged['job_work_from_home'].fillna(True)]
            remote_jobs.plot(ax=ax, color='darkturquoise', marker='o', markersize=5, label='Remote Jobs')
        
        if show_onsite:
            onsite_jobs = merged[merged['job_work_from_home'] == False]
            onsite_jobs.plot(ax=ax, color='darkcyan', marker='o', markersize=5, label='Onsite Jobs')

        ax.set_title('Job Locations')
        ax.legend()

    plot_jobs()

    def on_click(label):
        plot_jobs()
        plt.draw()

    check.on_clicked(on_click)
    plt.subplots_adjust(left=0.22)
    save_plot("world_map_with_filters.png")
    plt.show()

def plot_3d_stacked_bar(data):
    """
    Plots a 3D stacked bar chart of job counts by title, location, and schedule type.

    Parameters:
    - data: DataFrame containing job data

    Returns:
    None
    """
    top_jobs = data['job_title_short'].value_counts().nlargest(5).index
    top_locations = data['job_location'].value_counts().nlargest(5).index
    filtered_data = data[data['job_title_short'].isin(top_jobs) & data['job_location'].isin(top_locations)]
    job_counts = filtered_data.groupby(['job_title_short', 'job_location', 'job_schedule_type']).size().unstack(fill_value=0)

    fig = plt.figure(figsize=(15, 9))
    ax = fig.add_subplot(111, projection='3d')

    x_pos = np.arange(len(top_jobs))
    y_pos = np.arange(len(top_locations))
    x_pos, y_pos = np.meshgrid(x_pos, y_pos)
    x_pos = x_pos.flatten()
    y_pos = y_pos.flatten()

    z_pos = np.zeros_like(x_pos)
    dx = dy = 0.75

    colors = plt.cm.Spectral(np.linspace(0, 1, len(job_counts.columns)))

    for i, schedule_type in enumerate(job_counts.columns):
        dz = job_counts[schedule_type].values.flatten()
        ax.bar3d(x_pos, y_pos, z_pos, dx, dy, dz, color=colors[i], alpha=0.8, shade=True)
        z_pos += dz

    plt.rcParams.update({'font.size': 12})
    ax.set_xticklabels(top_jobs, rotation=45, ha='right', va='top', fontsize=12)
    ax.set_yticklabels(top_locations, fontsize=12)
    ax.set_xlabel('Job Title', fontsize=14, labelpad=20)
    ax.set_ylabel('Location', fontsize=14, labelpad=20)
    ax.set_zlabel('Number of Jobs', fontsize=14, labelpad=10)
    ax.set_title('3D Stacked Bar Chart of Job Counts by Title, Location, and Schedule Type', fontsize=16)

    ax.view_init(elev=20, azim=45)

    handles = [Patch(color=color, label=label) for label, color in zip(job_counts.columns, colors)]
    plt.figlegend(handles=handles, loc='upper right', bbox_to_anchor=(1.3, 0.7))

    plt.tight_layout(rect=[0, 0, 0.9, 1])
    save_plot("3d_stacked_bar.png")
    plt.show()

def plot_radar_chart(data):
    """
    Plots a radar chart showing the average salary by the top 10 job titles.

    Parameters:
    - data: DataFrame containing job title and salary information

    Returns:
    None
    """
    # Calculate the average salary by job title
    job_salaries = data.groupby('job_title_short')['salary_year_avg'].mean().dropna()
    # Get the top 10 job titles by average salary
    top_10_jobs = job_salaries.nlargest(10)
    # Create a radar chart
    categories = list(top_10_jobs.index)
    values = list(top_10_jobs.values)
    num_vars = len(categories)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

    # The plot is a circle, so we need to "complete the loop" and append the start to the end
    values += values[:1]
    angles += angles[:1]
    fig, ax = plt.subplots(figsize = (15, 9), subplot_kw=dict(polar=True))

    ax.fill(angles, values, color='skyblue', alpha=0.25)
    ax.plot(angles, values, color='blue', linewidth=2)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=10)

    ax.set_rlabel_position(0)
    y_ticks = ax.get_yticks()
    ax.set_yticklabels([f"${y/1000:.0f}k" for y in y_ticks], size=10)

    for label, angle in zip(ax.get_xticklabels(), angles):
        if angle in (0, np.pi):
            label.set_horizontalalignment('center')
        elif 0 < angle < np.pi:
            label.set_horizontalalignment('left')
        else:
            label.set_horizontalalignment('right')

    for i, value in enumerate(values[:-1]):
        angle = angles[i]
        ax.text(angle, value, f'${value/1000:.0f}k', ha='center', va='center')
    plt.title('Average Salary by Top 10 Job Titles', size=20, color='navy', y=1.1)
    plt.tight_layout()
    save_plot("radar_chart.png")
    plt.show()
