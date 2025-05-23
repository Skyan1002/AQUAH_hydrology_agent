# Disable OpenTelemetry warnings and tracing
import os
os.environ["OTEL_PYTHON_DISABLED"] = "true"
os.environ["OTEL_PYTHON_TRACER_PROVIDER"] = "none"
import pypandoc

import yaml
from crewai import Agent, Task, Crew
import warnings
from opentelemetry import trace

# Disable the TracerProvider warning by setting the environment variable
os.environ["OTEL_PYTHON_TRACER_PROVIDER"] = "none"

## Agents
# Fixed parse simulation info
def fixed_parse_simulation_info(input_text: str, agents_config: dict, tasks_config: dict):
    # Create agents
    location_parser = Agent(
        role=agents_config['location_parser_agent']['role'],
        goal=agents_config['location_parser_agent']['goal'],
        backstory=agents_config['location_parser_agent']['backstory'],
        verbose=agents_config['location_parser_agent']['verbose']
    )

    time_parser = Agent(
        role=agents_config['time_parser_agent']['role'],
        goal=agents_config['time_parser_agent']['goal'],
        backstory=agents_config['time_parser_agent']['backstory'],
        verbose=agents_config['time_parser_agent']['verbose']
    )

    # Create tasks with context as a list
    parse_location_task = Task(
        description=tasks_config['parse_location']['description'].format(input_text=input_text),
        expected_output=tasks_config['parse_location']['expected_output'],
        agent=location_parser
    )

    parse_time_task = Task(
        description=tasks_config['parse_time_period']['description'].format(input_text=input_text),
        expected_output=tasks_config['parse_time_period']['expected_output'],
        agent=time_parser
    )

    # Create crew
    crew = Crew(
        agents=[location_parser, time_parser],
        tasks=[parse_location_task, parse_time_task],
        verbose=True
    )

    # Run the crew
    result = crew.kickoff()
    
    # Extract the basin name from the output of the location parsing task, removing any surrounding quotes
    basin_name = parse_location_task.output.raw.strip('"\'')
    # Extract the time period from the output of the time parsing task (as a string, e.g., "[datetime(...)]")
    time_period = parse_time_task.output.raw

    # Return both pieces of information as a dictionary
    return {
        "basin_name": basin_name,
        "time_period": time_period
    }

def get_basin_center_coords(basin_name, agents_config: dict, tasks_config: dict):
    basin_center_agent = Agent(
        role=agents_config['basin_center_agent']['role'],
        goal=agents_config['basin_center_agent']['goal'],
        backstory=agents_config['basin_center_agent']['backstory'],
        verbose=agents_config['basin_center_agent']['verbose']
    )
    basin_center_task = Task(
        description=tasks_config['get_basin_center']['description'].format(basin_name=basin_name),
        expected_output=tasks_config['get_basin_center']['expected_output'],
        agent=basin_center_agent
    )
    crew = Crew(
        agents=[basin_center_agent],
        tasks=[basin_center_task],
        verbose=True
    )
    crew.kickoff()
    # Extract the output and try to parse the tuple using regex
    import re
    output = basin_center_task.output.raw
    match = re.search(r"\(?\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\)?", output)
    if match:
        lat = float(match.group(1))
        lon = float(match.group(2))
        return (lat, lon)
    else:
        return output  # fallback: return raw output if parsing fails

# Find the outlet station representing the basin
import base64
import mimetypes
import os
import re
from openai import OpenAI
def file_to_dataurl(path: str) -> str:
    """
    Convert a local image file to a data URL (data:image/png;base64,...).
    """
    mime = mimetypes.guess_type(path)[0] or "image/png"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f"data:{mime};base64,{b64}"

def select_outlet_gauge_by_image(image_path: str, basic_data_image_path: str = None, facc_image_path: str = None, OPENAI_MODEL: str = 'gpt-4o') -> str:
    """
    Use OpenAI Vision model to select the outlet gauge from a basin map image.
    Returns the gauge ID as a string.
    
    Args:
        image_path: Path to the basin map with gauges
        basic_data_image_path: Optional path to basic data image with DEM, flow accumulation, etc.
        facc_image_path: Optional path to flow accumulation map with gauges
    """
    client = OpenAI()
    
    content = [
        {
            "type": "image_url",
            "image_url": {"url": file_to_dataurl(image_path)},
        },
    ]
    
    # Add basic data image if provided
    if basic_data_image_path and os.path.exists(basic_data_image_path):
        content.append({
            "type": "image_url",
            "image_url": {"url": file_to_dataurl(basic_data_image_path)},
        })

    # Add flow accumulation image if provided
    if facc_image_path and os.path.exists(facc_image_path):
        content.append({
            "type": "image_url", 
            "image_url": {"url": file_to_dataurl(facc_image_path)},
        })
        
    content.append({
        "type": "text",
        "text": "Select the outlet gauge strictly following the rules and output only its ID. Carefully identify reservoirs and lakes from the basic data image and avoid gauges downstream of reservoirs. The second image contains DEM, flow accumulation, and other basic information to help identify terrain features, main channels, and water bodies. The third image shows flow accumulation with gauges - prioritize selecting gauges with high flow accumulation values that are located downstream.",
    })

    messages = [
        {
            "role": "system",
            "content": (
                "You are a hydrologist who can interpret maps.\n"
                "A watershed boundary (yellow) with all USGS gauge stations (blue triangles) is shown. Also, a basic data image with DEM, flow accumulation, etc. is shown. And a flow accumulation image with gauges is shown, which is the most important image."
                "Select the single gauge that best represents the basin outlet following priority:\n"
                "1) It is furthest downstream / near the outlet.\n"
                "2) Controls the largest drainage area.\n"
                "3) CRITICAL: Identify reservoirs and lakes from the basic data image and DO NOT select gauges at or downstream of reservoir outlets as they significantly affect flow patterns.\n"
                "4) Prefer gauges located upstream of reservoirs when possible.\n"
                "5) IMPORTANT: Choose a gauge that is commonly used and has good data availability.\n"
                "6) Prioritize gauges with high flow accumulation values shown in the third image.\n"
                "Return ONLY the gauge ID number, no explanation."
            ),
        },
        {
            "role": "user",
            "content": content,
        },
    ]

    rsp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        # max_tokens=10,
        # temperature=0.1,
    )
    txt = rsp.choices[0].message.content
    # Extract a 6–10 digit number as the gauge ID
    match = re.search(r"\b(\d{6,10})\b", txt)
    return match.group(1) if match else txt.strip()



import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
def visualize_model_results(ts_file='../Output/ts.07325850.crest.csv', figure_path=None):
    """
    Visualize hydrological model results comparing simulated vs observed discharge.
    
    Parameters:
    -----------
    ts_file : str, optional
        Path to the time series CSV file with model results
    figure_path : str, optional
        Directory to save the plot image as 'results.png' (default: None, plot is not saved)
    """
    
    # Check if file exists
    if not os.path.exists(ts_file):
        print(f"Error: Results file not found at {ts_file}")
        return False
        
    # Read the CSV file
    try:
        df = pd.read_csv(ts_file)
    except Exception as e:
        print(f"Error reading results file: {str(e)}")
        return False
    
    print(f"Visualizing model results from: {os.path.abspath(ts_file)}")
    
    # Create the figure with two y-axes
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()

    # Plot discharge on left y-axis
    ax1.plot(df['Time'], df['Discharge(m^3 s^-1)'], label='Simulated', linewidth=1)
    ax1.plot(df['Time'], df['Observed(m^3 s^-1)'], label='Observed', linewidth=1)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Discharge (m³/s)')
    # Set y-axis limit to 2.1 times the maximum observed discharge value
    max_observed = df['Observed(m^3 s^-1)'].max()
    ax1.set_ylim(0, max_observed * 2.1)

    # Plot precipitation on right y-axis (inverted)
    ax2.plot(df['Time'], df['Precip(mm h^-1)'], color='blue', alpha=0.3, label='Precipitation')
    ax2.set_ylabel('Precipitation (mm/h)')
    ax2.invert_yaxis()  # Invert the y-axis so 0 is at top
    # Set y-axis limit to 2.1 times the maximum precipitation value (inverted)
    max_precip = df['Precip(mm h^-1)'].max()
    ax2.set_ylim(max_precip * 2.1, 0)  # Set inverted y-axis limits

    # Add legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    # Set title
    plt.title('Simulated vs Observed Discharge with Precipitation')

    # Set x-axis limits to first and last time points
    ax1.set_xlim(df['Time'].iloc[0], df['Time'].iloc[-1])

    # Reduce x-axis density and rotate labels
    step = 24  # Show every 24th tick
    ax1.set_xticks(range(0, len(df), step))
    ax1.set_xticklabels([t.split()[0] for t in df['Time'][::step]], rotation=45, ha='right')

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save the plot to figure_path as 'results.png' with dpi=300
    if figure_path is not None:
        save_path = os.path.join(figure_path, 'results.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {os.path.abspath(save_path)}")
    
    plt.show()

    
def evaluate_model_performance(ts_file='../Output/ts.07325850.crest.csv'):
    """
    Evaluate hydrological model performance by calculating statistical metrics
    between simulated and observed discharge.
    
    Parameters:
    -----------
    ts_file : str, optional
        Path to the time series CSV file with model results
        
    Returns:
    --------
    dict: Dictionary containing the calculated performance metrics
    """
    # Check if file exists
    if not os.path.exists(ts_file):
        print(f"Error: Results file not found at {ts_file}")
        return None
        
    # Read the CSV file
    try:
        df = pd.read_csv(ts_file)
    except Exception as e:
        print(f"Error reading results file: {str(e)}")
        return None
    
    print(f"Evaluating model performance from: {os.path.abspath(ts_file)}")
    
    # Extract simulated and observed discharge
    sim = df['Discharge(m^3 s^-1)'].values
    obs = df['Observed(m^3 s^-1)'].values
    
    # Remove any rows where either simulated or observed values are NaN
    valid_indices = ~(np.isnan(sim) | np.isnan(obs))
    sim = sim[valid_indices]
    obs = obs[valid_indices]
    
    if len(sim) == 0 or len(obs) == 0:
        print("Error: No valid data points after removing NaN values")
        return None
    
    # Calculate Root Mean Square Error (RMSE)
    rmse = np.sqrt(np.mean((sim - obs) ** 2))
    
    # Calculate Bias (as percentage)
    bias = np.mean(sim - obs)
    bias_percent = (bias / np.mean(obs)) * 100
    
    # Calculate Correlation Coefficient (CC)
    cc = np.corrcoef(sim, obs)[0, 1]
    
    # Calculate Nash-Sutcliffe Coefficient of Efficiency (NSCE)
    mean_obs = np.mean(obs)
    nsce = 1 - (np.sum((sim - obs) ** 2) / np.sum((obs - mean_obs) ** 2))
    
    # Create a dictionary with the metrics
    metrics = {
        'RMSE': rmse,
        'Bias': bias,
        'Bias_percent': bias_percent,
        'CC': cc,
        'NSCE': nsce
    }
    
    # Print the metrics
    print("\nModel Performance Metrics:")
    print(f"RMSE: {rmse:.4f} m³/s")
    print(f"Bias: {bias:.4f} m³/s ({bias_percent:.2f}%)")
    print(f"CC: {cc:.4f}")
    print(f"NSCE: {nsce:.4f}")
    
    return metrics




"""
Takes 3 images (basin_map_with_gauges.png, basic_data.png, results.png) as input,
calls OpenAI Vision model, and returns a structured analysis report (Markdown + JSON).
"""

def file_to_dataurl(path: str) -> str:
    mime = mimetypes.guess_type(path)[0] or "image/png"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f"data:{mime};base64,{b64}"


def make_report_for_figures(fig_dir: str, OPENAI_MODEL: str = 'gpt-4o') -> str:
    """Given a figure directory, reads three images and generates a report using LLM, returns a Markdown string."""
    imgs = {
        "basin":  file_to_dataurl(os.path.join(fig_dir, "basin_map_with_gauges.png")),
        "basic":  file_to_dataurl(os.path.join(fig_dir, "basic_data.png")), 
        "result": file_to_dataurl(os.path.join(fig_dir, "results.png")),
    }

    messages = [
        {
            "role": "system",
            "content": (
                "You are a senior hydrologist writing an internal technical memo.\n"
                "You will receive three figures in order:\n"
                "① basin_map_with_gauges → basin outline + all USGS gauges;\n"
                "② basic_data → DEM / FAM / DDM overview of the same basin;\n"
                "③ results → hydro-simulation result plot (obs, precip, sim curves).\n\n"
                "Generate a concise **Markdown report** with **three sections**:\n"
                "1. **Basin & Gauge Map**  – main features, which gauges appear downstream, notable geography;\n"
                "2. **Fundamental Basin Data** – comment on DEM (relief), FAM (flow acc.), DDM (drain-dens.); \n"
                "3. **Simulation vs Observation** – describe each curve, comment on fit, peaks, lags, bias.\n\n"

            ),
        },
        {
            "role": "user", 
            "content": [
                {"type": "image_url", "image_url": {"url": imgs["basin"]}},
                {"type": "image_url", "image_url": {"url": imgs["basic"]}},
                {"type": "image_url", "image_url": {"url": imgs["result"]}},
                {"type": "text", "text": "Please draft the report as requested."},
            ],
        },
    ]

    client = OpenAI()
    rsp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        temperature=0.3,
        max_tokens=5000,
    )
    return rsp.choices[0].message.content

def generate_simulation_summary(args, crest_args):
    """
    Generate a concise textual summary of the hydrological simulation based on args and crest_args.
    
    Args:
        args: Object containing simulation metadata and results
        crest_args: Object containing CREST model parameters
    
    Returns:
        str: A concise English summary of key simulation information
    """
    # Format dates for better readability
    start_date = args.time_start.split()[0] if isinstance(args.time_start, str) else args.time_start.strftime("%Y-%m-%d")
    end_date = args.time_end.split()[0] if isinstance(args.time_end, str) else args.time_end.strftime("%Y-%m-%d")
    
    # Extract key performance metrics
    rmse = args.metrics['RMSE']
    bias = args.metrics['Bias']
    bias_percent = args.metrics['Bias_percent']
    correlation = args.metrics['CC']
    nsce = args.metrics['NSCE']
    
    # Create summary text
    summary = f"""
Hydrological Simulation Summary for {args.basin_name}:

The simulation was conducted for the period from {start_date} to {end_date} in the {args.basin_name} (basin area: {args.basin_area} km²). The analysis focused on USGS gauge #{args.gauge_id} located at ({args.latitude_gauge}, {args.longitude_gauge}).

Key model parameters included:

Water Balnce Parameters:
- Water capacity ratio (WM: {crest_args.wm}): Maximum soil water capacity in mm. Higher values allow soil to hold more water, reducing runoff.
- Infiltration curve exponent (B: {crest_args.b}): Controls water partitioning to runoff. Higher values reduce infiltration, increasing runoff.
- Impervious area ratio (IM: {crest_args.im}): Represents urbanized areas. Higher values increase direct runoff.
- PET adjustment factor (KE: {crest_args.ke}): Affects potential evapotranspiration. Higher values increase PET, reducing runoff.
- Soil saturated hydraulic conductivity (FC: {crest_args.fc}): Rate at which water enters soil (mm/hr). Higher values allow easier water entry, reducing runoff.
- Initial soil water value (IWU: {crest_args.iwu}): Initial soil moisture (mm). Higher values leave less space for water, increasing runoff.

Kinematic Wave (Routing) Parameters:
- Drainage threshold (TH: {crest_args.th}): Defines river cells based on flow accumulation (km²). Higher values result in fewer channels.
- Interflow speed multiplier (UNDER: {crest_args.under}): Higher values accelerate subsurface flow.
- Interflow reservoir leakage coefficient (LEAKI: {crest_args.leaki}): Higher values increase interflow drainage rate.
- Initial interflow reservoir value (ISU: {crest_args.isu}): Initial subsurface water. Higher values may cause early peak flows.
- Channel flow multiplier (ALPHA: {crest_args.alpha}): In Q = αAᵝ equation. Higher values slow wave propagation in channels.
- Channel flow exponent (BETA: {crest_args.beta}): In Q = αAᵝ equation. Higher values slow wave propagation in channels.
- Overland flow multiplier (ALPHA0: {crest_args.alpha0}): Similar to ALPHA but for non-channel cells. Higher values slow overland flow.

Performance metrics show a Nash-Sutcliffe Coefficient of Efficiency (NSCE) of {nsce:.3f}, indicating {get_nsce_interpretation(nsce)}. The correlation coefficient between observed and simulated streamflow was {correlation:.3f}. The model showed a bias of {bias:.2f} m³/s ({bias_percent:.1f}%), with a root mean square error (RMSE) of {rmse:.2f} m³/s.
"""
    return summary.strip()

def get_nsce_interpretation(nsce):
    """Return a qualitative interpretation of the NSCE value"""
    if nsce > 0.75:
        return "very good performance"
    elif nsce > 0.65:
        return "good performance"
    elif nsce > 0.5:
        return "satisfactory performance"
    elif nsce > 0.0:
        return "poor performance"
    else:
        return "unsatisfactory performance"

def final_report_writer(args, crest_args, agents_config, tasks_config):
    report_writer = Agent(
        role=agents_config['report_writer_agent']['role'],
        goal=agents_config['report_writer_agent']['goal'],
        backstory=agents_config['report_writer_agent']['backstory'],
        allow_delegation=agents_config['report_writer_agent']['allow_delegation'],
        verbose=agents_config['report_writer_agent']['verbose']
    )
    report_writer_task = Task(
        description=tasks_config['write_report']['description'].format(
            summary=args.summary,
            report_for_figures_md=args.report_for_figures_md,
            basin_name=args.basin_name,
            figure_path=args.figure_path
        ),
        expected_output=tasks_config['write_report']['expected_output'],
        agent=report_writer
    )
    crew = Crew(
        agents=[report_writer],
        tasks=[report_writer_task]
    )
    final_report_md = crew.kickoff()
    with open("Hydro_Report.md", "w", encoding="utf-8") as f:
        report_content = str(final_report_md)
        if report_content.startswith("```markdown"):
            report_content = report_content[len("```markdown"):].strip()
        if report_content.endswith("```"):
            report_content = report_content[:-3].strip()
        f.write(report_content)
    

    extra = ['--pdf-engine=xelatex',
         '--variable', 'mainfont=Latin Modern Roman']
    output_pdf_path = f'Hydro_Report_{args.basin_name}.pdf'
    output = pypandoc.convert_file('Hydro_Report.md', 'pdf', outputfile=output_pdf_path,extra_args=extra)
    print(f"PDF report saved to {os.path.abspath(output_pdf_path)}")



# Main
def aquah_run(gpt_key: str):
    # Warning control
    warnings.filterwarnings('ignore')
    import logging
    logging.getLogger("opentelemetry.trace").setLevel(logging.ERROR)  # 把这个 logger 的阈值调高

    os.environ['OPENAI_MODEL_NAME'] = 'gpt-4o-mini'
    os.environ['OPENAI_API_KEY'] = gpt_key

    # Define file paths for YAML configurations
    files = {
        'agents': 'config/agents.yaml',
        'tasks': 'config/tasks.yaml'
    }
    # Load configurations from YAML files
    configs = {}
    for config_type, file_path in files.items():
        with open(file_path, 'r') as file:
            configs[config_type] = yaml.safe_load(file)

    # Assign loaded configurations to specific variables
    agents_config = configs['agents']
    tasks_config = configs['tasks']
    input_text = input("Please enter the simulation information (e.g., 'I want to simulate basin Fort Cobb, from 2022 June to July'): ")
    # input_text = 'San Antonio Rv at San Antonio, TX,  20230420 to 20230501'
    print('User input: ', input_text)
    print('\n\033[1;31m\033[1m------------------------------------------------')
    print('Step 1: Determine Location and Time Period')
    print('------------------------------------------------\033[0m\033[0m\n')
    try:
        result = fixed_parse_simulation_info(input_text, agents_config, tasks_config)
        # print(result)
    except Exception as e:
        print(f"Error: {e}")
        
    print('\n\033[1;31m\033[1m------------------------------------------------')
    print('Step 2: Find a Point within the Basin')
    print('------------------------------------------------\033[0m\033[0m\n')
    basin_name = result["basin_name"]
    center_coords = get_basin_center_coords(basin_name, agents_config, tasks_config)
    # print(f"Basin '{basin_name}' center coordinates: {center_coords}")

    # Construct the args
    from types import SimpleNamespace
    from datetime import datetime
    import re

    args = SimpleNamespace()
    args.basin_name = result["basin_name"]

    # Helper to parse a string like "[datetime(2022, 6, 1), datetime(2022, 7, 1)]"
    def parse_time_period_string(s):
        # Find all datetime(...) patterns
        dt_matches = re.findall(r"datetime\((.*?)\)", s)
        if len(dt_matches) != 2:
            raise ValueError("Could not parse two datetime objects from time_period string.")
        dt_objs = []
        for dt_str in dt_matches:
            # Split by comma, strip, and convert to int
            parts = [int(x.strip()) for x in dt_str.split(",")]
            dt_objs.append(datetime(*parts))
        return dt_objs

    time_period = result["time_period"]
    if isinstance(time_period, str):
        try:
            time_period = parse_time_period_string(time_period)
        except Exception as e:
            raise ValueError(f"Failed to parse time_period string: {e}")

    args.time_start = time_period[0]
    args.time_end = time_period[1]
    args.selected_point = center_coords
    # default args
    args.basin_shp_path = f'shpFile/Basin_selected.shp'
    args.basin_level = 5
    args.gauge_meta_path = 'EF5_tools/gauge_meta.csv'
    args.figure_path = 'figures'
    args.basic_data_path = 'BasicData'
    args.basic_data_clip_path = 'BasicData_Clip'
    args.usgs_data_path = 'USGS_gauge'
    args.mrms_data_path = 'MRMS_data'
    args.crest_input_mrms_path = 'CREST_input/MRMS/'
    args.num_processes = 4
    args.pet_data_path = 'PET_data'
    args.crest_input_pet_path = 'CREST_input/PET/'
    args.crest_output_path = 'CREST_output'
    args.control_file_path = 'control.txt'

    # Basin shp and basic data download
    import importlib
    basin_processor_module = importlib.import_module("tools.basin_processor")
    importlib.reload(basin_processor_module)
    basin_processor = basin_processor_module.basin_processor
    Basin_Area = basin_processor(args)
    args.basin_area = Basin_Area


    print('\n\033[1;31m\033[1m--------------------------------------------------------')
    print('Step 3: Find the Outlet Gauge Representing the Basin')
    print('--------------------------------------------------------\033[0m\033[0m\n')

    os.environ['OPENAI_MODEL_NAME'] = 'gpt-4o'
    OPENAI_MODEL = os.getenv("OPENAI_VISION_MODEL", "gpt-4o")

    basin_map_png = "figures/basin_map_with_gauges.png"   # Change to your image path
    basic_data_png = "figures/basic_data.png"  # Basic data image with DEM, flow accumulation, etc.
    facc_png = os.path.join(args.figure_path, "facc_with_gauges.png")  # Flow accumulation with gauges
    args.gauge_id = select_outlet_gauge_by_image(basin_map_png, basic_data_png, facc_png, OPENAI_MODEL)
    print("Selected Outlet Gauge ID:", args.gauge_id)


    import importlib
    import tools.gauge_processor
    importlib.reload(tools.gauge_processor)
    args.latitude_gauge, args.longitude_gauge = tools.gauge_processor.gauge_processor(args)

    print('\n\033[1;31m\033[1m--------------------------------------------------------')
    print('Step 4: Download Precipitation and Potential Evapotranspiration Data')
    print('--------------------------------------------------------\033[0m\033[0m\n')
    # Input data download
    data_download_flag = True
    if data_download_flag:
        import importlib
        import tools.precipitation_processor
        importlib.reload(tools.precipitation_processor)
        tools.precipitation_processor.precipitation_processor(args)

        import importlib
        import tools.pet_processor
        importlib.reload(tools.pet_processor)
        tools.pet_processor.pet_processor(args)

    print('\n\033[1;31m\033[1m--------------------------------------------------------')
    print('Step 5: Run CREST')
    print('--------------------------------------------------------\033[0m\033[0m\n')
    import types
    crest_args = types.SimpleNamespace(
        wm=100.0,
        b=1.0,
        im=0.01,
        ke=1.0,
        fc=1.0,
        iwu=25.00,
        under=1.0,
        leaki=0.1,
        th=50.0,
        isu=0.000000,
        alpha=2.0,
        beta=0.50,
        alpha0=1.0,
        grid_on=False,

    )
    import importlib
    import tools.crest_run
    importlib.reload(tools.crest_run)
    tools.crest_run.crest_run(args,crest_args)


    # import subprocess


    # # Construct the full path to ef5_64.exe (assumed to be in the current working directory)
    # ef5_exe_path = os.path.join(os.getcwd(), "ef5_64.exe")

    # # Check if the file exists
    # if not os.path.isfile(ef5_exe_path):
    #     print(f"{ef5_exe_path} not found. Please make sure ef5_64.exe is in the current folder.")
    # else:
    #     try:
    #         # Run ef5_64.exe and wait for it to finish
    #         result = subprocess.run([ef5_exe_path], check=True)
    #         print("ef5_64.exe ran successfully.")
    #     except subprocess.CalledProcessError as e:
    #         print(f"Error running ef5_64.exe, return code: {e.returncode}")
    #     except Exception as e:
    #         print(f"An exception occurred while running ef5_64.exe: {e}")
    # Run EF5 model
    import platform
    import subprocess

    if platform.system() == 'Windows':
        # Windows path
        ef5_exe_path = os.path.join(os.getcwd(), "ef5_64.exe")
        
        if not os.path.isfile(ef5_exe_path):
            print(f"{ef5_exe_path} not found. Please make sure ef5_64.exe is in the current folder.")
        else:
            try:
                result = subprocess.run([ef5_exe_path], check=True)
                print("ef5_64.exe ran successfully.")
            except subprocess.CalledProcessError as e:
                print(f"Error running ef5_64.exe, return code: {e.returncode}")
            except Exception as e:
                print(f"An exception occurred while running ef5_64.exe: {e}")
    else:
        # Linux path 
        ef5_path = "./EF5/bin/ef5"
        control_path = "control.txt"
        
        if not os.path.isfile(ef5_path):
            print(f"{ef5_path} not found. Please make sure EF5 binary exists.")
        else:
            try:
                result = subprocess.run([ef5_path, control_path], check=True)
                print("EF5 ran successfully.")
            except subprocess.CalledProcessError as e:
                print(f"Error running EF5, return code: {e.returncode}")
            except Exception as e:
                print(f"An exception occurred while running EF5: {e}")


    visualize_model_results(ts_file=f'./CREST_output/ts.{args.gauge_id}.crest.csv',figure_path=args.figure_path)
    args.metrics = evaluate_model_performance(ts_file=f'./CREST_output/ts.{args.gauge_id}.crest.csv')


    print('\n\033[1;31m\033[1m--------------------------------------------------------')
    print('Step 6: Generate a Report')
    print('--------------------------------------------------------\033[0m\033[0m\n')
    
    OPENAI_MODEL = os.getenv("OPENAI_VISION_MODEL", "gpt-4o")
    # analysis report for important figures
    args.report_for_figures_md = make_report_for_figures(args.figure_path, OPENAI_MODEL)
    # summary of the simulation, which can be used as prompt for the next step
    args.summary = generate_simulation_summary(args, crest_args)

    final_report_writer(args, crest_args, agents_config, tasks_config)








