import os
from datetime import datetime
def generate_control_file(
    time_begin,
    time_end,
    time_step,
    basic_data_path,
    mrms_path,
    pet_path,
    gauge_id,
    gauge_lon,
    gauge_lat,
    usgs_data_path,
    basin_area,
    output_dir,
    wm,
    b,
    im,
    ke,
    fc,
    iwu,
    under,
    leaki,
    th,
    isu,
    alpha,
    beta,
    alpha0,
    
    control_file_path='control.txt',
    grid_on=False,

):
    """
    Generate a control.txt file for CREST model with variable parameters.
    
    Args:
        time_begin (datetime): Simulation start time
        time_end (datetime): Simulation end time
        basic_data_path (str): Path to basic data directory containing DEM, flow direction and flow accumulation
        mrms_path (str): Path to MRMS precipitation data directory
        pet_path (str): Path to PET data directory
        gauge_id (str): USGS gauge ID
        gauge_lon (float): Gauge longitude
        gauge_lat (float): Gauge latitude
        usgs_data_path (str): Path to USGS data directory
        basin_area (float): Basin area in square kilometers
        output_dir (str): Output directory path for model results
        wm (float): CREST parameter - Maximum water capacity
        b (float): CREST parameter - Exponent of the variable infiltration curve
        im (float): CREST parameter - Impervious area ratio
        ke (float): CREST parameter - Potential evapotranspiration adjustment factor
        fc (float): CREST parameter - Soil saturated hydraulic conductivity
        iwu (float): CREST parameter - Initial soil water content
        under (float): KW parameter - Overland runoff velocity multiplier
        leaki (float): KW parameter - Interflow reservoir discharge rate
        th (float): KW parameter - Overland flow velocity exponent
        isu (float): KW parameter - Initial value of overland reservoir
        alpha (float): KW parameter - Multiplier in channel velocity equation
        beta (float): KW parameter - Exponent in channel velocity equation
        alpha0 (float): KW parameter - Base flow velocity
        control_file_path (str): Path to save the control file
        grid_on (bool): Whether to output grid files for streamflow
        
    Returns:
        str: Absolute path to the generated control file
    """
    # Convert all paths to absolute paths
    basic_data_path = os.path.abspath(basic_data_path)
    mrms_path = os.path.abspath(mrms_path)
    pet_path = os.path.abspath(pet_path)
    usgs_data_path = os.path.abspath(usgs_data_path)
    output_dir = os.path.abspath(output_dir)
    
    # Prepare the Task Simu section with optional output_grids parameter
    task_simu = """[Task Simu]
STYLE=SIMU
MODEL=CREST
ROUTING=KW
BASIN=0
PRECIP=MRMS
PET=PET
OUTPUT={output_dir}
PARAM_SET=CrestParam
ROUTING_PARAM_Set=KWParam
TIMESTEP={time_step}
"""
    
    # Add OUTPUT_GRIDS parameter if grid_on is True
    if grid_on:
        task_simu += "OUTPUT_GRIDS=STREAMFLOW\n"
    
    task_simu += """
TIME_BEGIN={time_begin}
TIME_END={time_end}
"""
    
    # Determine file format based on date
    format_change_date = datetime(2020, 10, 15)
    if time_step == '1h':
        unit_precip = 'mm/h'
        if time_begin < format_change_date:
            # Before October 15, 2020: GaugeCorr format
            mrms_file_name = "GaugeCorr_QPE_01H_00.00_YYYYMMDD-HH0000.tif"
        else:
            # October 15, 2020 and after: MultiSensor format
            mrms_file_name = "MultiSensor_QPE_01H_Pass2_00.00_YYYYMMDD-HH0000.tif"
    elif time_step == '1d':
        unit_precip = 'mm/d'
        if time_begin < format_change_date:
            # Before October 15, 2020: GaugeCorr format
            mrms_file_name = "GaugeCorr_QPE_24H_00.00_YYYYMMDD-000000.tif"
        else:
            # October 15, 2020 and after: MultiSensor format
            mrms_file_name = "MultiSensor_QPE_24H_Pass2_00.00_YYYYMMDD-000000.tif"
    
    control_content = f"""[Basic]
DEM={basic_data_path}/dem_clip.tif
DDM={basic_data_path}/fdir_clip.tif
FAM={basic_data_path}/facc_clip.tif

PROJ=geographic
ESRIDDM=true
SelfFAM=true

[PrecipForcing MRMS]
TYPE=TIF
UNIT={unit_precip}
FREQ={time_step}
LOC={mrms_path}
NAME={mrms_file_name}

[PETForcing PET]
TYPE=TIF
UNIT=mm/100d
FREQ=d
LOC={pet_path}
NAME=et{time_begin.strftime('%y')}MMDD.tif

[Gauge {gauge_id}] 
LON={gauge_lon}
LAT={gauge_lat}
OBS={usgs_data_path}/USGS_{gauge_id}_UTC_m3s.csv
OUTPUTTS=TRUE
BASINAREA={basin_area}
WANTCO=TRUE

[Basin 0]
GAUGE={gauge_id}

[CrestParamSet CrestParam]
gauge={gauge_id}
wm={wm}
b={b}
im={im}
ke={ke}
fc={fc}
iwu={iwu}

[kwparamset KWParam]
gauge={gauge_id}
under={under}
leaki={leaki}
th={th}
isu={isu}
alpha={alpha}
beta={beta}
alpha0={alpha0}

{task_simu.format(
    output_dir=output_dir,
    time_begin=time_begin.strftime('%Y%m%d%H%M'),
    time_end=time_end.strftime('%Y%m%d%H%M'),
    time_step=time_step
)}

[Execute]
TASK=Simu
"""

    # Write the content to the control file
    with open(control_file_path, 'w') as f:
        f.write(control_content)
    
    # Return the absolute path of the control file
    return os.path.abspath(control_file_path)

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
def visualize_model_results(ts_file, figure_path=None):
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
    fig, ax1 = plt.subplots(figsize=(12, 4))
    ax2 = ax1.twinx()

    # Plot precipitation as bar chart on right y-axis (inverted) - at bottom layer
    ax2.bar(range(len(df)), df['Precip(mm h^-1)'], color='blue', alpha=0.6, width=1.0, label='Precipitation', zorder=1)
    ax2.set_ylabel('Precipitation (mm/h)')
    ax2.invert_yaxis()  # Invert the y-axis so 0 is at top
    # Set y-axis limit so maximum precipitation value occupies 80% of the axis (inverted)
    max_precip = df['Precip(mm h^-1)'].max()
    ax2.set_ylim(max_precip / 0.8, 0)  # Set inverted y-axis limits

    # Plot discharge on left y-axis - on top layer
    # Fill area under simulated discharge curve with light blue
    ax1.fill_between(df['Time'], 0, df['Discharge(m^3 s^-1)'], color='skyblue', alpha=0.3, zorder=2)
    # Plot simulated discharge as black line
    ax1.plot(df['Time'], df['Discharge(m^3 s^-1)'], label='Simulated', linewidth=1, color='black', zorder=3)
    ax1.scatter(df['Time'], df['Observed(m^3 s^-1)'], label='Observed', s=5, color='red', zorder=4)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Discharge (m³/s)')
    # Set y-axis limit so maximum discharge value (observed and simulated) occupies 80% of the axis
    max_observed = df['Observed(m^3 s^-1)'].max()
    max_simulated = df['Discharge(m^3 s^-1)'].max()
    max_discharge = max(max_observed, max_simulated)
    ax1.set_ylim(0, max_discharge / 0.8)

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
    
    # Create a second figure with log scale y-axis (semilogy)
    fig2, ax3 = plt.subplots(figsize=(12, 4))
    ax4 = ax3.twinx()

    # Plot precipitation as bar chart on right y-axis (inverted) - at bottom layer
    ax4.bar(range(len(df)), df['Precip(mm h^-1)'], color='blue', alpha=0.6, width=1.0, label='Precipitation', zorder=1)
    ax4.set_ylabel('Precipitation (mm/h)')
    ax4.invert_yaxis()  # Invert the y-axis so 0 is at top
    # Set y-axis limit so maximum precipitation value occupies 80% of the axis (inverted)
    ax4.set_ylim(max_precip / 0.8, 0)  # Set inverted y-axis limits

    # Plot discharge on left y-axis with log scale - on top layer
    # Plot simulated discharge as black line
    ax3.plot(df['Time'], df['Discharge(m^3 s^-1)'], label='Simulated', linewidth=1, color='black', zorder=3)
    # Apply log scale to observed data and plot
    observed_log = df['Observed(m^3 s^-1)'].apply(lambda x: max(x, 0.001) if not np.isnan(x) else np.nan)
    ax3.scatter(df['Time'], observed_log, label='Observed', s=5, color='red', zorder=4)
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Discharge (m³/s) - Log Scale')
    ax3.set_yscale('log')  # Set y-axis to log scale

    # Add legends
    lines3, labels3 = ax3.get_legend_handles_labels()
    lines4, labels4 = ax4.get_legend_handles_labels()
    ax3.legend(lines3 + lines4, labels3 + labels4, loc='upper right')

    # Set title
    plt.title('Simulated vs Observed Discharge with Precipitation (Log Scale)')

    # Set x-axis limits to first and last time points
    ax3.set_xlim(df['Time'].iloc[0], df['Time'].iloc[-1])

    # Reduce x-axis density and rotate labels
    ax3.set_xticks(range(0, len(df), step))
    ax3.set_xticklabels([t.split()[0] for t in df['Time'][::step]], rotation=45, ha='right')

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save the log scale plot to figure_path as 'results_log.png' with dpi=300
    if figure_path is not None:
        save_path_log = os.path.join(figure_path, 'results_log.png')
        plt.savefig(save_path_log, dpi=300, bbox_inches='tight')
        print(f"Log scale plot saved to {os.path.abspath(save_path_log)}")
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


def crest_run(args,crest_args):
    # print("==== args ====")
    # for k, v in vars(args).items():
    #     print(f"{k}: {v}")
    # print("==== crest_args ====")
    # for k, v in vars(crest_args).items():
    #     print(f"{k}: {v}")

    if not os.path.exists(args.crest_output_path):
        os.makedirs(args.crest_output_path)
    generate_control_file(
        time_begin=args.time_start,
        time_end=args.time_end,
        time_step=args.time_step,
        basic_data_path=args.basic_data_clip_path,
        mrms_path=args.crest_input_mrms_path,
        pet_path=args.crest_input_pet_path,
        gauge_id=args.gauge_id,
        gauge_lon=args.longitude_gauge,
        gauge_lat=args.latitude_gauge,
        usgs_data_path=args.usgs_data_path,
        basin_area=args.basin_area,
        output_dir=args.crest_output_path,
        wm=crest_args.wm,
        b=crest_args.b,
        im=crest_args.im,
        ke=crest_args.ke,
        fc=crest_args.fc,
        iwu=crest_args.iwu,
        under=crest_args.under,
        leaki=crest_args.leaki,
        th=crest_args.th,
        isu=crest_args.isu,
        alpha=crest_args.alpha,
        beta=crest_args.beta,
        alpha0=crest_args.alpha0,
        grid_on=crest_args.grid_on,
        control_file_path=args.control_file_path,
    )
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

