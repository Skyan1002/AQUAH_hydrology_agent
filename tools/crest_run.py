import os
from datetime import datetime
def generate_control_file(
    time_begin,
    time_end,
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
TIMESTEP=1h
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
    if time_begin < format_change_date:
        # Before October 15, 2020: GaugeCorr format
        mrms_file_name = "GaugeCorr_QPE_01H_00.00_YYYYMMDD-HH0000.tif"
    else:
        # October 15, 2020 and after: MultiSensor format
        mrms_file_name = "MultiSensor_QPE_01H_Pass2_00.00_YYYYMMDD-HH0000.tif"
    
    control_content = f"""[Basic]
DEM={basic_data_path}/dem_clip.tif
DDM={basic_data_path}/fdir_clip.tif
FAM={basic_data_path}/facc_clip.tif

PROJ=geographic
ESRIDDM=true
SelfFAM=true

[PrecipForcing MRMS]
TYPE=TIF
UNIT=mm/h
FREQ=1h
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
    time_end=time_end.strftime('%Y%m%d%H%M')
)}

[Execute]
TASK=Simu
"""

    # Write the content to the control file
    with open(control_file_path, 'w') as f:
        f.write(control_content)
    
    # Return the absolute path of the control file
    return os.path.abspath(control_file_path)

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

