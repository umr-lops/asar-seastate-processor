import os
import yaml
from datetime import datetime


def load_config(config_path):
    """
    Load configuration from YAML file.
    
    Args:
        config_path (str): Path to configuration file to generate L2 product.

    Returns
        dict: Loaded configuration 
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_output_path(output_directory, path, product_id, date_directories=True):
    """
    Generates the output path for the processed file.

    Args:
        output_directory (str): The base directory where the output will be saved.
        path (str): The input file path.
        product_id (str): Output product version.
        date_directories (boolean): If True, add year and day of year in front of filename (YYYY/ddd). Defaults to True.

    Returns:
        str: Full savepath.
    """
    filename = path.split(os.sep)[-1]
    filename = "".join([filename[:-6], f"{product_id.upper()}.nc"]) 
    filename = filename.replace("_WVI_XSP", "_WVI_WAV")

    # Add year/day_of_year between the output_dir and the filename
    if date_directories:
        date_str = filename[18:26]
        date = datetime.strptime(date_str, "%Y%m%d")
        year, day_of_year = date.year, date.timetuple().tm_yday

        save_dir = os.path.join(output_directory, str(year), f"{day_of_year:03d}")
    else:
        save_dir = output_directory

    save_path = os.path.join(save_dir, filename)
    return save_path


def format_l2(ds, attributes):
    """
    Format and standardize Level-2 sea state dataset metadata and attributes.
    
    This function processes an xarray Dataset containing sea state measurements from ASAR/Envisat to conform to CF-1.11 and ACDD-1.3 conventions.
    
    Args:
        ds (xarray.Dataset): Level-2 ASAR WV dataset containing sea states parameters unformatted.
        attributes (dict): Dictionnary containing the attributes of predicted variables.
        
    Returns:
        xarray.Dataset: Formatted dataset with:
        - Standardized coordinate names ('lon', 'lat');
        - CF-compliant variable attributes (units, standard_name, long_name);
        - Complete global attributes following ACDD-1.3 conventions;
        - ESA CCI Sea State project metadata.
    """
    # Add attributes to the predicted variables
    for var in attributes.keys():
        ds[var].attrs = attributes[var]

    # Make longitude and latitude name compliant with the format
    ds = ds.rename({'longitude': 'lon', 'latitude': 'lat'}) # might not be needed anymore if xsarslc is updated
    
    ds.lon.attrs = {
        "units": "degrees_east",
        "long_name": "longitude",
        "standard_name": "longitude"
    }    	
    
    ds.lat.attrs = {
        "units": "degrees_north",
        "long_name": "latitude",
        "standard_name": "latitude"
    }

    ds.pol.attrs = {
        "long_name": "polarization",
        "standard_name": "polarization"
    }

    # Add global attributes to the dataset
    creation_date = datetime.today().strftime("%Y-%m-%dT%H:%M:%S:%f")
    
    global_attributes = {
        "title": "ESA CCI Sea State L2P from ASAR onboard Envisat wave mode (WV).",
        "id": "?",
        "summary": "This dataset contains estimates of significant wave height, windsea significant wave height and mean wave period data derived from Level 1 ASAR measurements.",
        "platform": "Envisat",
        "instrument": "ASAR",
        "band": "C",
        "spatial_resolution": "?",
        "creation_date": creation_date,
        "history": f"{creation_date} - Creation",
        "track_id": "?",
        "geospatial_lat_min": -80.0,
        "geospatial_lat_max": 80.0,
        "geospatial_lon_min": -180.0,
        "geospatial_lon_max": 180.0,
        "cycle": "?",
        "relative_pass_number": "?",
        "cdm_data_type": "trajectory",
        "featureType": "trajectory",
        "naming_authority": "cersat.ifremer.fr",
        "keywords": "Oceans > Ocean Waves > Significant Wave Height, Oceans > Ocean Waves > Sea State",
        "key_variables": "swh, windwave_swh, Tm0",
        "processing_level": "?",
        "comment": "These data were produced at Ifremer as part of the ESA SST CCI project",
        "platform_type": "low earth orbit satellite",
        "instrument_type": "synthetic aperture radar (sar)",
        "Conventions": "CF-1.11, ACDD-1.3, ISO 8601",
        "standard_name_vocabulary": "Climate and Forecast (CF) Standard Name Table v79",
        "Metadata_Conventions": "Climate and Forecast (CF) 1.7, Attribute Convention for Data Discovery (ACDD) 1.3",
        "keywords_vocabulary": "NASA Global Change Master Directory (GCMD) Science Keywords",
        "format_version": "Data Standards v2.1",
        "platform_vocabulary": "CEOS mission table",
        "instrument_vocabulary": "CEOS instrument table",
        "institution": "Institut Francais de Recherche pour l'Exploitation de la mer / Centre d'Exploitation et de Recherche Satellitaire, European Space Agency",
        "institution_abbreviation": "Ifremer / CERSAT, ESA",
        "project": "Climate Change Initiative - Sea State (CCI SeaState)",
        "program": "Climate Change Initiative - European Space Agency",
        "license": "ESA CCI Data Policy - free and open access",
        "acknowledgment": "Please acknowledge the use of these data with the following statement: these data were obtained from the ESA CCI Sea State project",
        "publisher_name": "Ifremer / CERSAT",
        "publisher_url": "http://cersat.ifremer.fr",
        "publisher_email": "cersat@ifremer.fr",
        "publisher_institution": "Ifremer",
        "publisher_type": "institution",
        "creator_name": "CERSAT",
        "creator_url": "http://cersat.ifremer.fr",
        "creator_email": "cersat@ifremer.fr",
        "creator_type": "institution",
        "creator_institution": "Ifremer",
        "contributor_name": "Jean-Francois Piolle",
        "contributor_role": "principal investigator",
        "references": "?",
        "contact": "jfpiolle@ifremer.fr",
        "technical_support_contact": "cersat@ifremer.fr",
        "scientific_support_contact": "jfpiolle@ifremer.fr",
        "processing_software": "IFREMER ASAR Level-2 seastate processor",
        "product_version": "1.0",
        "source": "?",
        "source_version": "?",
        "geospatial_bounds": "POLYGON ((-180.0 -80.0, 180.0 -80.0, 180.0 80.0, -180.0 80.0, -180.0 -80.0))",
        "geospatial_bounds_crs": "EPSG:4326",
        "geospatial_bounds_vertical_crs": "EPSG:5831",
        "geospatial_lat_units": "degrees_north",
        "geospatial_lon_units": "degrees_east",
        "geospatial_vertical_min": 0.0,
        "geospatial_vertical_max": 0.0,
        "time_coverage_start": ds.attrs['time_coverage_start'],
        "time_coverage_end": ds.attrs['time_coverage_end'],
    }

    ds.attrs = global_attributes
    
    return ds