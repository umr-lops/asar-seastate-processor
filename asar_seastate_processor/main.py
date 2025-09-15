import numpy as np
import xarray as xr
import onnxruntime 
import argparse
import logging
import sys
import os

from asar_seastate_processor.processor import generate_l2_wave_product
from asar_seastate_processor.utils import load_config, get_output_path, format_l2


def setup_logging(verbose=False):
    """Configure logging with DEBUG level if verbose, INFO otherwise."""
    fmt = "%(asctime)s %(levelname)s %(filename)s(%(lineno)d) %(message)s"
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level, format=fmt, datefmt="%d/%m/%Y %H:%M:%S", force=True
    )
    

def parse_args():
    """Parse command line arguments for L2 WAVE product generation."""
    parser = argparse.ArgumentParser(description="Generate a L2 WAVE product from a L1B or L1C SAFE.")

    # Define arguments
    parser.add_argument("--input_path", required=True, help="l1b or l1c safe path or listing path (.txt file).")
    parser.add_argument("--save_directory", required=True, help="where to save output data.")
    parser.add_argument("--product_id", required=True, help="3 digits ID representing the processing options. Ex: E00.")
    
    # Other arguments
    parser.add_argument("--overwrite", action="store_true", default=False, help="overwrite the existing outputs")
    parser.add_argument("--verbose", action="store_true", default=False)
   
    args = parser.parse_args()
    return args


def apply_preprocessing(ds, preprocessing_config):
    """Apply preprocessing to dataset if configuration is provided."""
    if not preprocessing_config:
        return ds

    from functools import partial
    preprocess = partial(
        FUNCTION_MAP[preprocessing_config['function']], 
        **preprocessing_config['parameters']
    )
    return preprocess(ds)
    

def main():
    """
    Process L1B/L1C data to generate L2 WAVE product using ONNX model.
    """
    args = parse_args()
    setup_logging(args.verbose)
    
    output_path = get_output_path(args.save_directory, args.input_path, args.product_id)
    
    if os.path.exists(output_path) and not overwrite:
        logging.info(
            f"{output_path} already exists. Use --overwrite to overwrite."
        )
        return None

    logging.info("Loading configuration file...")
    config_path = os.path.join(os.path.dirname(__file__), 'config', f'{args.product_id.lower()}.yaml')
    config = load_config(config_path)

    logging.info("Loading model...")
    model_path = os.path.join(os.path.dirname(__file__), 'models', f'{config["model_name"]}.onnx')
    model = onnxruntime.InferenceSession(model_path)
    
    logging.info("Processing file...")
    asa_l1b = xr.open_dataset(args.input_path)
    asa_l1b = apply_preprocessing(asa_l1b, config.get('preprocessing'))

    asa_l2 = generate_l2_wave_product(
        asa_l1b.sel(pol='VV'),
        model, config['inputs'],
        config['outputs'],
        config['kept_variables']
    )
    asa_l2 = format_l2(asa_l2, config['attributes'])
    logging.info("Processing completed successfully.")
    
    logging.info("Saving L2 file...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    asa_l2.to_netcdf(output_path, engine="h5netcdf")
    logging.info("L2 file saved.")


FUNCTION_MAP = {
    'xr.Dataset.drop_sel': xr.Dataset.drop_sel,
}

if __name__ == "__main__":
    main()