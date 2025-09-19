import numpy as np
import xarray as xr
import onnxruntime 
import argparse
import logging
import sys
import os

from asar_seastate_processor.processor import generate_l2_wave_product
from asar_seastate_processor.utils import load_config, get_output_path, format_l2, apply_preprocessing, apply_range_filters, add_quality_indices, save_l2 


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
    parser.add_argument("--input_path", required=True, help="l1b/l1c path or listing (.txt file).")
    parser.add_argument("--save_directory", required=True, help="where to save output data.")
    parser.add_argument("--file_version", required=True, help="2 digits ID representing the file version. Ex: 01.")
    
    # Other arguments
    parser.add_argument("--overwrite", action="store_true", default=False, help="overwrite the existing outputs")
    parser.add_argument("--verbose", action="store_true", default=False)
   
    args = parser.parse_args()
    return args


def main():
    """
    Process L1B/L1C data to generate L2 WAVE product using ONNX model.
    Supports processing single files or multiple files listed in a .txt file.
    """
    args = parse_args()
    setup_logging(args.verbose)
    
    if args.input_path.endswith('.txt'):
        listing = np.loadtxt(args.input_path, dtype=str)
    else:
        listing = [args.input_path]
    
    logging.info("Loading configuration file...")
    config_path = os.path.join(os.path.dirname(__file__), 'config', f'fv{args.file_version}.yaml')
    config = load_config(config_path)
    
    logging.info("Loading model...")
    model_path = os.path.join(os.path.dirname(__file__), 'models', f'{config["model_name"]}.onnx')
    model = onnxruntime.InferenceSession(model_path)
    
    for path in listing:
        if not os.path.exists(path):
            logging.warning(f"File not found: {path}, skipping...")
            continue

        output_path = get_output_path(args.save_directory, path, args.file_version)
        
        if os.path.exists(output_path) and not args.overwrite:
            logging.info(
                f"{output_path} already exists. Use --overwrite to overwrite."
            )
            continue
        
        logging.info(f"Processing file...")
        try:
            asa_l1b = xr.open_dataset(path).sel(pol='VV')
            asa_l1b = apply_preprocessing(asa_l1b, config.get('preprocessing'))
            asa_l2 = generate_l2_wave_product(
                asa_l1b,
                model,
                config['inputs'],
                config['outputs'],
                config['kept_variables']
            )
            asa_l2 = apply_range_filters(asa_l1b, asa_l2, config.get('range_filters'))
            asa_l2 = format_l2(asa_l2, os.path.basename(path), config['attributes'])
            asa_l2 = add_quality_indices(asa_l2, config.get('quality_variables'))
            asa_l2 = asa_l2.reset_coords(["line", "sample"], drop=True)
            logging.info(f"Processing completed successfully for {path}")
            
            logging.info("Saving L2 file...")
            save_l2(asa_l2, output_path)
            logging.info(f"L2 file saved: {output_path}")
            
        except Exception as e:
            logging.error(f"Error processing {path}: {str(e)}")
            continue        


if __name__ == "__main__":
    main()