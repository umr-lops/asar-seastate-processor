import xarray as xr
import numpy as np
from datetime import datetime


def generate_l2_wave_product(
    ds, model, model_inputs, model_outputs, kept_variables=[]
):
    """
    Generate a level-2 wave (L2 WAV) product from an input L1B/C dataset.
    
    Parameters:
    - ds (xarray.Dataset): Input L1B/C dataset.
    - model (onnxruntime.InferenceSession): ML model used for prediction.
    - model_inputs (list of str): List of variables inputted to the model.
    - model_outputs (list of str): List of variables predicted by the model.
    - kept_variables (list of str): List of variables from the input dataset that are kept in the final product. Defaults to an empty list.

    Returns:
    - l2_product (xarray.Dataset): Level-2 wave product.
    """
    # Pass dataset to another function if product acquisitions are on land only
    if ds.land_flag.all():
        config_path = 'l2_ref.nc'
        l2_ref = xr.open_dataset(config_path)
        ds = generate_product_on_land(ds, l2_ref, model_outputs, kept_variables)
        
    # Stack tiles if cwaves are in model inputs to allow predictions
    if 'cwave_params' in model_inputs:
        ds_input = ds.stack(k_phi=["k_gp", "phi_hf"]) 
    else:
        ds_input = ds 

    # Prepare input arrays and make predictions
    input_arrays = [ds_input[v] for v in model_inputs]
    predictions = xr.apply_ufunc(
        predict_variables,
        model,
        *input_arrays,
        input_core_dims=[[], *[a.dims for a in input_arrays]],
        output_core_dims=[['preds', 'time']],
        vectorize=False
    )

    # Format predictions accordingly to model outputs
    predictions = xr.Dataset({
        variable_label: predictions.sel(preds=i)
        for i, variable_label in enumerate(model_outputs)
    })

    # Finish formatting l2 product
    l2_product = xr.merge([ds[kept_variables], predictions])
    l2_product = l2_product.reset_coords(["line", "sample"], drop=True)

    return l2_product

    
def predict_variables(
    model, *input_arrays
):
    """
    Launch predictions using a neural model.

    Parameters:
    - model (onnxruntime.InferenceSession): ML model used for prediction.
    - input_arrays (array like): Arrays containing the input data for ML predictions.

    Returns:
    - res (tuple): Tuple containing predictions for each variable.
    """
    # Reshape input arrays to allow concatenation
    reshaped_vars = [
        data[:, np.newaxis] if data.ndim == 1 else data
        for data in input_arrays
    ]

    # Concatenate along the second axis (axis=1)
    X_stacked = np.concatenate(reshaped_vars, axis=1)

    # Prepare inputs for the model
    input_name = model.get_inputs()[0].name
    inputs = {input_name: X_stacked.astype(np.float32)}

    res = model.run(None, inputs)

    res = [r[:, i] for r in res for i in range(r.shape[-1])]
    return res


def generate_product_on_land(
    ds_land, l2_ref, model_outputs, kept_variables
):
    """
    Generate a level-2 product when the input dataset contains only data acquired on land.

    Parameters:
    - ds_land (xarray.Dataset): Input L1B/C dataset.
    - l2_ref (xarray.Dataset): Reference level-2 dataset.
    - model_outputs (list of str): List of variables predicted by the model.
    - kept_variables (list of str): List of variables from the input dataset that are kept in the final product.

    Returns:
    - xarray.Dataset: Level-2 wave product.
    """
    
    l2_land = xr.Dataset()

    # Manage variables to keep from land dataset and those to fill with nans
    for var in kept_variables:
        if (var in ds_land.variables):
            l2_land[var] = ds_land[var]

        elif (var in l2_ref.coords) and (var not in ds_land.coords):
            l2_land[var] = l2_ref[var]

        else:
            var_dims = ('time', ) + l2_ref[var].dims
            var_shape = tuple(
                l2_ref.sizes[dim] if dim!='time' else ds_land.sizes['time']
                for dim in var_dims
            )
            l2_land[var] = (var_dims, np.full(var_shape, np.nan))
            l2_land[var].attrs = l2_ref[var].attrs

    # Manage coordinates 
    coords_to_add = set(l2_ref.coords) - set(l2_land.coords)
    
    for coord in coords_to_add:
        l2_land[coord] = l2_ref[coord]

    l2_land.attrs = ds_land.attrs
    return l2_land

    
def generate_reference_l2(
    input_path, output_path, model, model_inputs, model_outputs, kept_variables
):
    """
    Generate and save an l2-like dataset to serve as the default structure for land-only products
    It is important to be consistent with the version that will be used for the processing.
    
    Parameters:
    - input_path (str): path to the input level 1b file.
    - output_path (str): path to save the resulting dataset.
    - model (onnxruntime.InferenceSession): ML model used for prediction.
    - model_inputs (list of str): List of variables inputted to the model.
    - model_outputs (list of str): List of variables predicted by the model.
    - kept_variables (list of str): List of variables from the input dataset that are kept in the final product.
    """
    # Load l1b dataset
    l1b = xr.open_dataset(input_path).sel(pol='VV')

    # Generate the l2 product 
    l2_ref = generate_l2_wave_product(l1b, model, model_inputs, model_outputs, kept_variables).isel(time=0)

    # Initialize the dataset with default values
    l2_ref = l2_ref.where(False)
    l2_ref['time'] = datetime(1970, 1, 1, 0, 0, 0)
    l2_ref['longitude'] = 0
    l2_ref['latitude'] = 0
    l2_ref.attrs = {}

    l2_ref.to_netcdf(output_path)