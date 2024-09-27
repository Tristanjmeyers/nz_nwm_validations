### Evaluating the accuracy of weather forecasts across a given event.

import xarray as xr
import numpy as np

# Function for collating several forecasts into single Dataset.

def collate_forecasts(dates, prefix):
    # Make list of forecasts
    forecasts = []
    for date in dates:
        filename = f"{prefix}{date}.grib"
        forecasts.append(xr.open_dataset(filename, engine='cfgrib'))
    
    # Merge into a single Dataset
    all_forecasts = xr.concat(forecasts, "time")
    
    return all_forecasts

# Function for calculating error of several forecasts over a fixed period, indexed by lead time.

def format_ref_data(ref_data, forecast_data, coarsen=6, window_length = 40):
    """Put ref_data into a format that will allow it to be subtracted from forecast_data in order to compute error.
    
    This is done by first coarsening the data by a factor of `coarsen` and then adding a `step` dimension of length `window_length`.
    
    The default coarsen=6 is for taking hourly reference data to data every 6 hours.
    The default window_length=40 is for 10-day forecasts evaluated every 6 hours."""
    
    # Coarsen reference data
    coarsetime_ref_data = ref_data.coarsen(time = coarsen, boundary = 'trim').construct(
                                            time = ('new_step', 'old_step')).sel(old_step=0)
    
    coarsetime_ref_data = coarsetime_ref_data.swap_dims({'new_step':'time'})
    
    assert (coarsetime_ref_data.time.values[1] - coarsetime_ref_data.time.values[0] ==
           forecast_data.time.values[1] - forecast_data.time.values[0], 
           "Wrong `coarsen` parameter. Look at the difference in time values between ref_data and forecast_data, if any. Default is 6.")
    
    # Remove the empty "step" co-ordinate. Then construct a Dataset from a rolling window and call the new dimension "step."
    # Finally, label the new dimension using the values of the forecast_data "step" dimension.
    ref_data_reformat = coarsetime_ref_data.reset_coords("step", drop=True).rolling(time=window_length
                                 ).construct("step").assign_coords({"step":forecast_data.step.values})
    # Unfortunately, .rolling().construct() indexes each rolling window by its last element, rather than by its first.
    # We therefore need to drop the first few timesteps (which will be full of 'nan's) and relabel the time.
    dates_to_drop = ref_data_reformat.time.values[:window_length-1]
    drop_initial = ref_data_reformat.drop_sel(time = dates_to_drop)
    
    timediff = drop_initial.time.values[1] - ref_data_reformat.time.values[0]
    reassign_time = drop_initial.assign_coords(time = drop_initial.time.values - timediff)
    
    return reassign_time

def mse_over_period(data, reference_data, variables, daterange):
    """Take output data from collate_forecasts. Calculate error relative to reference data for 
    the period given by daterange [=(start date, end date), inclusive]. Find the mean squared error 
    over the lead time.
    
    "variables" should be a list of the desired variables for which to calculate the error.
    
    Reference data should already be in the correct format, using format_ref_data or otherwise."""
    
    error = data[variables] - reference_data[variables]
    
    squared_error = error*error
    
    mse = squared_error.where(np.logical_and(error.time + error.step >np.datetime64(daterange[0]),
                                         error.time + error.step <np.datetime64(daterange[1]))
                         ).mean(dim = ['latitude', 'longitude', 'time'], skipna=True)
    
    return mse

def wind_vector_mse(data, reference_data, daterange):
    """Compute the MSE for the wind speed vector at 10m above the surface (contained in u10 
    and v10 variables of data and reference_data), over the period given by daterange [=(start date, end date), inclusive].
    
    Reference data should already be in the correct format, using format_ref_data or otherwise."""
    
    squared_error = (data.u10-reference_data.u10)**2 + (data.v10-reference_data.v10)**2
    
    
    mse = squared_error.where(np.logical_and(data.time + data.step >np.datetime64(daterange[0]),
                                         data.time + data.step <np.datetime64(daterange[1]))
                         ).mean(dim = ['latitude', 'longitude', 'time'], skipna=True)
    
    return mse

def wind_speed_mse(data, reference_data, daterange):
    """Compute the MSE for the wind speed (not velocity!) at 10m above the surface (contained in u10 
    and v10 variables of data and reference_data), over the period given by daterange [=(start date, end date), inclusive].
    
    Reference data should already be in the correct format, using format_ref_data or otherwise."""
    
    error = np.sqrt(data.u10**2 + data.v10**2) - np.sqrt(reference_data.u10**2 + reference_data.v10**2)
    
    squared_error = error*error
    
    mse = squared_error.where(np.logical_and(data.time + data.step >np.datetime64(daterange[0]),
                                         data.time + data.step <np.datetime64(daterange[1]))
                         ).mean(dim = ['latitude', 'longitude', 'time'], skipna=True)
    
    return mse
    
    

# def format_ref_data(ref_data, forecast_data, coarsen=6, window_length = 40):
#     """Put ref_data into a format that will allow it to be subtracted from forecast_data in order to compute error.
    
#     This is done by first coarsening the data by a factor of `coarsen` and then adding a `step` dimension of length `window_length`.
    
#     The default coarsen=6 is for taking hourly reference data to data every 6 hours.
#     The default window_length=40 is for 10-day forecasts evaluated every 6 hours."""
    
#     # Coarsen reference data
#     coarsetime_ref_data = ref_data.coarsen(time = coarsen, boundary = 'trim').construct(
#                                             time = ('new_step', 'old_step')).sel(old_step=0)
    
#     coarsetime_ref_data = coarsetime_ref_data.swap_dims({'new_step':'time'})
    
#     assert (coarsetime_ref_data.time.values[1] - coarsetime_ref_data.time.values[0] ==
#            forecast_data.time.values[1] - forecast_data.time.values[0], 
#            "Wrong `coarsen` parameter. Look at the difference in time values between ref_data and forecast_data, if any. Default is 6.")
    
#     # Remove the empty "step" co-ordinate. Then construct a Dataset from a rolling window and call the new dimension "step."
#     # Finally, label the new dimension using the values of the forecast_data "step" dimension.
#     ref_data_reformat = coarsetime_ref_data.reset_coords("step", drop=True).rolling(time=window_length
#                                  ).construct("step").assign_coords({"step":forecast_data.step.values})
#     # Unfortunately, .rolling().construct() indexes each rolling window by its last element, rather than by its first.
#     # We therefore need to drop the first few timesteps (which will be full of 'nan's) and relabel the time.
#     dates_to_drop = ref_data_reformat.time.values[:window_length-1]
#     drop_initial = ref_data_reformat.drop_sel(time = dates_to_drop)
    
#     timediff = drop_initial.time.values[1] - ref_data_reformat.time.values[0]
#     reassign_time = drop_initial.assign_coords(time = drop_initial.time.values - timediff)
    
#     return reassign_time

# def mse_over_period(data, reference_data, daterange):
#     """Take output data from collate_forecasts. Calculate error relative to reference data for 
#     the period given by daterange [=(start date, end date), inclusive]. Find the mean squared error 
#     over the lead time."""
    
#     error = data - format_ref_data(reference_data, data)
    
#     squared_error = error*error
    
#     mse = squared_error.where(np.logical_and(error.time + error.step >np.datetime64(daterange[0]),
#                                          error.time + error.step <np.datetime64(daterange[1]))
#                          ).mean(dim = ['latitude', 'longitude', 'time'], skipna=True)
    
#     return mse
    
    