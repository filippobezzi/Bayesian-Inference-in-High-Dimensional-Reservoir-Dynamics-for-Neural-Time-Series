import torch
from scipy.signal import savgol_filter
def recursive_inference(esn_var, valid_ds, start_at_step=50, total_steps=150, num_samples=1000, noise = True):
    """
    Function that does recursive inference using SVI
    start_at_step can be used to "guide" initially the recursive forecasting by starting with a direct forecast
    and then switching to recursive for the remaining "total_steps - start_at_step" steps
    """
    device = valid_ds.predictions.device
    #mu, std = valid_ds.mean.to(device), valid_ds.std.to(device)
    
    # get to the correct state
    with torch.no_grad():
        # start the inference at states[start_at_steps]
        esn_var.reservoir.states = valid_ds.states[start_at_step : start_at_step + 1].to(device)
    
    if total_steps is None:
        total_steps = valid_ds.__len__()

    # containers
    n_pred_steps = total_steps - start_at_step
    y_samples = torch.zeros(num_samples, n_pred_steps, valid_ds.predictions.shape[1], device=device)

    with torch.no_grad():
        for i in range(n_pred_steps):
            # original index in time series
            t = start_at_step + i
            
            # predict the next point
            y_abs_pred = esn_var.predict(test_states=esn_var.reservoir.states, num_samples=num_samples)
            y_next_mean = y_abs_pred.mean(dim=0, keepdim=True)
            
            # dimension checks:
            if y_abs_pred.dim() == 3:
                # [Num_Samples, 1, Num_Regions] -> [Num_Samples, Num_Regions]
                y_next_samples = y_abs_pred.squeeze(1)
            elif y_abs_pred.dim() == 1:
                # [Num_Samples] -> [Num_Samples, 1]
                y_next_samples = y_abs_pred.unsqueeze(1)
            else:
                # [Num_Samples, Num_Regions]
                y_next_samples = y_abs_pred

            # save prediction
            y_samples[:, i, :] = y_next_samples

            #feedback_scaled = (y_next_mean - mu) / (std + 1e-8)
            esn_var.reservoir.forward(y_next_mean)
            
            if i % 10 == 0:
                print(f"Step {t}")
                
        # quantiles and final mean
        y_mean = y_samples.mean(dim=0)
        
        if noise:
            # SAVITZKY-GOLAY
            y_mean_filtered_np = savgol_filter(y_mean, window_length=21, polyorder=3)
            y_mean_filtered = torch.from_numpy(y_mean_filtered_np).to(device)
        else:
            y_mean_filtered = y_mean
            
        
        q_levels = torch.tensor([0.025, 0.975], dtype=y_samples.dtype, device=y_samples.device)
        q_vals = torch.quantile(y_samples, q_levels, dim=0)
        
        y_lower = q_vals[0]
        y_upper = q_vals[1]
                
    return y_mean_filtered, y_upper, y_lower, y_samples