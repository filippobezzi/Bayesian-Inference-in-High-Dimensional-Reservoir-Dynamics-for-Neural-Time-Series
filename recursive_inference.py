import torch

def recursive_inference(esn_var, valid_ds, steps=None, num_samples=1000):
    """
    Esegue l'inferenza ricorsiva prevedendo i VALORI ASSOLUTI (senza Delta e senza Gamma).
    
    Returns:
        y_mean: Tensor [steps, n_regions] -> Ottimo per i grafici.
        y_samples: Tensor [num_samples, steps, n_regions] -> Necessario per le metriche.
    """
    if steps is None:
        steps = valid_ds.predictions.shape[0]
    
    n_regions = valid_ds.predictions.shape[1]
    device = valid_ds.predictions.device
    
    esn_var.reservoir.reset_state(batch_size=1)
    
    with torch.no_grad():
        esn_var.reservoir.states = valid_ds.states[0:1].to(device)
        current_y_anchor = valid_ds.predictions[0:1].to(device)

    # Tensor per tutti i campioni (es. 1000 campioni per stabilizzare il rumore)
    y_samples = torch.zeros(num_samples, steps, n_regions, device=device)

    with torch.no_grad():
        for t in range(steps):
            # 1. Predizione Bayesiana DIRETTA del valore assoluto
            # In questo branch "vecchio", il modello sputa direttamente la scala fMRI
            y_abs_pred = esn_var.predict(test_states=esn_var.reservoir.states, num_samples=num_samples)
            
            # 2. Nessuna integrazione Leaky: prendiamo l'output così com'è
            # y_next_wave: [num_samples, n_regions]
            y_next_wave = y_abs_pred.squeeze(1)
            
            # 3. Salvataggio della distribuzione completa
            y_samples[:, t, :] = y_next_wave
            
            # 4. Feedback al Reservoir usando la MEDIA (niente ancore esterne)
            # Usiamo la media dei 1000 sample per limitare il rumore Monte Carlo
            current_y_anchor = y_next_wave.mean(dim=0, keepdim=True)
            esn_var.reservoir.forward(current_y_anchor)
            
            if t % 100 == 0:
                print(f"Step {t}/{steps}...")

    # Calcoliamo la media finale da restituire insieme ai campioni
    y_mean = y_samples.mean(dim=0) # [steps, n_regions]

    q = torch.tensor([0.025, 0.975], device=device)
    quantiles = torch.quantile(y_samples, q, dim=0)
    
    y_lower = quantiles[0] # [steps, n_regions]
    y_upper = quantiles[1] # [steps, n_regions]
                
    return y_mean, y_lower, y_upper, y_samples
                