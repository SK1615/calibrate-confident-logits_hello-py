import numpy as np

def solve(logits_val, y_val, logits_test):
    # Temperature Scaling Calibration
    
    # Softmax function
    def softmax(x, temperature=1.0):
        exp_x = np.exp((x - np.max(x, axis=-1, keepdims=True)) / temperature)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    # Negative log-likelihood for temperature optimization
    def nll_loss(temperature, logits, labels):
        probs = softmax(logits, temperature)
        return -np.mean(np.log(probs[np.arange(len(labels)), labels] + 1e-10))
    
    # Grid search for temperature (simple but effective)
    temperatures = np.linspace(0.1, 5.0, 50)
    best_temp = 1.0
    best_nll = float('inf')
    
    for temp in temperatures:
        curr_nll = nll_loss(temp, logits_val, y_val)
        if curr_nll < best_nll:
            best_nll = curr_nll
            best_temp = temp
    
    # Apply best temperature to test logits
    return softmax(logits_test, best_temp)