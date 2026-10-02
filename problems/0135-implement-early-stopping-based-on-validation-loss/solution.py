from typing import Tuple

def early_stopping(val_losses: list[float], patience: int, min_delta: float) -> Tuple[int, int]:
    # Your code here
    best_loss = val_losses[0] + 2 * min_delta
    best_when = -1
    stop_count = 0
    for i in range(len(val_losses)):
        loss = val_losses[i]
        if best_loss - loss > min_delta:
            best_loss = loss
            best_when = i
            stop_count = -1
        elif stop_count + 1 == patience:
            return (i, best_when)

        stop_count += 1

    return (len(val_losses) - 1, best_when)            
