def calculate_batch_health(predictions: list, confidence_threshold: float = 0.5) -> dict:
    """
    Calculate health metrics for a batch prediction job.
    
    Args:
        predictions: list of prediction results, each a dict with 'status' and optionally 'confidence'
        confidence_threshold: threshold below which a prediction is considered low confidence
    
    Returns:
        dict with keys: 'success_rate', 'avg_confidence', 'low_confidence_rate'
        All values as percentages (0-100), rounded to 2 decimal places.
    """

    success_preds = 0.0
    error_preds = 0.0
    all_preds = 0.0 # succes + error

    sum_of_confidence_of_successful = 0.0
    how_many_successful_below_reate = 0.0

    for pred in predictions:
        if pred["status"] == "error":
            error_preds += 1
        else:
            success_preds += 1
            sum_of_confidence_of_successful += pred["confidence"]
            how_many_successful_below_reate += float(pred["confidence"] < confidence_threshold)
        all_preds += 1

    if all_preds == 0:
        return {}
    else:
        success_rate = round(100.0 * (success_preds / all_preds), 2)
    if success_preds == 0:
        avg_confidence = 0
        low_confidence_rate = 0
    else:
        avg_confidence = round(100.0 * sum_of_confidence_of_successful / success_preds, 2)
        low_confidence_rate = round(100. * how_many_successful_below_reate / success_preds, 2)

    return {
        "success_rate": success_rate,
        "avg_confidence": avg_confidence,
        "low_confidence_rate": low_confidence_rate
    }