from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def calculate_error(y_val, y_pred):
    """ Calculates the MAE and MSE for the provided target and prediction
        values.
        
        :param y_val:   Numpy array of target values
        :param y_pred:  Numpy array of prediction values

        :return:        dict of metrics keyed by metric names

    """

    metrics = {
        "mae": mean_absolute_error, 
        "mse": mean_squared_error,
        "r2_score": r2_score
        }

    res = {}

    for key, func in metrics.items():
        res = {**res, key: func(y_val, y_pred)}
    
    return res