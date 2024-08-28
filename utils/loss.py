from tensorflow.keras import backend as K

def arobust_loss(alpha=1.0, c=1.0, gamma=2.0):
    def loss(y_true, y_pred):
        # Calculate the difference between true and predicted values
        x = y_pred - y_true

        # Adaptive robust loss calculation
        term1 = K.abs(alpha - 2.0) / alpha
        term2 = K.pow(K.abs(x / c), gamma - y_true)
        term3 = term2 / K.abs(alpha - 2.0) + 1.0
        loss_value = term1 * (K.pow(term3, alpha / 2.0) - 1.0)
        
        return K.mean(loss_value)
    
    return loss