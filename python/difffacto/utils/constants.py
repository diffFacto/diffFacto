import enum 

class ModelMeanType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    PREVIOUS_X = 1  # the model predicts x_{t-1}
    START_X = 2  # the model predicts x_0
    EPSILON = 3  # the model predicts epsilon
    DRIFTED_EPSILON1=4
    DRIFTED_EPSILON2=5
    DRIFTED_EPSILON3=6
    DRIFTED_EPSILON4=7
    EPSILON_AND_ANCHOR=8
    SCALED_EPSILON=9
    DRIFTED_EPSILON5=10


class ModelVarType(enum.Enum):
    """
    What is used as the model's output variance.
    The LEARNED_RANGE option has been added to allow the model to predict
    values between FIXED_SMALL and FIXED_LARGE, making its job easier.
    """

    LEARNED = 1
    FIXED_SMALL = 2
    FIXED_LARGE = 3
    LEARNED_RANGE = 4
    
class LossType(enum.Enum):
    MSE = 1  # use raw MSE loss (and KL when learning variances)
    RESCALED_MSE = 2  # use raw MSE loss (with RESCALED_KL when learning variances)
    KL = 3  # use the variational lower-bound
    RESCALED_KL = 4  # like KL, but rescale to estimate the full VLB

    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALED_KL


model_mean_dict = {
    'x_0':ModelMeanType.START_X, 
    'prev_x':ModelMeanType.PREVIOUS_X, 
    'epsilon':ModelMeanType.EPSILON, 
    'drifted_epsilon1':ModelMeanType.DRIFTED_EPSILON1, 
    'drifted_epsilon2':ModelMeanType.DRIFTED_EPSILON2, 
    'drifted_epsilon4':ModelMeanType.DRIFTED_EPSILON4, 
    'drifted_epsilon3':ModelMeanType.DRIFTED_EPSILON3, 
    'drifted_epsilon5':ModelMeanType.DRIFTED_EPSILON5,
    'epsilon_and_anchor':ModelMeanType.EPSILON_AND_ANCHOR, 
    'scaled_epsilon':ModelMeanType.SCALED_EPSILON
    }
model_var_dict = {'learned':ModelVarType.LEARNED, 'learned_range':ModelVarType.LEARNED_RANGE, 'fixed_large':ModelVarType.FIXED_LARGE, 'fixed_small':ModelVarType.FIXED_SMALL}
model_loss_dict = {'kl':LossType.KL, 'rescaled_kl':LossType.RESCALED_KL, 'mse':LossType.MSE, 'rescaled_mse':LossType.RESCALED_MSE}

