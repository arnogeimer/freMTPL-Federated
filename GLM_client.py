import warnings
import flwr as fl
import sys
'''
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from sklearn.compose import ColumnTransformer'''
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_poisson_deviance
import auxiliary_functions as aux
# #############################################################################
# 1. Regular PyTorch pipeline: nn.Module, train, test, and DataLoader
# #############################################################################

warnings.filterwarnings("ignore", category=UserWarning)
clientnbr=int(sys.argv[1])

localupdates=aux.localupdates

### The Preprocessing

df_=aux.df[aux.df["Region"]==aux.Regions[clientnbr-1]]

class GLM():
    def __init__(self) -> None:
        self.model= Pipeline(
            [
                ("preprocessor", aux.linear_model_preprocessor),
                ("regressor", PoissonRegressor(alpha=1e-12,warm_start=True)),
            ]
        )

def train(glm, trainloader, epochs): 
    glm.model.named_steps['regressor'].max_iter = epochs
    features,targets,sample_weights=trainloader
    glm.model.fit(features, targets, regressor__sample_weight=sample_weights)          

def test(glm, testloader):
    features,targets,sample_weights=testloader
    #exposure = features["Exposure"].values
    #y_true = features["Frequency"].values
    y_pred=glm.model.predict(features)
    #mse = mean_squared_error(targets, y_pred, sample_weight=sample_weights)
    #mae = mean_absolute_error(targets, y_pred, sample_weight=sample_weights)
    mask = y_pred > 0
    if (~mask).any():
        n_masked, n_samples = (~mask).sum(), mask.shape[0]
        print(
            "WARNING: Estimator yields invalid, non-positive predictions "
            f" for {n_masked} samples out of {n_samples}. These predictions "
            "are ignored when computing the Poisson deviance."
        )
    mpd=mean_poisson_deviance(
        targets[mask],
        y_pred[mask],
        sample_weight=sample_weights[mask],
    )
    accuracy=0
    return mpd,accuracy

def load_data(dataframe):
    return (dataframe,dataframe["Frequency"],dataframe["Exposure"])

# #############################################################################
# 2. Federation of the pipeline with Flower
# #############################################################################

glm = GLM()
trainloader = load_data(df_)
testloader = load_data(aux.df)

# We have to train the model once to initialize model.coef_ and model.intercept_
train(glm, trainloader, 1)

# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    
    def get_parameters(self, config):
        return [glm.model.named_steps['regressor'].coef_, glm.model.named_steps['regressor'].intercept_]

    def set_parameters(self, parameters):
        coefficients,intercept=parameters
        glm.model.named_steps['regressor'].coef_=coefficients
        glm.model.named_steps['regressor'].intercept_=intercept

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(glm, trainloader, epochs=localupdates)
        return self.get_parameters(config),len(trainloader[0]), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(glm, testloader)
        return loss,len(testloader[0]),{"accuracy": accuracy}

print(f"Client {clientnbr} starting")

# Start Flower client
fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=FlowerClient())

print(f"Client {clientnbr} shutting down")

