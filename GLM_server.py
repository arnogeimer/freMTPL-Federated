import flwr as fl
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import auxiliary_functions as aux

num_rounds=aux.num_rounds

from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
)

from flwr.server.client_proxy import ClientProxy

class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:
            # Convert `Parameters` to `List[np.ndarray]`
            aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(aggregated_parameters)

            # Save aggregated_ndarrays
            print(f"Saving round {server_round} aggregated_ndarrays...")
            weights=aggregated_ndarrays[0]
            intercept=aggregated_ndarrays[1]
            np.save(f"./Data/round-weights{server_round}", weights)
            np.save(f"./Data/round-intercept{server_round}", intercept)

        return aggregated_parameters, aggregated_metrics

strategy = SaveModelStrategy(
    min_fit_clients=22, # 22 available
    min_evaluate_clients=22,
    min_available_clients=22,
)

hist = fl.server.start_server(config=fl.server.ServerConfig(num_rounds=num_rounds),strategy=strategy)

server_mpds=np.array([item[1] for item in hist.losses_distributed])
np.save("./Data/ServerLoss", server_mpds)
