from Explainers.utils import Explainer, ExplanationResult

from DyGLib.utils.DataLoader import Data
from DyGLib.utils.utils import NeighborSampler
from DyGLib.models.modules import TGNN

from Config.config import CONFIG

from .tgnnexplainer.tgnnexplainer.xgraph.method.subgraphx_tg import SubgraphXTG
from .tgnnexplainer.tgnnexplainer.xgraph.evaluation.metrics_tg import EvaluatorMCTSTG

import numpy as np
import pandas as pd
import itertools

CONFIG = CONFIG()

class SubgraphXTExplainer(Explainer):
    def __init__(self, model:TGNN, neighbor_finder: NeighborSampler, data: Data):
        super().__init__(model, neighbor_finder, data)

        self.explainer = SubgraphXTG(
                model,
                neighbor_finder,
                CONFIG.model.model_name.lower(),
                "subgraphx_tg",
                CONFIG.data.dataset_name,
                2025,
                data.dataset if data.dataset is not None else pd.DataFrame(),
                "event",
                device=CONFIG.model.device,
                results_dir="Logs/TGNNExplainer/"+CONFIG.data.dataset_name,
                debug_mode=True,
                save_results=False,
                mcts_saved_dir="Logs/TGNNExplainer/"+CONFIG.data.dataset_name,
                load_results=False,
                rollout=CONFIG.tgnnExplainerConfig.num_rollouts,
                min_atoms=CONFIG.tgnnExplainerConfig.min_atoms,
                c_puct=5,
                pg_explainer_model=None,
                pg_positive=True,
                num_layers=CONFIG.model.num_layers,
                num_neighbors=CONFIG.model.num_neighbors,
            )

        self.evaluator = EvaluatorMCTSTG(
            CONFIG.model.model_name,
            explainer_name="subgraphx_tg",
            dataset_name=CONFIG.data.dataset_name,
            explainer=self.explainer,
            results_dir=CONFIG.data.folder,
            seed=2025,
            cpuct=5
        )


    def explain_instance(self, src, dst, timestamp, silent = False):
        mask = (self.data.src_node_ids == src) & (self.data.dst_node_ids == dst) & (self.data.node_interact_times == timestamp)
        edge_id = self.data.edge_ids[mask]

        if silent == True:
            self.explainer.debug_mode = False
            self.explainer.verbose = False
            self.evaluator.debug_mode = False

        explain_results = self.explainer(event_idxs=int(edge_id))

        coalitions = self.evaluator.evaluate(explain_results, edge_id)["coalition"]
        
        return coalitions
    
    def build_coalitions(self, explanation):

        coalitions = np.array(list(itertools.zip_longest(*explanation, fillvalue=0))).T
        base_events = np.unique(np.array(self.explainer.base_events)).reshape((1,-1))
        base_events = np.repeat(base_events, axis=0, repeats=coalitions.shape[0])
        coalitions = np.concat([base_events,coalitions], axis=1)
        
        return coalitions, None, None
        