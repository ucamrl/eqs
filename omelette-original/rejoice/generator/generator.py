import logging
import numpy as np
from .logical_graph import GeneratorSolution, LogicalMasterGraph
from .layer import MasterNetwork

default_params = {
    "G1_nodes": 3,
    "G1_K": 2,
    "G1_P": 0.1,
    "G2_nodes": 3,
    "G2_P": 0.2,
    "G3_nodes": 3,
    "G3_K": 2,
    "G3_P": 0.1,

    "ch1_ratio": 1,
    "ch2_ratio": 2,
    "ch3_ratio": 4,

    "stage1_ratio": 1.0,
    "stage2_ratio": 1.0,
    "stage3_ratio": 1.0,
    "image_size": 32,
    "n_param_limit": 4.0e6,
    "num_classes": 10,
}


class NAGOGenerator:
    """Search space of NAGO."""

    def __init__(self, **kwargs):
        """Construct the Hierarchical Neural Architecture Generator class.
        """
        logging.info("start init NAGO")
        kwargs = kwargs if bool(kwargs) else default_params
        print(kwargs)
        # to prevent invalid graphs with G_nodes <= G_k
        kwargs['G1_K'] = int(np.min([kwargs['G1_nodes'] - 1, kwargs['G1_K']]))
        kwargs['G3_K'] = int(np.min([kwargs['G3_nodes'] - 1, kwargs['G3_K']]))
        logging.info("NAGO desc: {}".format(kwargs))

        top_graph_params = ['WS', kwargs['G1_nodes'], kwargs['G1_P'], kwargs['G1_K']]
        mid_graph_params = ['ER', kwargs['G2_nodes'], kwargs['G2_P']]
        bottom_graph_params = ['WS', kwargs['G3_nodes'], kwargs['G3_P'], kwargs['G3_K']]

        channel_ratios = [kwargs['ch1_ratio'], kwargs['ch2_ratio'], kwargs['ch3_ratio']]
        stage_ratios = [kwargs['stage1_ratio'], kwargs['stage2_ratio'], kwargs['stage3_ratio']]

        conv_type = 'normal'
        top_merge_dist = [1.0, 0.0, 0.0]
        mid_merge_dist = [1.0, 0.0, 0.0]
        bottom_merge_dist = [1.0, 0.0, 0.0]
        op_dist = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        solution = GeneratorSolution(top_graph_params, mid_graph_params, bottom_graph_params,
                                     stage_ratios, channel_ratios, op_dist, conv_type,
                                     top_merge_dist, mid_merge_dist, bottom_merge_dist)

        # Generate an architecture from the generator
        model_frame = LogicalMasterGraph(solution)
        # Compute the channel multipler factor based on the parameter count limit
        n_params_base = model_frame._get_param_count()
        multiplier = int(np.sqrt(float(kwargs['n_param_limit']) / n_params_base))
        self.model = MasterNetwork(model_frame, multiplier, kwargs['image_size'],
                                   kwargs['num_classes'], None, False)

    def forward(self, x):
        """Calculate the output of the model.
        :param x: input tensor
        :return: output tensor of the model
        """
        y, aux_logits = self.model(x)
        return y