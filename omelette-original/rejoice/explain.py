from captum.attr import IntegratedGradients


class GNNExplainer:
    def __init__(self, forward, feature_names: list[str]):
        self.forward = forward
        self.feature_names = feature_names

    def explain(self, data):
        data = data.to(device)
        ig = IntegratedGradients(self.forward)
        mask = ig.attribute(inputs=data.x,
                            additional_forward_args=(data,))
        edge_mask = np.abs(mask.cpu().detach().numpy())
        if edge_mask.max() > 0:  # avoid division by zero
            edge_mask = edge_mask / edge_mask.max()

        df = pd.DataFrame(data=edge_mask, columns=self.feature_names)
        return df

