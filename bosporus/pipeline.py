import numpy as np
import pandas as pd
from .fit import ConstantFit, MichaelisMentenFit, PiecewiseLinearFit, ExponentialSaturationFit
from .evaluate_fit import relative_likelihood, calculate_AIC_weight_entropy
from .graph_construction import construct_graph
from .centrality_measures import compute_centrality_measures

class Flow():
    def __init__(self, scores, distances):
        self._scores = scores
        self._distances = distances
        
        self._measures = list(scores.columns)
        self.observations = pd.DataFrame(index=range(len(scores)))
        self.best_fit = dict()

    @classmethod
    def from_coords(cls, coordinates, distance_fn, measures, graph_type, distance_kwargs=None, graph_kwargs=None):
        """Path 1: full pipeline — coords + distance fn + graph construction"""
        distances = distance_fn(coordinates, **(distance_kwargs or None))
        edge_list = construct_graph(coordinates, graph_type, **(graph_kwargs or None))
        scores = compute_centrality_measures(edge_list, N=len(coordinates), measures=measures)
        obj = cls(scores=scores, distances=distances)
        return obj

    @classmethod
    def from_coords_and_edgelist(cls, coordinates, distance_fn, measures, edge_list, distance_kwargs=None):
        """Path 2: coords + distance fn + pre-built edge list"""
        distances = distance_fn(coordinates, **(distance_kwargs or None))
        scores = compute_centrality_measures(edge_list, N=len(coordinates), measures=measures)
        obj = cls(scores=scores, distances=distances)
        return obj

    @classmethod
    def from_coords_and_scores(cls, coordinates, distance_fn, scores: pd.DataFrame, distance_kwargs=None):
        """Path 3: coords + distance fn + pre-computed scores"""
        distances = distance_fn(coordinates, **(distance_kwargs or None))
        obj = cls(scores=scores, distances=distances)
        return obj

    @classmethod
    def from_distances_and_scores(cls, distances: pd.Series, scores: pd.DataFrame):
        """Path 4: no coords — pre-computed distances and scores only"""
        obj = cls(scores=scores, distances=distances)
        return obj
    
    def flow(self, distance_key, measures, fits, calculate_rel_ll_to_baseline):
        self.fit_models(measures=measures, distance_key=distance_key, fits=fits, calculate_rel_ll_to_baseline=calculate_rel_ll_to_baseline)
    
    @staticmethod
    def _set_entropy_weights(fit_instances, baseline_fit):
        rel_ll = [fit_instance.scaled_relative_loglikelihood_over_baseline for fit_instance in fit_instances if fit_instance != baseline_fit]
        entropy = calculate_AIC_weight_entropy(np.array(rel_ll))
        
        i = 0
        for fit in fit_instances:
            if fit == baseline_fit:
                continue
            else:
                fit.entropy_AIC_weights = entropy
                i += 1
        
        
    def fit_models(
        self,
        measures,
        distance_key,
        fits=None, 
        calculate_rel_ll_to_baseline=None
    ):
        
        if fits is None:
            fits = [ConstantFit, PiecewiseLinearFit, ExponentialSaturationFit, MichaelisMentenFit]
        if calculate_rel_ll_to_baseline is None:
            calculate_rel_ll_to_baseline = ConstantFit
        if calculate_rel_ll_to_baseline not in fits:
            raise ValueError("baseline fit class must be included in fits")
            
        self.best_fits = dict()
        fit_quality_data = []

        d = self.observations[distance_key].values

        for measure in measures:
            S = self.observations[measure].values
            
            baseline_fit = calculate_rel_ll_to_baseline(S, d)
            baseline_fit.fit()
            baseline_aic = baseline_fit.AIC
            
            fit_instances = []
            for fit_class in fits:
                if fit_class == calculate_rel_ll_to_baseline:
                    continue
                fit_instance = fit_class(S, d)
                fit_instance.fit()
                fit_instance.scaled_relative_loglikelihood_over_baseline = relative_likelihood(fit_instance.AIC, baseline_aic,  len(d))
                fit_instances.append(fit_instance)
                
            best_fit = min(fit_instances, key=lambda x: x.AIC)
            self._set_entropy_weights(fit_instances, baseline_fit)
            
            fit_quality_data[measure] = best_fit.params_summary()
            self.observations[f"BOSPORUS corrected {measure}"] = best_fit.S_corrected

        self.fit_quality = pd.DataFrame(fit_quality_data)