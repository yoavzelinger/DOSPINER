from abc import ABC, abstractmethod
from collections import defaultdict
import pandas as pd
from copy import deepcopy

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from DOSPINER.ModelMapping.ATreeBasedMappedModel import ATreeBasedMappedModel
from DOSPINER.ModelMapping.TreeNodeComponent import TreeNodeComponent
from DOSPINER.ModelMapping.MappedDecisionTree import MappedDecisionTree
from DOSPINER.ModelMapping.MappedRandomForest import MappedRandomForest

from DOSPINER.Diagnosers import *

from .AFixer import AFixer, ATreeFixer

class ForestFixerWrapper(AFixer):
    alias = "forest_fixer_wrapper"

    original_mapped_model: MappedRandomForest

    def __init__(self, 
                 base_fixer_class: type[ATreeFixer],
                 mapped_model: MappedRandomForest,
                 X: pd.DataFrame,
                 y: pd.Series,
                 faulty_nodes_indices: list[int],
                 X_prior: pd.DataFrame = None,
                 y_prior: pd.Series = None,
                 sklearn_model: RandomForestClassifier = None,
                 **kwargs: dict
    ):
        super().__init__(
            mapped_model,
            X,
            y,
            faulty_nodes_indices,
            X_prior,
            y_prior,
            sklearn_model
        )
        self.base_fixer_class: type[ATreeFixer] = base_fixer_class
        self.base_fixer_parameters = kwargs
        self.faulty_estimators: dict[int, list[int]] = self.get_faulty_estimators()

    def get_faulty_estimators(self) -> dict[int, list[int]]:
        """
        Get the faulty estimators and their faulty nodes.
        
        Returns:
            dict[int, list[int]]: A mapping from estimator index to list of faulty node indices in that estimator.
        """
        faulty_estimators: dict[int, list[int]] = defaultdict(list)
        for faulty_node_index in self.faulty_nodes_indices:
            estimator_index, sklearn_node_index = self.original_mapped_model.component_estimator_map[faulty_node_index]
            faulty_estimators[estimator_index].append(sklearn_node_index)
        return faulty_estimators
    
    def fix_model(self) -> RandomForestClassifier:
        """
        Fix the model.

        Returns:
            RandomForestClassifier: The fixed random forest.
        """
        self.fixed_model: RandomForestClassifier = deepcopy(self.sklearn_model)
        for faulty_estimator_index, faulty_estimator_faulty_nodes_indices in self.faulty_estimators.items():
            fixer: ATreeFixer = self.base_fixer_class(
                mapped_model=self.original_mapped_model.mapped_estimators[faulty_estimator_index],
                X=self.X,
                y=self.y,
                faulty_nodes_indices=faulty_estimator_faulty_nodes_indices,
                X_prior=self.X_prior,
                y_prior=self.y_prior,
                **self.base_fixer_parameters
            )
            self.fixed_model.estimators_[faulty_estimator_index] = fixer.fix_model()

        return self.fixed_model