import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from .ATreeBasedMappedModel import ATreeBasedMappedModel
from .TreeNodeComponent import TreeNodeComponent
from .MappedDecisionTree import MappedDecisionTree

class MappedRandomForest(ATreeBasedMappedModel):
    mapped_estimators: list[MappedDecisionTree]
    component_estimator_map: dict[int, tuple[int, int]] # component index to (estimator index, sklearn index in estimator)

    def __init__(self, 
                 model: RandomForestClassifier,
                 feature_types: dict[str, str],
                 X: pd.DataFrame = None,
                 y: pd.Series = None
    ):
        """
        Initialize the MappedRandomForest.
        
        Parameters:
            model (RandomForestClassifier): The sklearn random forest model.
            X (DataFrame): The data.
            y (Series): The target column.
        """
        self.mapped_estimators: list[MappedDecisionTree] = [
            MappedDecisionTree(decision_tree_model=decision_tree,
                               feature_types=feature_types,
                               X=X,
                               y=y
                               )
            for decision_tree in model.estimators_
        ] # TODO: Check about filtering the data based on the bootstrap and the features used in the tree?
        
        super().__init__(model, X, y, feature_types)


    def map_model(self, 
    ) -> dict[int, 'MappedDecisionTree.DecisionTreeNode']:
        self.components_map = {}
        self.component_estimator_map = {}
        current_component_index = 0
        mapped_estimator: MappedDecisionTree
        component: TreeNodeComponent
        for estimator_index, mapped_estimator in enumerate(self.mapped_estimators):
            for component in mapped_estimator:
                self.components_map[current_component_index] = component
                self.component_estimator_map[current_component_index] = (estimator_index, component.get_index())
                component.component_index = current_component_index
                current_component_index += 1
                
    def get_model_representation(self) -> str:
        for i, tree in enumerate(self.mapped_estimators):
            tree_repr = f"Tree {i}:\n{tree.__repr__()}\n\n"
        return tree_repr
    
    def __len__(self) -> int:
        return sum(map(len, self.mapped_estimators))