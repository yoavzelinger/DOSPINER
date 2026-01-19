import pandas as pd

from DOSPINER.ModelMapping.MappedRandomForest import MappedRandomForest

from .ADiagnoser import *
from .SFLDT import SFLDT

class DistributedForestDiagnoser(ADiagnoser):
    """
    The distributed forest diagnoser.
    Using a base diagnoser and run it for each tree separately. At the end, aggregate the results.
    """
    mapped_model: MappedRandomForest

    def __init__(self, 
                 base_diagnoser_class: type[ADiagnoser],
                 mapped_model: MappedRandomForest,
                 X: pd.DataFrame,
                 y: pd.Series,
                 **kwargs: object
    ):
        super().__init__(mapped_model, X, y)
        self.base_diagnoser_class: type[ADiagnoser] = base_diagnoser_class
        self.base_diagnoser_parameters = kwargs

    def convert_indices_to_global(self,
                                  estimator_index: int,
                                  local_diagnosis: list[int]
    ) -> list[int]:
        """
        Convert local diagnosis indices to global indices.

        Parameters:
            local_diagnosis (list[int]): The local diagnosis indices.

        Returns:
            list[int]: The global diagnosis indices.
        """
        for index_in_diagnosis, component_local_index in enumerate(local_diagnosis):
            global_index = self.mapped_model.inverse_component_estimator_map[(estimator_index, component_local_index)]
            local_diagnosis[index_in_diagnosis] = global_index
        return local_diagnosis


    def get_diagnoses(self,
                      retrieve_ranks: bool = False,
     ) -> list[list[int]] | list[tuple[list[int], float]]:
        """
            Get the diagnosis.
            Achieved by running the base diagnoser on each tree separately and aggregate the results.
        """
        self.diagnoses: list[tuple[list[int], float]] = []
        for estimator_index, mapped_estimator in enumerate(self.mapped_model.mapped_estimators):
            estimator_diagnoses: list[tuple[list[int], float]]
            try:
                base_diagnoser = self.base_diagnoser_class(
                    mapped_estimator,
                    self.X_after,
                    self.y_after,
                    **self.base_diagnoser_parameters
                )
                estimator_diagnoses = base_diagnoser.get_diagnoses(retrieve_ranks=True)
            except SFLDT.UnaffectedModelError:
                # The specific estimator was not affected from the concept drift, skip it
                continue
            for diagnosis_index, (estimator_diagnosis, estimator_diagnosis_rank) in enumerate(estimator_diagnoses):
                if diagnosis_index == len(self.diagnoses):
                    self.diagnoses.append(([], 0.0))
                global_diagnosis = self.convert_indices_to_global(estimator_index, estimator_diagnosis)
                current_diagnosis, current_rank = self.diagnoses[diagnosis_index]
                current_diagnosis.extend(global_diagnosis)
                self.diagnoses[diagnosis_index] = (current_diagnosis, current_rank + estimator_diagnosis_rank)
            
        return super().get_diagnoses(retrieve_ranks)