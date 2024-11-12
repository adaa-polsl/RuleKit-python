from typing import Optional

import numpy as np
from jpype import JObject


class KaplanMeierEstimator:
    """Kaplan-Meier estimator of survival function.
    """

    def __init__(self, java_object: JObject) -> None:
        """
        Args:
            java_object (JObject): \
            `adaa.analytics.rules.logic.representation.KaplanMeierEstimator` \
            object instance from Java
        """
        self._java_object: JObject = java_object
        self._times: np.ndarray = np.array([
            float(t) for t in self._java_object.getTimes()
        ])
        self._probabilities: Optional[np.ndarray] = None
        self._events_count: Optional[np.ndarray] = None
        self._censored_count: Optional[np.ndarray] = None
        self._at_risk_count: Optional[np.ndarray] = None

    @property
    def times(self) -> np.ndarray:
        """
        Returns:
            np.ndarray: time points of the Kaplan-Meier estimator
        """
        return self._times

    @property
    def probabilities(self) -> np.ndarray:
        """
        Returns:
            np.ndarray: survival probabilities for each time point
        """
        if self._probabilities is None:
            self._probabilities = np.array([
                self.get_probability_at(t) for t in self._times
            ])
        return self._probabilities

    @property
    def events_count(self) -> np.ndarray:
        """
        Returns:
            np.ndarray: number of events for each time point
        """
        if self._events_count is None:
            self._events_count = np.array([
                self.get_events_count_at(t) for t in self._times
            ])
        return self._events_count

    @property
    def at_risk_count(self) -> np.ndarray:
        """
        Returns:
            np.ndarray: risks for each time point
        """
        if self._at_risk_count is None:
            self._at_risk_count = np.array([
                self.get_risk_set_count_at(t) for t in self._times
            ])
        return self._at_risk_count

    def get_probability_at(self, time: float) -> float:
        """Gets survival probability at given time point.

        Args:
            time (float): time point

        Returns:
            float: survival probability at given time point
        """
        return float(self._java_object.getProbabilityAt(time))

    def get_events_count_at(self, time: float) -> int:
        """Gets number of events at given time point.

        Args:
            time (float): time point

        Returns:
            int: number of events at given time point
        """
        return int(self._java_object.getEventsCountAt(time))

    def get_risk_set_count_at(self, time: float) -> int:
        """Gets risk at given time.

        Args:
            time (float): time point

        Returns:
            int: risk at given time
        """
        return int(self._java_object.getRiskSetCountAt(time))
