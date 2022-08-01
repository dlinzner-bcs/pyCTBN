import numpy as np
import pandas as pd
from ctbn_model import CTBN, CTBNNode
from scipy import linalg

from ctbn.types import Trajectory


class InferenceProvider:
    def __init__(self, ctbn: 'CTBN', trajectory: Trajectory):
        self._ctbn = ctbn
        self._trajectory = trajectory

    def filter_node(self, node: 'CTBNNode', time_grid: np.numarray, p0=np.array([0.5, 0.5])) -> pd.DataFrame:
        df = pd.DataFrame(columns=['time', 'p(state)'])
        dt = time_grid[1]-time_grid[0]
        time = 0
        p = p0
        for trans in self._trajectory._transitions:
            tau = trans._exit_time
            self._ctbn.state(trans._s_init)
            p = p@linalg.expm(tau*node.cim.im)

            t = time
            for t in time_grid:
                if t > time and t <= time+tau:
                    p = p@linalg.expm(dt*node.cim.im)
                    df.loc[len(df.index)] = [t, p]
            time += tau

        return df

    def resample_node(self, node: 'CTBNNode', p0=np.array([0.5, 0.5])) -> 'Trajectory':
        pass
