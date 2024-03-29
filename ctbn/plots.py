from enum import Enum
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from numpy.linalg import norm
from plotly.subplots import make_subplots

from ctbn.ctbn_model import CTBN
from ctbn.learner import CTBNLearner
from ctbn.types import Trajectory


class PlotType(Enum):
    PARAMS = 1
    STRUCT = 2

class TrajectoryPlotter():

    def __init__(self,trajectory: 'Trajectory') -> None:
        self._trajectory = trajectory

    def plot_as_heatmap(self,names,time_grid: np.numarray)-> pd.DataFrame:
        df = pd.DataFrame(columns=names)

        i=0
        t_0_map = t_0_map
        for transition in self._trajectory._transitions:
            s0 = transition._s_init
            t_1_map = t_0_map + transition._exit_time
            t = time_grid[0]
            j = i
            for t in time_grid[j:]:
                if t>t_0_map and t<= t_1_map:
                    row=[t]+[int(s) for s in s0]
                    df.loc[len(df.index)] =row
                    i+=1
            j = i
            t_0_map = t_1_map
        return df




class LearningCurve():

    def __init__(self, reference: CTBN, target: CTBNLearner, type: PlotType):
        self._type = type
        self._reference = reference
        self._target = target
        self._mse = list()
        self._iteration = list()
        self._iteration_counter = 0

    def add_point(self):
        if self._type == PlotType.PARAMS:
            self.add_point_params()
        else:
            pass

    def add_point_params(self):
        mse = 0
        xi = 0
        for (n_ref, n_target) in zip(self._reference.nodes, self._target.nodes):
            for states in n_ref.all_state_combinations():
                cim_ref = n_ref.cims[states]
                cim_target = n_target.cims[states]
                mse += norm(cim_ref.im-cim_target.im, 'fro')
                xi += 1

        self._mse.append(mse/xi)
        self._iteration.append(self._iteration_counter)
        self._iteration_counter += 1


class LearningPlotter():

    def __init__(self):
        pass

    def plot(self, curve: LearningCurve):
        fig = px.line(x=curve._iteration, y=curve._mse,
                      labels={'x': 'Iteration', 'y': 'MSE'})
        fig.show()

    def plot(self, curves=List[LearningCurve]):

        x = curves[0]._iteration
        estimate = []
        upper = []
        lower = []
        for i in x:
            vals = [curve._mse[i] for curve in curves]
            estimate.append(np.median(vals))
            upper.append(np.percentile(vals, q=95))
            lower.append(np.percentile(vals, q=5))

        fig = go.Figure([
            go.Scatter(
                name='Median',
                x=x,
                y=estimate,
                mode='lines',
                line=dict(color='rgb(31, 119, 180)'),
            ),
            go.Scatter(
                name='Upper Bound',
                x=x,
                y=upper,
                mode='lines',
                marker=dict(color="#444"),
                line=dict(width=0),
                showlegend=False
            ),
            go.Scatter(
                name='Lower Bound',
                x=x,
                y=lower,
                marker=dict(color="#444"),
                line=dict(width=0),
                mode='lines',
                fillcolor='rgba(68, 68, 68, 0.3)',
                fill='tonexty',
                showlegend=False
            )
        ])

        fig.update_layout(
            yaxis_title='MSE',
            title='MSE over Iterations',
            hovermode="x"
        )
        fig.show()

    def plot(self, curves_list=Dict[str, List[LearningCurve]]):
        figs = []
        for name, curves in zip(curves_list.keys(), curves_list.values()):
            x = curves[0]._iteration
            estimate = []
            upper = []
            lower = []
            for i in x:
                vals = [curve._mse[i] for curve in curves]
                estimate.append(np.median(vals))
                upper.append(np.percentile(vals, q=95))
                lower.append(np.percentile(vals, q=5))

            figs.append(go.Scatter(
                name=name,
                x=x,
                y=estimate,
                mode='lines',
                line=dict(color='rgb(31, 119, 180)'),
            ))
            figs.append(go.Scatter(
                name='Upper Bound',
                x=x,
                y=upper,
                mode='lines',
                marker=dict(color="#444"),
                line=dict(width=0),
                showlegend=False
            ))
            figs.append(go.Scatter(
                name='Lower Bound',
                x=x,
                y=lower,
                marker=dict(color="#444"),
                line=dict(width=0),
                mode='lines',
                fillcolor='rgba(68, 68, 68, 0.3)',
                fill='tonexty',
                showlegend=False
            ))
        fig = make_subplots()

        for plots in figs:
            fig.add_trace(plots)
        fig.update_layout(
            yaxis_title='MSE',
            title='MSE over Iterations',
            hovermode="x"
        )
        fig.show()
 