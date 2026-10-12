"""Simulation-only grid conversion of a checksum-validated 6 ms template.

No new training data or future online states. For each new anchor, use the
nearest earlier stored anchor and overlap-average its piecewise-constant
6 ms intervals. Nodes use linear vector / SO(3) interpolation within that
same stored anchor's heading frame. This cannot recover sub-6 ms impacts.
"""
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from disturbance_predictor import FullTaskTemplatePredictor
from disturbance_template.full_task_protocol import FullTaskProtocol, FullTaskContinuousHeadingFrame


def interval_weights(start, width, source_dt=.006, count=9):
    edges = np.arange(count+1)*source_dt
    weights = np.maximum(0., np.minimum(start+width, edges[1:])-np.maximum(start, edges[:-1]))/width
    if not np.isclose(weights.sum(), 1., atol=1e-10):
        raise ValueError('resampled interval exceeds available source prediction')
    return weights


class ResampledTemplate(FullTaskTemplatePredictor):
    def __init__(self, **kwargs):
        dt, n = float(kwargs['control_dt']), int(kwargs['horizon'])
        if dt not in (.002, .004, .006) or n != 9:
            raise ValueError('research conversion supports 2/4/6 ms, nine intervals')
        super().__init__(**dict(kwargs, control_dt=.006, horizon=9))
        protocol = FullTaskProtocol(mpc_dt=dt, horizon=n)
        if dt != .006:
            nodes = {k:[] for k in self._nodes}; intervals = {k:[] for k in self._intervals}
            source_times = np.arange(10)*.006
            for t in protocol.headline_anchor_times:
                old = min(int(np.floor((t+1e-12)/.006)),len(self._anchor_task_time)-1)
                offset = max(0., t-self._anchor_task_time[old])
                query = offset+np.arange(n+1)*dt
                weights = np.array([interval_weights(offset+k*dt,dt) for k in range(n)])
                for key in ('acc','omega','alpha'):
                    nodes[key].append(np.stack([np.interp(query,source_times,self._nodes[key][old,:,j]) for j in range(3)],axis=-1))
                    intervals[key].append(weights@self._intervals[key][old])
                nodes['rotation'].append(Slerp(source_times,Rotation.from_matrix(self._nodes['rotation'][old]))(query).as_matrix())
                # C acceleration mode only uses measured node0 rotation, but
                # keep every interface rotation a valid SO(3) matrix.
                centers=np.arange(9)*.006+.003
                mid=np.clip(offset+(np.arange(n)+.5)*dt,centers[0],centers[-1])
                intervals['rotation'].append(Slerp(centers,Rotation.from_matrix(self._intervals['rotation'][old]))(mid).as_matrix())
            self._nodes={k:np.array(v) for k,v in nodes.items()}
            self._intervals={k:np.array(v) for k,v in intervals.items()}
        self._protocol=protocol;self._control_dt=dt;self._horizon=n
        self._anchor_task_time=protocol.headline_anchor_times
        self._heading=FullTaskContinuousHeadingFrame(protocol)
        self._template_validation=dict(self._template_validation,anchor_count=len(self._anchor_task_time),
            research_resampled=True,source_anchor_dt=.006,target_anchor_dt=dt)
        self.reset()

    def metadata(self):
        return dict(super().metadata(),anchor_mode='research_resampled_from_validated_6ms_asset',
            future_source='historical template only; no future online measurements',
            control_dt=self._control_dt, original_asset_unchanged=True,
            interval_definition='overlap average of existing 6ms means; no new high-frequency information',
            hardware_predictor_changed=False)
