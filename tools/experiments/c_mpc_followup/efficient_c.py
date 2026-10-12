"""Identical C allocation, computing only model nodes actually consumed."""
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'simple_controller_comparison'))
from controllers import AccelerationPID


class EfficientAccelerationPID(AccelerationPID):
    def terms(self,q,dq,helpers):
        k=min(self.N-1,max(0,int(self.config.get('simple_acc_preview_steps',0))))
        preview=self.ff and not self.config.get('research_hold_future_disturbance',False)
        return {i:helpers.compute_mpc_terms(q,dq,
            helpers.disturbance_prediction[i] if preview else helpers.disturbance,
            helpers.interval_disturbance_prediction[i] if preview else helpers.disturbance,True)
            for i in {0,k}}
