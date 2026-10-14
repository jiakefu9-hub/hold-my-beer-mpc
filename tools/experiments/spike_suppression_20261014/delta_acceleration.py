"""Research-only acceleration-increment cost; no robot launcher imports this.

J_delta = weight * (||a[0]-a_previous||^2 + sum ||a[k]-a[k-1]||^2).
Physical acceleration units, no implicit dt normalization. Hard constraints and
the existing integrator stay unchanged. Hardware replay updates a_previous from
the accepted packet's modeled acceleration; simulation assumes immediate policy
acceptance and separately reports mapper fallback/latency limitations.
"""
import numpy as np
from arm_mpc import ArmMPCPolicy


class DeltaAccelerationMixin:
    delta_weight = 0.0

    def __init__(self, *args, **kwargs):
        self.delta_weight = float(self.delta_weight)
        if not np.isfinite(self.delta_weight) or self.delta_weight < 0:
            raise ValueError('invalid delta acceleration weight')
        super().__init__(*args, **kwargs)
        self.previous_acceleration = np.zeros(self.nu)

    def accept_acceleration(self, acceleration):
        value = np.asarray(acceleration, dtype=float)
        if value.shape != (self.nu,) or not np.isfinite(value).all():
            raise ValueError('invalid accepted acceleration')
        self.previous_acceleration = value.copy()

    def compute_action(self, observation, helpers=None):
        result = super().compute_action(observation, helpers)
        # In immediate simulation this is the command accepted by the executor.
        # The recorded-input hardware wrapper replaces it after packet acceptance.
        self.accept_acceleration(result[2])
        return result

    def reset(self):
        super().reset()
        self.previous_acceleration = np.zeros(self.nu)

    def _build_cost(self, terms):
        blocks, linear = super()._build_cost(terms)
        if self.delta_weight == 0:
            return blocks, linear
        for k in range(self.horizon):
            diagonal = self.nx + np.arange(self.nu)
            blocks[k][diagonal, diagonal] += 2*self.delta_weight*(2 if k < self.horizon-1 else 1)
        linear[self._cu(0):self._cu(0)+self.nu] -= 2*self.delta_weight*self.previous_acceleration
        return blocks, linear


class DeltaSparsePolicy(DeltaAccelerationMixin, ArmMPCPolicy):
    def _build_hessian_pattern(self):
        rows, cols = super()._build_hessian_pattern()
        if self.delta_weight == 0:
            return rows, cols
        extra_rows = np.concatenate([self._cu(k-1)+np.arange(self.nu) for k in range(1,self.horizon)])
        extra_cols = np.concatenate([self._cu(k)+np.arange(self.nu) for k in range(1,self.horizon)])
        rows, cols = np.r_[rows,extra_rows], np.r_[cols,extra_cols]
        self._delta_order = np.lexsort((rows,cols))
        self._delta_extra_count = len(extra_rows)
        return rows[self._delta_order].astype(np.int32), cols[self._delta_order].astype(np.int32)

    def _pack_upper_triangles(self, hessians):
        values = super()._pack_upper_triangles(hessians)
        if self.delta_weight == 0:
            return values
        return np.r_[values,np.full(self._delta_extra_count,-2*self.delta_weight)][self._delta_order]


def hardware_policy(base, weight):
    """Add the same cross-stage cost after exact input condensation."""
    class DeltaCondensedPolicy(DeltaAccelerationMixin, base):
        delta_weight = weight

        def condense(self, *args, **kwargs):
            pc,qc,lc,uc,offset,full_p = super().condense(*args, **kwargs)
            if self.delta_weight:
                for k in range(1,self.horizon):
                    i=(k-1)*self.nu+np.arange(self.nu);j=k*self.nu+np.arange(self.nu)
                    cross=-2*self.delta_weight*self.max_ddq**2
                    pc[i,j]+=cross;pc[j,i]+=cross
            return pc,qc,lc,uc,offset,full_p

        def _objective(self, solution, linear, full_p):
            value = super()._objective(solution,linear,full_p)
            if self.delta_weight:
                u=solution[self._input_rows].reshape(self.horizon,self.nu)
                value-=2*self.delta_weight*float(np.sum(u[1:]*u[:-1]))
            return value
    return DeltaCondensedPolicy


def algebra_check():
    """Independently compare sparse/condensed cost against direct differences."""
    from scipy import sparse
    from tools.g1_commissioning.hardware_mpc_solver import CondensedArmMPCPolicy
    rng=np.random.default_rng(14)
    class Positive(DeltaSparsePolicy):delta_weight=.004
    kwargs=dict(default_q=np.zeros(5),horizon=9,control_dt=.006)
    for cls in (Positive,hardware_policy(CondensedArmMPCPolicy,.004)):
        policy=cls(**kwargs);policy.previous_acceleration=rng.normal(size=5)
        terms=[]
        for _ in range(policy.horizon+1):
            terms.append({key:rng.normal(size=shape) for key,shape in policy._step_term_shapes().items()})
        original=(ArmMPCPolicy if cls is Positive else CondensedArmMPCPolicy)(**kwargs)
        h0,l0=original._build_cost(terms);h1,l1=policy._build_cost(terms)
        # Explicit independent construction of the full added quadratic form.
        n=policy.num_variables;D=np.zeros((policy.horizon*5,n))
        for k in range(policy.horizon):
            D[k*5:(k+1)*5,policy._cu(k):policy._cu(k)+5]=np.eye(5)
            if k:D[k*5:(k+1)*5,policy._cu(k-1):policy._cu(k-1)+5]=-np.eye(5)
        target=np.r_[policy.previous_acceleration,np.zeros((policy.horizon-1)*5)]
        if cls is Positive:
            def matrix(p,h):
                up=sparse.csc_matrix((p._pack_upper_triangles(h),(p._p_rows,p._p_cols)),shape=(n,n)).toarray()
                return up+up.T-np.diag(np.diag(up))
            np.testing.assert_allclose(matrix(policy,h1)-matrix(original,h0),2*.004*D.T@D,atol=1e-10)
            np.testing.assert_allclose(l1-l0,-2*.004*D.T@target,atol=1e-10)
        else:
            bounds=policy._l_template.copy();bounds[:10]=rng.normal(size=10)
            p1,q1,*_=policy.condense(None,l1,bounds,policy._u_template,cost_blocks=policy._cost_blocks)
            p0,q0,*_=original.condense(None,l0,bounds,original._u_template,cost_blocks=original._cost_blocks)
            np.testing.assert_allclose(p1-p0,2*.004*(D@policy._T).T@(D@policy._T),atol=1e-9)
            np.testing.assert_allclose(q1-q0,-2*.004*(D@policy._T).T@target,atol=1e-9)
    zero=DeltaSparsePolicy(**kwargs);normal=ArmMPCPolicy(**kwargs)
    hz,lz=zero._build_cost(terms);hn,ln=normal._build_cost(terms)
    np.testing.assert_array_equal(zero._pack_upper_triangles(hz),normal._pack_upper_triangles(hn))
    np.testing.assert_array_equal(lz,ln)
    return dict(passed=True,checks=['sparse increment quadratic','condensed increment quadratic','zero cost exact parity'])
