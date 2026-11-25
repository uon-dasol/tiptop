"""Pick-and-carry path mathematics, without hardware or inference."""
import numpy as np
import pytest
import torch
from curobo.types.state import JointState

from tiptop.doosan.in_place import source_region, stationary_boundaries, wrist_travel_deg


def path():
    q=torch.zeros((4,6))
    q[2:,0]=torch.tensor([.01,.02])
    v=torch.zeros_like(q)
    v[0,0]=.003
    v[2,0]=.01
    return JointState(position=q,velocity=v)


def test_boundary_residue_is_removed_only_at_a_stationary_initial_hold():
    raw=path()
    clean=stationary_boundaries(raw)
    assert clean.velocity[0,0]==0 and raw.velocity[0,0]!=0
    assert torch.equal(clean.position,raw.position)
    assert torch.equal(clean.velocity[1:-1],raw.velocity[1:-1])
    raw.position[1,0]=.01
    with pytest.raises(ValueError,match='initial'):
        stationary_boundaries(raw)
    raw=path()
    raw.velocity[-1,0]=.1
    with pytest.raises(ValueError,match='final'):
        stationary_boundaries(raw)


def test_terminal_velocity_up_to_the_default_tolerance_is_zeroed_and_more_is_rejected():
    raw = path()
    raw.velocity[-1, 0] = .0147
    clean = stationary_boundaries(raw)
    assert torch.count_nonzero(clean.velocity[[0, -1]]) == 0
    assert torch.equal(clean.position, raw.position)
    raw.velocity[-1, 0] = .0151
    with pytest.raises(ValueError, match='final'):
        stationary_boundaries(raw)


def test_wrist_travel_counts_only_j4_and_j5_excursion_from_the_start():
    q=torch.zeros((5,6))
    q[:,5]=torch.linspace(0,np.deg2rad(190),5)  # J6 may turn freely
    assert wrist_travel_deg(JointState(position=q)) == 0
    q[2,3]=np.deg2rad(40)   # an excursion that returns still counts
    q[4,4]=-np.deg2rad(165)
    assert wrist_travel_deg(JointState(position=q)) == pytest.approx(165)


def test_source_region_needs_neither_box_nor_free_space_observations():
    region=source_region([[-.39,-1.25,-.05],[.39,1.25,-.02]],[[.1,.6,0],[.15,.65,.1]],
                         [[-.365,.47],[.365,1.225]])
    assert region['source']=='pick_source'
    assert region['support_z_m']==pytest.approx(0)
