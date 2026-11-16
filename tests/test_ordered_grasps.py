"""One particle per supplied grasp, in the supplied order (cuTAMP patch used by the known-object adapter)."""

import torch

from cutamp.config import TAMPConfiguration
from cutamp.particle_initialization import ordered_grasp_indices


def test_each_grasp_gets_one_particle_in_the_supplied_order():
    assert ordered_grasp_indices(5, 5, "cpu").tolist() == [0, 1, 2, 3, 4]


def test_more_grasps_than_particles_keeps_the_best_leading_ones():
    assert ordered_grasp_indices(68, 32, "cpu").tolist() == list(range(32))


def test_surplus_particles_repeat_the_grasps_cyclically_without_randomness():
    first = ordered_grasp_indices(3, 8, "cpu")
    assert first.tolist() == [0, 1, 2, 0, 1, 2, 0, 1]
    assert torch.equal(first, ordered_grasp_indices(3, 8, "cpu"))


def test_ordered_grasps_is_opt_in_so_other_pipelines_keep_random_sampling():
    assert TAMPConfiguration().ordered_grasps is False
    assert TAMPConfiguration(ordered_grasps=True).ordered_grasps is True
