"""Regression: reused output buffers must not retain costs or old pair gradients."""
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


@pytest.mark.parametrize('sphere_count', [32, 383], ids=['parallel-kernel', 'fallback-kernel'])
def test_reused_self_collision_buffers_clear_cost_and_gradient(sphere_count):
    from curobo.curobolib.geom import SelfCollisionDistance

    device = 'cuda'
    distance = torch.zeros((2, 1), device=device)
    gradient = torch.zeros((2, 1, sphere_count, 4), device=device)
    sparse = torch.zeros((2, 1, sphere_count), device=device, dtype=torch.uint8)
    offsets = torch.zeros(sphere_count, device=device)
    weight = torch.tensor([2.], device=device)
    matrix = torch.zeros((sphere_count, sphere_count), device=device, dtype=torch.uint8)
    matrix[0, 1] = matrix[1, 0] = 1
    matrix[2, 3] = matrix[3, 2] = 1
    locations = torch.empty(0, device=device, dtype=torch.int16)
    for colliding_pair in [0, None, 2, None, 0]:
        spheres = torch.zeros((2, 1, sphere_count, 4), device=device)
        spheres[..., 3] = .1
        spheres[..., 1, 0] = .5
        spheres[..., 2, 0] = 1.
        spheres[..., 3, 0] = 1.5
        if colliding_pair is not None:
            spheres[..., colliding_pair + 1, 0] = spheres[..., colliding_pair, 0] + .15
        # The second query in the batch is always collision-free.
        spheres[1, 0, 1, 0] = .5
        spheres[1, 0, 3, 0] = 1.5
        spheres.requires_grad_(True)
        cost = SelfCollisionDistance.apply(
            distance, gradient, sparse, spheres, offsets, weight, matrix,
            locations, 0, 32, False, True)
        expected = torch.tensor([[0. if colliding_pair is None else .1], [0.]], device=device)
        torch.testing.assert_close(cost, expected, atol=2e-6, rtol=1e-5)
        cost.sum().backward()
        active = torch.zeros_like(spheres.grad)
        if colliding_pair is not None:
            active[0, 0, colliding_pair, 0] = 2.
            active[0, 0, colliding_pair + 1, 0] = -2.
        torch.testing.assert_close(spheres.grad, active)
