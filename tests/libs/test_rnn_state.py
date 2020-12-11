import torch

from libs.inferencer import RNNState


def test_get_state_from_empty_object():
    num_layer = 2
    num_dim = 10

    empty_state = torch.zeros(num_layer, 0, num_dim)
    state = RNNState(known_user_id_idx={},
                     h_t=empty_state,
                     c_t=empty_state)

    out = state.get_state(user_ids=[0, 5, 3])

    assert out is None


def test_get_state_from_non_empty_object():

    ht_init_state = torch.stack([
        torch.tensor([[1, 1, 1], [1, 1, 1]], dtype=torch.float),
        torch.tensor([[2, 2, 2], [2, 2, 2]], dtype=torch.float),
        torch.tensor([[3, 3, 3], [3, 3, 3]], dtype=torch.float),
    ], dim=1)
    state = RNNState(known_user_id_idx={1111: 0, 2222: 1, 3333: 2},
                     h_t=ht_init_state,
                     c_t=ht_init_state*10)

    h_t, c_t = state.get_state(user_ids=[1111, 3333, 2222, 2222])

    expected_h_t = ht_init_state.index_select(dim=1, index=torch.tensor([0, 2, 1, 1]))
    expected_c_t = expected_h_t*10

    assert torch.all(h_t.eq(expected_h_t))
    assert torch.all(c_t.eq(expected_c_t))


def test_get_unknown_user_state():
    ht_init_state = torch.stack([
        torch.tensor([[1, 1, 1], [1, 1, 1]], dtype=torch.float),
        torch.tensor([[2, 2, 2], [2, 2, 2]], dtype=torch.float),
        torch.tensor([[3, 3, 3], [3, 3, 3]], dtype=torch.float),
    ], dim=1)
    state = RNNState(known_user_id_idx={1111: 0, 2222: 1, 3333: 2},
                     h_t=ht_init_state,
                     c_t=ht_init_state * 10)

    h_t, c_t = state.get_state(user_ids=[1111, 3333, 7777, 8888, 2222, 9999])

    expected_h_t = ht_init_state.index_select(dim=1, index=torch.tensor([0, 2, 1]))
    expected_h_t = torch.cat([
        expected_h_t[:, :2, :],
        torch.zeros(2, 2, 3),
        expected_h_t[:, 2:, :],
        torch.zeros(2, 1, 3),
    ], dim=1)
    expected_c_t = expected_h_t * 10

    assert torch.all(h_t.eq(expected_h_t))
    assert torch.all(c_t.eq(expected_c_t))


def test_update_and_get():
    ht_init_state = torch.stack([
        torch.tensor([[1, 1, 1], [1, 1, 1]], dtype=torch.float),
        torch.tensor([[2, 2, 2], [2, 2, 2]], dtype=torch.float),
        torch.tensor([[3, 3, 3], [3, 3, 3]], dtype=torch.float),
    ], dim=1)
    state = RNNState(known_user_id_idx={1111: 0, 2222: 1, 3333: 2},
                     h_t=ht_init_state,
                     c_t=ht_init_state * 10)

    update_user_ids = [9999, 1111, 8888]
    update_h_t = torch.stack([
        torch.tensor([[5, 6, 3], [9, 8, 7]], dtype=torch.float),
        torch.tensor([[6, 8, 1], [2, 4, 6]], dtype=torch.float),
        torch.tensor([[7, 8, 9], [1, 2, 5]], dtype=torch.float),
    ], dim=1)
    state.update_state(update_user_ids, (update_h_t, update_h_t*10))

    h_t, c_t = state.get_state(user_ids=[8888, 44, 1111, 2222])

    expected_h_t = torch.stack([
        torch.tensor([[7, 8, 9], [1, 2, 5]], dtype=torch.float),
        torch.tensor([[0, 0, 0], [0, 0, 0]], dtype=torch.float),
        torch.tensor([[6, 8, 1], [2, 4, 6]], dtype=torch.float),
        torch.tensor([[2, 2, 2], [2, 2, 2]], dtype=torch.float)
    ], dim=1)
    expected_c_t = expected_h_t * 10

    assert torch.all(h_t.eq(expected_h_t))
    assert torch.all(c_t.eq(expected_c_t))
