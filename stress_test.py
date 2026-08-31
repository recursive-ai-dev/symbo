import numpy as np
import sympy as sp

import symbo

def test_tensor_nd():
    print("Testing n-dimensional tensor...")
    # 3D tensor
    nt = symbo.NanoTensor((2, 2, 2), max_order=2)
    x, y, z = sp.symbols('x y z')
    nt.data[0, 1, 0] = x**2 * y + z
    res = nt.eval_numeric({'x': 2, 'y': 3, 'z': 4})
    assert res[0, 1, 0] == 2**2 * 3 + 4

def test_taylor():
    print("Testing Taylor expansion...")
    nt = symbo.NanoTensor((1,), max_order=2, base_vars=['k', 'a'])
    nt.generate_taylor({'k': 1.0, 'a': 0.0}, ss_value=sp.S(1))
    # It sets nt.data[0] to taylor expansion
    assert len(nt.coeff_vars) > 0

def test_groebner():
    print("Testing Groebner...")
    nt = symbo.NanoTensor((1,))
    x, y = sp.symbols('x y')
    F = [x**2 - y, y**2 - x]
    sol = nt.groebner_solve(F, [x, y])
    assert sol is not None

def test_serialization():
    print("Testing serialization...")
    x, y = sp.symbols('x y')
    F = [x**2 - y, y**2 - x]
    G = sp.groebner(F, x, y)

    # MessagePack
    msg = symbo.serialize_basis_msgpack(G)
    print("Msgpack OK, size:", len(msg))
    assert len(msg) > 0, "MessagePack serialization failed"

    # Arrow
    arr = symbo.serialize_basis_arrow(G)
    print("Arrow OK, size:", len(arr))
    assert len(arr) > 0, "Arrow serialization failed"

def test_wasm():
    print("Testing WASM...")
    res = symbo.wasm_eval_expression("x**2 + y", {"x": 2.0, "y": 3.0})
    assert abs(res - 7.0) < 1e-9

def test_pathfinding():
    print("Testing pathfinding...")
    nt = symbo.NanoTensor((1,))
    Z = np.array([[1, 1, 5], [1, 5, 1], [1, 1, 1]])
    path = nt.find_path_on_grid(Z, start=(0,0), goal=(2,2))
    assert path is not None

def test_explainability():
    print("Testing explainability...")
    nt = symbo.NanoTensor((1,))
    x, y = sp.symbols('x y')
    nt.data[0] = x**2 + y
    tree = symbo.deriv_tree(nt, ['x', 'y'])
    assert 'x' in tree

def test_military():
    print("Testing military features...")
    nt = symbo.NanoTensor((1,), max_order=2, base_vars=['x'])
    nt.set_validation_bounds('x', -10, 10)
    nt.data[0] = sp.Symbol('x')**2

    # Do enough operations to keep health high
    for _ in range(25):
        nt.eval_numeric({'x': 1})

    res = nt.eval_numeric({'x': 5})
    assert res[0] == 25

    try:
        nt.eval_numeric({'x': 15})
    except ValueError:
        pass
    else:
        raise AssertionError("out-of-bounds input should have raised ValueError")

    health = nt.health_check()
    assert health['status'] in ['optimal', 'good', 'degraded'], f"Health Status is {health['status']}"


def run_all():
    test_tensor_nd()
    test_taylor()
    test_groebner()
    test_serialization()
    test_wasm()
    test_pathfinding()
    test_explainability()
    test_military()
    print("All basic stress tests completed!")

if __name__ == '__main__':
    run_all()
