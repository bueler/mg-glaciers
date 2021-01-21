from meshlevel import MeshLevel1D
import numpy as np

def test_basics():
    ml = MeshLevel1D(k=2)
    assert ml.m == 7
    assert ml.l2norm(ml.zeros()) == 0.0
    v = ml.zeros()
    assert len(v) == ml.m + 2
    s = np.sqrt(ml.xx())
    assert (ml.l2norm(ml.xx()) - 1.0/np.sqrt(2.0)) < 1.0e-10

def test_cR():
    ml = MeshLevel1D(k=1)
    assert ml.m == 3
    assert ml.mcoarser == 1
    r = ml.zeros()
    r[1:4] = 1.0
    assert all(ml.cR(r) == [0.0,2.0,0.0])

def test_P():
    ml = MeshLevel1D(k=1)
    v = np.array([0.0,1.0,0.0])
    assert all(ml.P(v) == [0.0,0.5,1.0,0.5,0.0])

def test_mR():
    ml = MeshLevel1D(k=2)
    assert ml.m == 7
    v = np.array([0.0,1.0,1.0,0.5,0.5,0.5,0.5,1.0,0.0])
    assert all(ml.mR(v) == [0.0,1.0,0.5,1.0,0.0])
    vback = ml.P(ml.mR(v))  # note P uses zero boundary values
    assert all(vback[2:-2] >= v[2:-2])
    assert all(vback[[1,-2]] == [0.5,0.5])

