import numpy as np 
from ParticleFilter import Particles

class Test
# Test particle ctor
def test_nd_array_check():

    def _check(T, Lambda, sigma, xlast, zlast, N, expectedmsg):
        try: 
            example_h = Particles(T, 
                                  Lambda, 
                                  sigma, 
                                  xlast,
                                  zlast,
                                  N) 
        except AssertionError as e:
            expected = str(e) == expectedmsg
            if not expected: 
                raise AssertionError(f"Expected error message: '{expectedmsg}' but received {str(e)}")
            return 

        raise AssertionError("Input type check for np.ndarray did not trigger an assertion error." + 
                             f" Expected to see error msg: {expectedmsg}")

    example_T = np.array([100, 100, 1, 90])
    example_lambda = np.array([0, 0, 0, 0])
    example_sigma = np.array([1e-19, 0.01, 0.01, 25])
    example_X_last = np.array([100, 100, 100, 100])
    example_Z_last = np.array([0, 0, 0, 0])
    N = 4
    
    expectedmsg = "T is not a numpy array"
    _check(list(example_T), example_lambda, example_sigma, example_X_last, example_Z_last, N, expectedmsg)

    expectedmsg = "Lambda is not a numpy array"
    _check(example_T, list(example_lambda), example_sigma, example_X_last, example_Z_last, N, expectedmsg)

    expectedmsg = "Sigma is not a numpy array"
    _check(example_T, example_lambda, list(example_sigma), example_X_last, example_Z_last, N, expectedmsg)

    expectedmsg = "X is not a numpy array"
    _check(example_T, example_lambda, example_sigma, list(example_X_last), example_Z_last, N, expectedmsg)

    expectedmsg = "Z is not a numpy array"
    _check(example_T, example_lambda, example_sigma, example_X_last, list(example_Z_last), N, expectedmsg)


def test_arr_length_check():
    example_T = np.array([100, 100, 1, 90])
    example_lambda = np.array([0, 0, 0, 0])
    example_sigma = np.array([1e-19, 0.01, 0.01, 25])
    example_X_last = np.array([100, 100, 100])
    example_Z_last = np.array([0, 0, 0])
    N = 4
    
    try: 
        example_h = Particles(example_T, 
                            example_lambda, 
                            example_sigma, 
                            example_X_last,
                            example_Z_last,
                            N)
    except AssertionError as e:
        expectedmsg = ""
        expected = str(e) == expectedmsg
        if expected: 
            raise AssertionError(f"Expected error message: '{expectedmsg}' but received {str(e)}")
        return

    raise AssertionError("input length check did not trigger an assertion error")

def test_len():
    example_T = np.array([100, 100, 1, 90])
    example_lambda = np.array([0, 0, 0, 0])
    example_sigma = np.array([1e-19, 0.01, 0.01, 25])
    example_X_last = np.array([100, 100, 100, 100])
    example_Z_last = np.array([0, 0, 0, 0])
    N = 4

    example_h = Particles(example_T, 
                        example_lambda, 
                        example_sigma, 
                        example_X_last,
                        example_Z_last,
                        N)

    assert len(example_h) == N, f"Expected len(Particle) to return {N}, but got {len(example_h)} instead"

def test_shape():
    example_T = np.array([100, 100, 1, 90])
    example_lambda = np.array([0, 0, 0, 0])
    example_sigma = np.array([1e-19, 0.01, 0.01, 25])
    example_X_last = np.array([100, 100, 100, 100])
    example_Z_last = np.array([0, 0, 0, 0])
    N = 4

    example_h = Particles(example_T, 
                        example_lambda, 
                        example_sigma, 
                        example_X_last,
                        example_Z_last,
                        N)

    expected_shape = (5, N)
    assert example_h.shape == expected_shape, f"Expected Particle.shape property to return {expected_shape}, " + \
                                              f"but got {example_h.shape} instead"

def test_as_matrix():
    example_T = np.array([100, 100, 1, 90])
    example_lambda = np.array([0, 0, 0, 0])
    example_sigma = np.array([1e-19, 0.01, 0.01, 25])
    example_X_last = np.array([100, 100, 100, 100])
    example_Z_last = np.array([0, 0, 0, 0])
    N = 4

    example_h = Particles(example_T, 
                        example_lambda, 
                        example_sigma, 
                        example_X_last,
                        example_Z_last,
                        N)

    h_mat = example_h.as_matrix()

    assert h_mat.shape == (5, N), f"as_matrix() method returned a matrix of shape {h_mat.shape}. " + \
        f"Expected shape of {(5,N)}"

    expected = np.stack((example_T, example_lambda, example_sigma, example_X_last, example_Z_last))

    assert np.array_equal(h_mat, expected), "as_matrix() returned a value that was not expected."

def test_particle_eq():

    t = np.array(range(0,10))
    l = np.array(range(0,10)) * 2 # multiply just to differentiate values for each var 
    s = np.array(range(0,10)) * 3
    x = np.array(range(0,10)) * 4 
    z = np.array(range(0,10)) * 5

    h1 = Particles(t, l, s, x, z, len(t))
    h2 = Particles(t, l, s, x, z, len(t))

    assert h1 == h2, f"Test case 1: __eq__ method check returned False when True was expected."

    t = np.array(range(0,10)) * 100
    l = np.array(range(0,10)) * 2 # multiply just to differentiate values for each var 
    s = np.array(range(0,10)) * 3
    x = np.array(range(0,10)) * 4 
    z = np.array(range(0,10)) * 5

    h3 = Particles(t, l, s, x, z, len(t))

    assert h1 != h3, f"Test case 2: __eq__ method check returned True when False was expected."

    t = np.array([1])
    l = np.array([1])
    s = np.array([1])
    x = np.array([1])
    z = np.array([1])
    h4 = Particles(t, l, s, x, z, 1)
    h5 = Particles(t, l, s, x, z, 1)

    assert h4 != h1, "Test case 3: __eq__ method check returned True when False was expected."
    assert h4 != h2, "Test case 4: __eq__ method check returned True when False was expected."
    assert h4 != h3, "Test case 5: __eq__ method check returned True when False was expected."
    assert h4 == h5, f"Test case 6: __eq__ method check returned False when True was expected."


def test_particle_getitem():
    N = 10
    t = np.array(range(0,N))
    l = np.array(range(0,N)) * 2 # multiply just to differentiate values for each var 
    s = np.array(range(0,N)) * 3
    x = np.array(range(0,N)) * 4 
    z = np.array(range(0,N)) * 5

    h = Particles(t, l, s, x, z, len(t))

    # test out of index for integer index 
    try: 
        M = N + 100
        h[N]
        raise AssertionError(f"Expected Particles[{M}] to raise IndexError as Particles " + \
            f"object contains {M} particles, but no IndexError was raised")
    except IndexError as e: 
        pass 

    # test out of index for slice
    try: 
        sidx, eidx = int(N/2), N + 100
        h[sidx:eidx]
        raise AssertionError(f"Expected Particles[{sidx}:{eidx}] to raise IndexError as Particles " + \
            f"object contains {M} particles, but no IndexError was raised")
    except IndexError as e:
        pass 

    # test integer index 
    for i in range(-3, N):
        expected = Particles(np.array([t[i]]), 
                             np.array([l[i]]), 
                             np.array([s[i]]), 
                             np.array([x[i]]), 
                             np.array([z[i]]), 
                             1)
        msg = f"{i}: Expected {h[i]} to be equal to {expected} but __eq__ returned False"
        assert h[i] == expected,  msg

    sidx = 0
    eidx = 3
    expected = Particles(np.array(t[sidx:eidx]), 
                         np.array(l[sidx:eidx]), 
                         np.array(s[sidx:eidx]), 
                         np.array(x[sidx:eidx]), 
                         np.array(z[sidx:eidx]),
                         eidx - sidx)
    got = h[sidx:eidx]
    assert got == expected, f"Expected {expected} but received {got} for Particles[{sidx}:{eidx}]"

    sidx = -3
    eidx = -1
    expected = Particles(np.array(t[sidx:eidx]), 
                         np.array(l[sidx:eidx]), 
                         np.array(s[sidx:eidx]), 
                         np.array(x[sidx:eidx]), 
                         np.array(z[sidx:eidx]),
                         2)
    got = h[sidx:eidx]
    assert got == expected, f"Expected {expected} but received {got} for Particles[{sidx}:{eidx}]"

def test_particle_setitem():
    N = 10
    t = np.ones(N)
    l = np.ones(N)
    s = np.ones(N)
    x = np.ones(N)
    z = np.ones(N)

    h = Particles(t, l, s, x, z, len(t))

    # set single index
    arr = np.ones(N)
    for idx in range(-3,N):
        h[idx] = Particles(np.zeros(1),np.zeros(1),np.zeros(1),np.zeros(1),np.zeros(1),1)
        arr[idx] = 0
        expected = Particles(arr, arr, arr, arr, arr, N)
        assert h == expected, f"Expected {expected} but got {h} when setting Particles[{idx}]"

        assert np.array_equal(h.T, arr), f"Expected h.T to give {arr} but got {h.T}"
        assert np.array_equal(h.Lambda,arr), f"Expected h.Lambda to give {arr} but got {h.Lambda}"
        assert np.array_equal(h.Sigma,arr), f"Expected h.Sigma to give {arr} but got {h.Sigma}"
        assert np.array_equal(h.X,arr), f"Expected h.X to give {arr} but got {h.X}"
        assert np.array_equal(h.Z,arr), f"Expected h.Z to give {arr} but got {h.Z}"

    # set using slice
    t = np.ones(N)
    l = np.ones(N)
    s = np.ones(N)
    x = np.ones(N)
    z = np.ones(N)

    h = Particles(t, l, s, x, z, len(t)) # reset 
    sidx, eidx = 3,6
    n = eidx - sidx 
    tmp = np.ones(n) * 100
    h[sidx:eidx] = Particles(tmp, tmp, tmp, tmp, tmp, n)
    arr = np.ones(N)
    arr[sidx:eidx] = 100

    assert np.array_equal(h.T, arr), f"Expected h.T to give {arr} but got {h.T}"
    assert np.array_equal(h.Lambda,arr), f"Expected h.Lambda to give {arr} but got {h.Lambda}"
    assert np.array_equal(h.Sigma,arr), f"Expected h.Sigma to give {arr} but got {h.Sigma}"
    assert np.array_equal(h.X,arr), f"Expected h.X to give {arr} but got {h.X}"
    assert np.array_equal(h.Z,arr), f"Expected h.Z to give {arr} but got {h.Z}"

    # set usng slice with neatices 

def test_particle_container_properties():
    test_particle_eq()
    test_particle_getitem()
    test_particle_setitem()

test_particle_container_properties()