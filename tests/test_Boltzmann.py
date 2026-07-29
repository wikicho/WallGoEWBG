"""
Tests of the boltzmann module
"""
import pytest  # for tests
import numpy as np  # arrays and maths
import pathlib
import WallGo
from WallGo.particle import ComplexMassParticle


real_path = pathlib.Path(__file__)
dir_path = pathlib.Path(__file__).parent.resolve()


@pytest.mark.parametrize(
    "spatialGridSize, momentumGridSize, a, b, c, d, e, f",
    [(25, 19, 1, 2, 3, 4, 5, 6), (5, 5, 1, 1, 2, 3, 5, 8)]
)
def test_Delta00(
    boltzmannTestBackground:  WallGo.BoltzmannBackground,
    particle: WallGo.Particle,
    spatialGridSize: int,
    momentumGridSize: int,
    a: float,
    b: float,
    c: float,
    d: float,
    e: float,
    f: float,
) -> None:
    r"""
    Tests that the Delta integral gives a known analytic result for
    :math:`\delta f = E \sqrt{(1 - \rho_z^2)(1 - \rho_\Vert)}`.
    """
    # setting up objects
    # This is the fixture background constructed with input M. pytest magic
    # that works because argument name here matches that used in fixture def
    bg = boltzmannTestBackground
    grid = WallGo.grid.Grid(spatialGridSize, momentumGridSize, 1, 100)
    collisionPath = dir_path / "TestData/N19"
    boltzmann = WallGo.BoltzmannSolver(grid, "Cardinal", "Cardinal", "Spectral")
    boltzmann.truncationOption = WallGo.ETruncationOption.NONE

    boltzmann.updateParticleList([particle])
    boltzmann.setBackground(bg)

    boltzmann.loadCollisions(collisionPath)

    # coordinates
    _, rz, rp = grid.getCompactCoordinates()  # compact
    rz = rz[np.newaxis, :, np.newaxis]
    rp = rp[np.newaxis, np.newaxis, :]
    _, pz, pp = grid.getCoordinates()  # non-compact
    pz = pz[np.newaxis, :, np.newaxis]
    pp = pp[np.newaxis, np.newaxis, :]

    # fluctuation mode
    msq = particle.msqVacuum(bg.fieldProfiles)
    ## Drop start and end points in field space
    msq = msq[1:-1, np.newaxis, np.newaxis]
    energy = np.sqrt(msq + pz**2 + pp**2)

    # integrand with known result
    eps = 2e-16
    integrandAnalytic = (
        2
        * energy
        * (1 - rz**2)
        * (1 - rp**2)
        * np.sqrt((1 - rz**2) * (1 - rp) ** 2 / (1 - rp**2 + eps))
        / (np.log(2 / (1 - rp)) + eps)
    )
    integrandAnalytic *= a + b * rz + c * rz**2
    integrandAnalytic *= d + e * rp + f * rp**2

    # doing computation
    boltzmannResults = boltzmann.getDeltas(integrandAnalytic[None, ...])
    Deltas = boltzmannResults.Deltas  # pylint: disable=invalid-name

    # comparing to analytic result
    Delta00Analytic = (4 * a + c) * (4 * d + f) * bg.temperatureProfile**3 / 64  # pylint: disable=invalid-name

    # asserting result
    np.testing.assert_allclose(
        Deltas.Delta00.coefficients[0], Delta00Analytic[1:-1], rtol=1e-14, atol=0
    )


@pytest.mark.parametrize("spatialGridSize, momentumGridSize", [(3, 3), (5, 5)])
def test_solution(
    boltzmannTestBackground: WallGo.BoltzmannBackground,
    particle: WallGo.Particle,
    spatialGridSize: int,
    momentumGridSize: int,
) -> None:
    """
    Tests that the Boltzmann equation is satisfied by the solution
    """
    # setting up objects
    # This is the fixture background constructed with input M. pytest magic
    # that works because argument name here matches that used in fixture def
    bg = boltzmannTestBackground
    grid = WallGo.grid.Grid(spatialGridSize, momentumGridSize, 1, 1)

    collisionPath = dir_path / "TestData/N11"
    boltzmann = WallGo.BoltzmannSolver(grid)
    boltzmann.updateParticleList([particle])
    boltzmann.setBackground(bg)
    boltzmann.loadCollisions(collisionPath)

    # solving Boltzmann equations
    deltaF = boltzmann.solveBoltzmannEquations()

    # building Boltzmann equation terms
    operator, source, _, _ = boltzmann.buildLinearEquations()

    # checking difference
    diff = operator @ deltaF.flatten(order="C") - source

    # getting norms
    diffNorm = np.linalg.norm(diff)
    sourceNorm = np.linalg.norm(source)
    ratio = diffNorm / sourceNorm

    # asserting solution works
    assert ratio == pytest.approx(0, abs=1e-14)


@pytest.mark.parametrize("spatialGridSize", [3])
def test_ewbg_solution_keeps_helicity_index(
    boltzmannTestBackground: WallGo.BoltzmannBackground,
    spatialGridSize: int,
) -> None:
    """The CP-odd EWBG source and solution must be odd in helicity."""

    momentumGridSize = 3
    grid = WallGo.grid.Grid(spatialGridSize, momentumGridSize, 1, 1)
    particle = ComplexMassParticle(
        name="top",
        index=0,
        msqVacuum=lambda fields: 0.5 * fields.getField(0) ** 2,
        msqDerivative=lambda fields: np.transpose([fields.getField(0)]),
        phase=lambda fields: 0.2 * fields.getField(0),
        statistics="Fermion",
        totalDOFs=12,
    )

    boltzmann = WallGo.EWBGBoltzmannSolver(
        grid,
        basisM="Cardinal",
        basisN="Cardinal",
        truncationOption=WallGo.ETruncationOption.NONE,
    )
    boltzmann.updateParticleList([particle])
    # EWBGBoltzmannSolver normally constructs this background from WallGoResults.
    # Assign the analytic test background directly to keep this a focused unit test.
    boltzmann.background = boltzmannTestBackground
    boltzmann.background.boostToPlasmaFrame()
    collisionArray = WallGo.CollisionArray(grid, "Cardinal", [particle])
    collisionArray.polynomialData.coefficients.fill(0)
    for pzIndex in range(momentumGridSize - 1):
        for ppIndex in range(momentumGridSize - 1):
            collisionArray.polynomialData.coefficients[
                0, pzIndex, ppIndex, 0, pzIndex, ppIndex
            ] = 1
    boltzmann.setCollisionArray(collisionArray)

    operator, source, _, _ = boltzmann.buildLinearEquations()

    assert tuple(boltzmann.helicities) == (-1, 1)
    assert source.shape == (operator.shape[0], 2)
    assert np.linalg.norm(source[:, 0]) > 0
    np.testing.assert_allclose(source[:, 0], -source[:, 1])

    deltaF = boltzmann.solveBoltzmannEquations()
    assert deltaF.shape == (
        1,
        2,
        spatialGridSize - 1,
        momentumGridSize - 1,
        momentumGridSize - 1,
    )
    np.testing.assert_allclose(deltaF[:, 0], -deltaF[:, 1])

    for helicity in boltzmann.helicities:
        helicityIndex = boltzmann.getHelicityIndex(helicity)
        residual = (
            operator @ deltaF[:, helicityIndex].reshape(-1)
            - source[:, helicityIndex]
        )
        relativeResidual = np.linalg.norm(residual) / np.linalg.norm(
            source[:, helicityIndex]
        )
        assert relativeResidual < 1e-13

    results = boltzmann.getDeltas(deltaF)
    assert results.Deltas.Delta10 is not None
    assert results.Deltas.Delta10.coefficients.shape == (
        1,
        2,
        spatialGridSize - 1,
    )
    np.testing.assert_allclose(
        results.Deltas.Delta10.coefficients[:, 0],
        -results.Deltas.Delta10.coefficients[:, 1],
    )
    assert np.all(np.isfinite(boltzmann.checkLinearization(deltaF)))


@pytest.mark.parametrize("helicities", [(), (-1, -1), (0, 1)])
def test_ewbg_rejects_invalid_helicities(helicities: tuple[int, ...]) -> None:
    """Only unique physical helicities can be configured."""
    grid = WallGo.grid.Grid(3, 3, 1, 1)
    with pytest.raises(ValueError, match="helicities"):
        WallGo.EWBGBoltzmannSolver(grid, helicities=helicities)


@pytest.mark.parametrize("spatialGridSize, momentumGridSize, slope", [(7, 9, -0.1), (9, 9, -0.1), (11, 9, 0.1)])
def test_checkSpectralConvergence(
    boltzmannTestBackground: WallGo.BoltzmannBackground,
    particle: WallGo.Particle,
    spatialGridSize: int,
    momentumGridSize: int,
    slope: float,
) -> None:
    """
    Tests that the Boltzmann equation is satisfied by the solution
    """
    # setting up objects
    grid = WallGo.grid.Grid(spatialGridSize, momentumGridSize, 1, 1)
    boltzmann = WallGo.BoltzmannSolver(
        grid,
        basisM="Chebyshev",
        basisN="Chebyshev",
        truncationOption=WallGo.ETruncationOption.AUTO,
    )

    # solving Boltzmann equations
    deltaFShape = (
        1,
        spatialGridSize - 1,
        momentumGridSize - 1,
        momentumGridSize - 1,
    )
    deltaF = np.zeros(deltaFShape)
    for z in range(spatialGridSize - 1):
        for pz in range(momentumGridSize - 1):
            for pp in range(momentumGridSize - 1):
                deltaF[0, z, pz, pp] = np.exp(slope*z - 1e-2*pz - 1e-3*pp) / (1 + pz) / (1 + pp)**2
        

    # checking spectral convergence
    deltaFTruncated, truncatedShape, _ = boltzmann.checkSpectralConvergence(deltaF)

    nTruncated = (spatialGridSize - 1) // 3
    expectTruncated = deltaFTruncated[:, -nTruncated:, :, :]

    # asserting truncation
    if slope > 0:
        assert truncatedShape == (1, spatialGridSize - 1 - nTruncated, momentumGridSize - 1, momentumGridSize - 1)
        assert np.allclose(expectTruncated, np.zeros_like(expectTruncated), atol=1e-2)
    else:
        assert truncatedShape == (1, spatialGridSize - 1, momentumGridSize - 1, momentumGridSize - 1)
        assert not np.allclose(expectTruncated, np.zeros_like(expectTruncated))
