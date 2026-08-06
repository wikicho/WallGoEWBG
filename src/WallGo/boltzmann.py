"""
Classes for solving the Boltzmann equations for out-of-equilibrium particles.
"""

import sys
import typing
from copy import deepcopy
import logging
import pathlib
from enum import Enum, auto
import numpy as np
import findiff  # finite difference methods
from .containers import BoltzmannBackground, BoltzmannDeltas
from .grid import Grid
from .polynomial import Polynomial, SpectralConvergenceInfo
from .particle import Particle, ComplexMassParticle, ChiralParticle, KineticState
from .collisionArray import CollisionArray
from .results import BoltzmannResults, WallGoResults
from .exceptions import CollisionLoadError

if typing.TYPE_CHECKING:
    import importlib


class ETruncationOption(Enum):
    """Enums for what to do with truncating the spectral expansion."""

    NONE = auto()
    """Do not truncate early, use all coefficients."""

    AUTO = auto()
    """Truncate early if it seems the UV is not converging."""

    THIRD = auto()
    """Drop the last third of the coefficients."""


class EWBGSourceType(Enum):
    """Source component used by :class:`EWBGBoltzmannSolver`."""

    EVEN = auto()
    """CP-even source from mass, temperature, and fluid-velocity gradients."""

    ODD = auto()
    """CP-odd, helicity-odd semiclassical source."""

    TOTAL = auto()
    """Sum of the CP-even and CP-odd source terms."""


class BoltzmannSolver:
    """
    Class for solving Boltzmann equations for small deviations from equilibrium.
    """

    # Static value holding of natural log of the maximum expressible float
    MAX_EXPONENT: typing.Final[float] = sys.float_info.max_exp * np.log(2)

    # Member variables
    grid: Grid
    offEqParticles: list[Particle]
    background: BoltzmannBackground
    collisionArray: CollisionArray
    truncationOption: ETruncationOption

    def __init__(
        self,
        grid: Grid,
        basisM: str = "Cardinal",
        basisN: str = "Chebyshev",
        derivatives: str = "Spectral",
        collisionMultiplier: float = 1.0,
        truncationOption: ETruncationOption = ETruncationOption.AUTO,
    ):
        """
        Initialisation of BoltzmannSolver

        Parameters
        ----------
        grid : Grid
            An object of the Grid class.
            integrals.
        basisM : str, optional
            The position polynomial basis type, either 'Cardinal' or 'Chebyshev'.
            Default is 'Cardinal'.
        basisN : str, optional
            The momentum polynomial basis type, either 'Cardinal' or 'Chebyshev'.
            Default is 'Chebyshev'.
        derivatives : {'Spectral', 'Finite Difference'}, optional
            Choice of method for computing derivatives. Default is 'Spectral'
            which is expected to be more accurate.
        collisionMultiplier : float, optional
            Factor by which the collision term is multiplied. Can be used for testing.
            Default is 1.0.
        truncationOption : ETruncationOption, optional
            Option for truncating the spectral expansion. Default is
            ETruncationOption.AUTO. Other options
            are ETruncationOption.NONE and ETruncationOption.THIRD.

        Returns
        -------
        cls : BoltzmannSolver
            An object of the BoltzmannSolver class.
        """

        self.grid = grid
        BoltzmannSolver._checkDerivatives(derivatives)
        self.derivatives = derivatives
        BoltzmannSolver._checkBasis(basisM)
        BoltzmannSolver._checkBasis(basisN)
        if derivatives == "Finite Difference":
            assert (
                basisM == "Cardinal" and basisN == "Cardinal"
            ), "Must use Cardinal basis for Finite Difference method"

        # Position polynomial type
        self.basisM = basisM
        # Momentum polynomial type
        self.basisN = basisN

        self.collisionMultiplier = collisionMultiplier
        self.truncationOption = truncationOption

        # These are set, and can be updated, by our member functions
        # TODO: are these None types the best way to go?
        self.background = None  # type: ignore[assignment]
        self.collisionArray = None  # type: ignore[assignment]
        self.offEqParticles = []

    def setBackground(self, background: BoltzmannBackground) -> None:
        """
        Setter for the BoltzmannBackground
        """
        self.background = deepcopy(
            background
        )  # do we need a deepcopy? Does this even work generally?
        self.background.boostToPlasmaFrame()

    def setCollisionArray(self, collisionArray: CollisionArray) -> None:
        """
        Setter for the CollisionArray
        """
        self.collisionArray = collisionArray

    def updateParticleList(self, offEqParticles: list[Particle]) -> None:
        """
        Setter for the list of out-of-equilibrium Particle objects
        """
        # TODO: update the collision array as well when one updates the particle list
        for p in offEqParticles:
            assert isinstance(p, Particle)

        self.offEqParticles = offEqParticles

    def getDeltas(
        self,
        deltaF: typing.Optional[np.ndarray] = None,
    ) -> BoltzmannResults:
        """
        Computes Deltas necessary for solving the Higgs equation of motion.

        These are defined in equation (15) of 2204.13120 [LC22]_.

        Parameters
        ----------
        deltaF : array_like, optional
            The deviation of the distribution function from local thermal
            equilibrium.

        Returns
        -------
        Deltas : BoltzmannDeltas
            Defined in equation (15) of [LC22]_. A collection of 4 arrays,
            each of which is of size :py:data:`len(z)`.
        """
        # checking if result pre-computed
        if deltaF is None:
            deltaF = self.solveBoltzmannEquations()

        # checking spectral convergence
        deltaF, shapeTruncated, spectralPeaks = self.checkSpectralConvergence(deltaF)

        # getting (optimistic) estimate of truncation error
        truncationError = self.estimateTruncationError(
            deltaF, shapeTruncated
        )
        truncatedTail = (
            shapeTruncated[1] != deltaF.shape[1],
            shapeTruncated[2] != deltaF.shape[2],
            shapeTruncated[3] != deltaF.shape[3],
        )

        particles = self.offEqParticles

        # constructing Polynomial class from deltaF array
        deltaFPoly = Polynomial(
            deltaF,
            self.grid,
            ("Array", self.basisM, self.basisN, self.basisN),
            ("Array", "z", "pz", "pp"),
            False,
        )
        deltaFPoly.changeBasis(("Array", "Cardinal", "Cardinal", "Cardinal"))

        # Take all field-space points, but throw the boundary points away
        # TODO: LN: why throw away boundary points?
        field = self.background.fieldProfiles.takeSlice(
            1, -1, axis=self.background.fieldProfiles.overFieldPoints
        )

        # adding new axes, to make everything rank 3 like deltaF (z, pz, pp)
        # for fast multiplication of arrays, using numpy's broadcasting rules
        pz = self.grid.pzValues[None, None, :, None]
        pp = self.grid.ppValues[None, None, None, :]
        msq = np.array([particle.msqVacuum(field) for particle in particles])[
            :, :, None, None
        ]
        # constructing energy with (z, pz, pp) axes
        energy = np.sqrt(msq + pz**2 + pp**2)

        _, dpzdrz, dppdrp = self.grid.getCompactificationDerivatives()
        dpzdrz = dpzdrz[None, None, :, None]
        dppdrp = dppdrp[None, None, None, :]

        # base integrand, for '00'
        integrand = dpzdrz * dppdrp * pp / (4 * np.pi**2 * energy)

        Delta00 = deltaFPoly.integrate(  # pylint: disable=invalid-name
            (2, 3), integrand
        )
        Delta02 = deltaFPoly.integrate(  # pylint: disable=invalid-name
            (2, 3), pz**2 * integrand
        )
        Delta20 = deltaFPoly.integrate(  # pylint: disable=invalid-name
            (2, 3), energy**2 * integrand
        )
        Delta11 = deltaFPoly.integrate(  # pylint: disable=invalid-name
            (2, 3), energy * pz * integrand
        )

        Deltas = BoltzmannDeltas(  # pylint: disable=invalid-name
            Delta00=Delta00, Delta02=Delta02, Delta20=Delta20, Delta11=Delta11
        )

        # returning results
        return BoltzmannResults(
            deltaF=deltaF,
            Deltas=Deltas,
            truncationError=truncationError,
            truncatedTail=truncatedTail,
            spectralPeaks=spectralPeaks,
        )

    def solveBoltzmannEquations(self) -> np.ndarray:
        r"""
        Solves Boltzmann equation for :math:`\delta f`, equation (32) of [LC22].

        The Boltzmann equations are linearised and expressed in a spectral expansion,
        so that they take the form

        .. math::
            \left(\mathcal{L}[\alpha,\beta,\gamma;i,j,k]\delta_{ab} + \bar T_i(\chi^{(\alpha)})\mathcal{C}_{ab}[\beta,\gamma; j,k] \right) \delta f^b_{ijk} = \mathcal{S}_a[\alpha,\beta,\gamma],

        where :math:`\mathcal{L}` is the Lioville operator, :math:`\mathcal{C}`
        is the collision operator, and :math:`\mathcal{S}` is the source.

        As regards the indicies,

            - :math:`\alpha, \beta, \gamma` denote points on the coordinate lattice :math:`\{\xi^{(\alpha)},p_{z}^{(\beta)},p_{\Vert}^{(\gamma)}\}`,

            - :math:`i, j, k` denote elements of the basis of spectral functions :math:`\{\bar{T}_i, \bar{T}_j, \tilde{T}_k\}`,

            - :math:`a, b` denote particle species.

        For more details see the WallGo paper.

        Parameters
        ----------

        Returns
        -------
        delta_f : array_like
            The deviation from equilibrium, a rank 6 array, with shape
            :py:data:`(len(z), len(pz), len(pp), len(z), len(pz), len(pp))`.

        References
        ----------
        .. [LC22] B. Laurent and J. M. Cline, First principles determination
            of bubble wall velocity, Phys. Rev. D 106 (2022) no.2, 023501
            doi:10.1103/PhysRevD.106.023501
        """

        # contructing the various terms in the Boltzmann equation
        operator, source, _, _ = self.buildLinearEquations()

        # solving the linear system: operator.deltaF = source
        deltaF = np.linalg.solve(operator, source)

        # returning result
        deltaFShape = (
            len(self.offEqParticles),
            self.grid.M - 1,
            self.grid.N - 1,
            self.grid.N - 1,
        )
        deltaF = np.reshape(deltaF, deltaFShape, order="C")

        return deltaF

    def estimateTruncationError(self, deltaF: np.ndarray, shapeTruncated: tuple[int, ...]) -> float:
        r"""
        Quick estimate of the polynomial truncation error using
        John Boyd's Rule-of-thumb-2: the last coefficient of a Chebyshev
        polynomial expansion is the same order-of-magnitude as the truncation
        error.

        Parameters
        ----------
        deltaF : array_like
            The solution for which to estimate the truncation error,
            a rank 3 array, with shape :py:data:`(len(z), len(pz), len(pp))`.

        Returns
        -------
        truncationError : float
            Estimate of the relative trucation error.
        """
        # constructing Polynomial
        basisTypes = ("Array", self.basisM, self.basisN, self.basisN)
        basisNames = ("Array", "z", "pz", "pp")
        deltaFPoly = Polynomial(deltaF, self.grid, basisTypes, basisNames, False)

        # sum(|deltaF|) as the norm
        deltaFPoly.changeBasis(("Array", "Chebyshev", "Chebyshev", "Chebyshev"))
        deltaFTuncated = deltaFPoly.coefficients[
            :shapeTruncated[0],
            :shapeTruncated[1],
            :shapeTruncated[2],
            :shapeTruncated[3],
        ]
        deltaFSumAbs = np.sum(
            np.abs(deltaFTuncated),
            axis=(1, 2, 3),
        )

        # estimating truncation errors in each direction
        truncationErrorChi = np.sum(
            np.abs(deltaFTuncated[:, -1, :, :]),
            axis=(1, 2),
        ) / deltaFSumAbs
        truncationErrorPz = np.sum(
            np.abs(deltaFTuncated[:, :, -1, :]),
            axis=(1, 2),
        ) / deltaFSumAbs
        truncationErrorPp = np.sum(
            np.abs(deltaFTuncated[:, :, :, -1]),
            axis=(1, 2),
        ) / deltaFSumAbs

        # estimating the total truncation error as the maximum of these three
        return max(  # type: ignore[no-any-return]
            np.max(truncationErrorChi),
            np.max(truncationErrorPz),
            np.max(truncationErrorPp),
        )

    def checkSpectralConvergence(self, deltaF: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int], tuple[int, int, int]]:
        """
        Check for spectral convergence.

        Fits to the exponential slope of the last 1/3 of coefficients in the
        Chebyshev basis, and truncates if they are increasing. Also returns the
        positions of the spectral peaks of the distribution in each dimension.

        Parameters
        ----------
        deltaF : array_like
            The solution for which to estimate the truncation error,
            a rank 3 array, with shape :py:data:`(len(z), len(pz), len(pp))`.

        Returns
        -------
        deltaFTruncated : np.ndarray
            Potentially truncated version of input :py:data:`deltaF`, padded with zeros if truncated, so same shape as input.
        shapeTruncated : tuple[int, int, int, int]
            Shape of truncated array.
        spectralPeaks : tuple[int, int, int]
            Indices of the peaks in the (potentially truncated) spectral expansion.
        """
        # constructing Polynomial
        basisTypes = ("Array", self.basisM, self.basisN, self.basisN)
        basisNames = ("Array", "z", "pz", "pp")
        deltaFPoly = Polynomial(deltaF, self.grid, basisTypes, basisNames, False)
        truncatedShape = list(deltaF.shape)

        # changing to Chebyshev basis
        deltaFPoly.changeBasis(("Array", "Chebyshev", "Chebyshev", "Chebyshev"))

        # looking at convergence of spectral expansion
        spectralCoeffsChi = np.sum(
            np.abs(deltaFPoly.coefficients),
            axis=(0, 2, 3),
        )
        spectralCoeffsPz = np.sum(
            np.abs(deltaFPoly.coefficients),
            axis=(0, 1, 3),
        )
        spectralCoeffsPp = np.sum(
            np.abs(deltaFPoly.coefficients),
            axis=(0, 1, 2),
        )

        # how much to cut, if truncating
        cutSpatial = -((self.grid.M - 1) // 3)
        cutMomentum = -((self.grid.N - 1) // 3)

        # checking spectral convergence of spatial direction
        chiConvergenceTailInfo = SpectralConvergenceInfo(
            spectralCoeffsChi[cutSpatial:],
            # weightPower=0,
            offset=self.grid.M - 1 + cutSpatial,
        )

        # checking spectral convergence of pz direction
        pzConvergenceTailInfo = SpectralConvergenceInfo(
            spectralCoeffsPz[cutMomentum:],
            # weightPower=1,  # removed as max(pz) only grows as log(N)
            offset=self.grid.N - 1 + cutMomentum,
        )

        # checking spectral convergence of pp direction
        ppConvergenceTailInfo = SpectralConvergenceInfo(
            spectralCoeffsPp[cutMomentum:],
            # weightPower=2,  # removed as max(pp) only grows as log(N)
            offset=self.grid.N - 1 + cutMomentum,
        )

        allTailsConverging = (
            chiConvergenceTailInfo.apparentConvergence and
            pzConvergenceTailInfo.apparentConvergence and
            ppConvergenceTailInfo.apparentConvergence
        )

        # Deciding what to do based on truncationOption
        if self.truncationOption == ETruncationOption.AUTO:
            # if the slope is not definitely negative, we will truncate
            if not chiConvergenceTailInfo.apparentConvergence:
                deltaFPoly.coefficients[:, cutSpatial:, :, :] = 0
                truncatedShape[1] = deltaF.shape[1] + cutSpatial
            if not pzConvergenceTailInfo.apparentConvergence:
                deltaFPoly.coefficients[:, :, cutMomentum:, :] = 0
                truncatedShape[2] = deltaF.shape[2] + cutMomentum
            if not ppConvergenceTailInfo.apparentConvergence:
                deltaFPoly.coefficients[:, :, :, cutMomentum:] = 0
                truncatedShape[3] = deltaF.shape[3] + cutMomentum
        elif self.truncationOption == ETruncationOption.THIRD:
            # truncating regardless
            deltaFPoly.coefficients[:, cutSpatial:, :, :] = 0
            deltaFPoly.coefficients[:, :, cutMomentum:, :] = 0
            deltaFPoly.coefficients[:, :, :, cutMomentum:] = 0
            truncatedShape[1:] = [
                deltaF.shape[1] + cutSpatial,
                deltaF.shape[2] + cutMomentum,
                deltaF.shape[3] + cutMomentum,
            ]
            if allTailsConverging:
                logging.info(
                    "Tails of spectral expansions converging but truncated, consider changing truncation option."
                )
        else:
            # not truncating regardless
            if not allTailsConverging:
                logging.info(
                    "Tails of spectral expansions not converging, consider changing truncation option, or changing grid parameters."
                )

        # checking spectral convergence of z direction
        chiConvergenceInfo = SpectralConvergenceInfo(
            spectralCoeffsChi[:truncatedShape[1]], weightPower=0
        )

        # checking spectral convergence of pz direction
        pzConvergenceInfo = SpectralConvergenceInfo(
            spectralCoeffsPz[:truncatedShape[2]], weightPower=1
        )

        # checking spectral convergence of pp direction
        ppConvergenceInfo = SpectralConvergenceInfo(
            spectralCoeffsPp[:truncatedShape[3]], weightPower=2
        )

        # putting together the spectral peaks
        spectralPeaks = (
            chiConvergenceInfo.spectralPeak,
            pzConvergenceInfo.spectralPeak,
            ppConvergenceInfo.spectralPeak,
        )

        if self.truncationOption == ETruncationOption.NONE:
            return deltaF, tuple(truncatedShape), spectralPeaks

        # changing back to original basis
        deltaFPoly.changeBasis(basisTypes)

        return deltaFPoly.coefficients, tuple(truncatedShape), spectralPeaks

    @staticmethod
    def _smoothTruncation(length: int, cut: int, sharp: float = 3) -> np.ndarray:
        """
        Internal function to smooth the truncation of the spectral expansion. """
        x = np.arange(length)
        return 1 / (1 + np.exp(sharp * (x - cut)))

    def checkLinearization(
        self, deltaF: typing.Optional[np.ndarray] = None
    ) -> tuple[float, float]:
        r"""
        Compute two criteria to verify the validity of the linearisation of the
        Boltzmann equation: :math:`\delta f/f_{eq}` and
        :math:`\delta f_2/(f_{eq}+\delta f)`, with :math:`\delta f_2` the first-order
        correction due to nonlinearities.
        To be valid, at least one of the two criteria must be small for each particle.

        Parameters
        ----------
        deltaF : array-like, optional
            Solution of the Boltzmann equation. The default is None.

        Returns
        -------
        deltaFCriterion : tuple
        collCriterion : tuple
            Criteria for the validity of the linearization.

        """
        if deltaF is None:
            deltaF = self.solveBoltzmannEquations()

        particles = self.offEqParticles

        # constructing Polynomial class from deltaF array
        deltaFPoly = Polynomial(
            deltaF,
            self.grid,
            ("Array", self.basisM, self.basisN, self.basisN),
            ("z", "z", "pz", "pp"),
            False,
        )
        deltaFPoly.changeBasis(("Array", "Cardinal", "Cardinal", "Cardinal"))

        # Computing \delta f^2
        deltaFSqPoly = deltaFPoly * deltaFPoly
        deltaFSqPoly.changeBasis(("Array", self.basisM, self.basisN, self.basisN))

        operator, _, _, collision = self.buildLinearEquations()
        source = np.sum(
            collision * deltaFSqPoly.coefficients[None, None, None, None, ...],
            axis=(4, 5, 6, 7),
        )

        # Computing the correction from nonlinear terms
        deltaNonlin = np.linalg.solve(
            operator, np.reshape(source, source.size, order="C")
        )
        deltaNonlinShape = (
            len(self.offEqParticles),
            self.grid.M - 1,
            self.grid.N - 1,
            self.grid.N - 1,
        )
        deltaNonlin = np.reshape(deltaNonlin, deltaNonlinShape, order="C")
        deltaNonlinPoly = Polynomial(
            coefficients=deltaNonlin,
            grid=self.grid,
            basis=("Array", self.basisM, self.basisN, self.basisN),
            direction=("z", "z", "pz", "pp"),
            endpoints=False,
        )
        deltaNonlinPoly.changeBasis(("Array", "Cardinal", "Cardinal", "Cardinal"))

        msqFull = np.array(
            [
                particle.msqVacuum(self.background.fieldProfiles)
                for particle in particles
            ]
        )

        msqPoly = Polynomial(
            msqFull,
            self.grid,
            ("Array", "Cardinal"),
            "z",
            True,
        )
        dmsqdChi = msqPoly.derivative(axis=1).coefficients[:, 1:-1, None, None]

        # adding new axes, to make everything rank 3 like deltaF (z, pz, pp)
        # for fast multiplication of arrays, using numpy's broadcasting rules
        pz = self.grid.pzValues[None, None, :, None]
        pp = self.grid.ppValues[None, None, None, :]
        msq = msqFull[:, 1:-1, None, None]
        # constructing energy with (z, pz, pp) axes
        energy = np.sqrt(msq + pz**2 + pp**2)

        temperature = self.background.temperatureProfile[None, 1:-1, None, None]
        statistics = np.array(
            [-1 if particle.statistics == "Fermion" else 1 for particle in particles]
        )[:, None, None, None]

        fEq = BoltzmannSolver._feq(energy / temperature, statistics)
        fEqPoly = Polynomial(
            fEq,
            self.grid,
            ("Array", "Cardinal", "Cardinal", "Cardinal"),
            ("z", "z", "pz", "pp"),
            False,
        )

        _, dpzdrz, dppdrp = self.grid.getCompactificationDerivatives()
        dpzdrz = dpzdrz[None, None, :, None]
        dppdrp = dppdrp[None, None, None, :]

        dofs = np.array([particle.totalDOFs for particle in particles])[
            :, None, None, None
        ]
        integrand = dofs * dmsqdChi * dpzdrz * dppdrp * pp / (4 * np.pi**2 * energy)

        # Computing the pressure contributions of the equilibrium part, the linear
        # out-of-equilibrium part and the first-order correction due to nonlinearities.
        pressureEq = np.sum(fEqPoly.integrate((1, 2, 3), integrand).coefficients)
        pressureDeltaF = np.sum(deltaFPoly.integrate((1, 2, 3), integrand).coefficients)
        pressureNonlin = np.sum(
            deltaNonlinPoly.integrate((1, 2, 3), integrand).coefficients
        )

        # Computing the 2 linearisation criteria
        criterion1 = abs(pressureDeltaF / pressureEq)
        criterion2 = abs(pressureNonlin / (pressureEq + pressureDeltaF))

        return criterion1, criterion2

    def buildLinearEquations(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Constructs matrix and source for Boltzmann equation.

        Note, we make extensive use of numpy's broadcasting rules.
        """

        particles = self.offEqParticles

        # coordinates
        xi, pz, pp = self.grid.getCoordinates()  # non-compact
        # adding new axes, to make everything rank 3 like deltaF, (z, pz, pp)
        # for fast multiplication of arrays, using numpy's broadcasting rules
        xi = xi[None, :, None, None]
        pz = pz[None, None, :, None]
        pp = pp[None, None, None, :]

        # compactified coordinates
        # chi, rz, rp = self.grid.getCompactCoordinates(endpoints=False)

        # background profiles
        temperatureFull = self.background.temperatureProfile
        vFull = self.background.velocityProfile
        msqFull = np.array(
            [
                particle.msqVacuum(self.background.fieldProfiles)
                for particle in particles
            ]
        )
        velocityWall = self.background.velocityWall

        # expanding to be rank 3 arrays, like deltaF
        temperature = self.background.temperatureProfile[None, 1:-1, None, None]
        v = vFull[None, 1:-1, None, None]
        msq = msqFull[:, 1:-1, None, None]
        energy = np.sqrt(msq + pz**2 + pp**2)

        # fluctuation mode
        statistics = np.array(
            [-1 if particle.statistics == "Fermion" else 1 for particle in particles]
        )[:, None, None, None]

        # building parts which depend on the 'derivatives' argument
        if self.derivatives == "Spectral":
            # fit the background profiles to polynomials
            temperaturePoly = Polynomial(
                temperatureFull,
                self.grid,
                "Cardinal",
                "z",
                True,
            )
            vPoly = Polynomial(vFull, self.grid, "Cardinal", "z", True)
            msqPoly = Polynomial(
                msqFull, self.grid, ("Array", "Cardinal"), ("Array", "z"), True
            )
            # intertwiner matrices
            intertwinerChiMat = temperaturePoly.matrix(self.basisM, "z")
            intertwinerRzMat = temperaturePoly.matrix(self.basisN, "pz")
            intertwinerRpMat = temperaturePoly.matrix(self.basisN, "pp")
            # derivative matrices
            derivMatrixChi = temperaturePoly.derivMatrix(self.basisM, "z")[1:-1]
            derivMatrixRz = temperaturePoly.derivMatrix(self.basisN, "pz")[1:-1]
            # spatial derivatives of profiles
            dTemperaturedChi = temperaturePoly.derivative(0).coefficients[
                None, 1:-1, None, None
            ]
            dvdChi = vPoly.derivative(0).coefficients[None, 1:-1, None, None]
            dMsqdChi = msqPoly.derivative(1).coefficients[:, 1:-1, None, None]
        else:  # self.derivatives == "Finite Difference"
            # intertwiner matrices are simply unit matrices
            # as we are in the (Cardinal, Cardinal) basis
            intertwinerChiMat = np.identity(self.grid.M - 1)
            intertwinerRzMat = np.identity(self.grid.N - 1)
            intertwinerRpMat = np.identity(self.grid.N - 1)
            # derivative matrices
            chiFull, rzFull, _ = self.grid.getCompactCoordinates(endpoints=True)
            derivOperatorChi = findiff.FinDiff((0, chiFull, 1), acc=2)
            derivMatrixChi = derivOperatorChi.matrix((self.grid.M + 1,))
            derivOperatorRz = findiff.FinDiff((0, rzFull, 1), acc=2)
            derivMatrixRz = derivOperatorRz.matrix((self.grid.N + 1,))
            # spatial derivatives of profiles, endpoints used for taking
            # derivatives but then dropped as deltaF fixed at 0 at endpoints
            dTemperaturedChi = (derivMatrixChi @ temperatureFull)[
                None, 1:-1, None, None
            ]
            dvdChi = (derivMatrixChi @ vFull)[None, 1:-1, None, None]
            # the following is equivalent to:
            # dMsqdChiEinsum = np.einsum(
            #   "ij,aj->ai", derivMatrixChi.toarray(), msqFull
            # )[:, 1:-1, None, None]
            dMsqdChi = np.sum(
                derivMatrixChi.toarray()[None, :, :] * msqFull[:, None, :],
                axis=-1,
            )[:, 1:-1, None, None]
            # restructuring derivative matrices to appropriate forms for
            # Liouville operator
            derivMatrixChi = derivMatrixChi.toarray()[1:-1, 1:-1]
            derivMatrixRz = derivMatrixRz.toarray()[1:-1, 1:-1]

        # dot products with wall velocity
        gammaWall = 1 / np.sqrt(1 - velocityWall**2)
        momentumWall = gammaWall * (pz - velocityWall * energy)

        # dot products with plasma profile velocity
        gammaPlasma = 1 / np.sqrt(1 - v**2)
        energyPlasma = gammaPlasma * (energy - v * pz)
        momentumPlasma = gammaPlasma * (pz - v * energy)

        # dot product of velocities
        uwBaruPl = gammaWall * gammaPlasma * (velocityWall - v)

        # (exact) derivatives of compactified coordinates
        dxidchi, dpzdrz, _ = self.grid.getCompactificationDerivatives()
        dchidxi = 1 / dxidchi[None, :, None, None]
        drzdpz = 1 / dpzdrz[None, None, :, None]

        # derivative of equilibrium distribution
        dfEq = BoltzmannSolver._dfeq(energyPlasma / temperature, statistics)

        ##### source term #####
        # Given by S_i on the RHS of Eq. (5) in 2204.13120, with further details
        # given in Eq. (6).
        source = (
            (dfEq / temperature)
            * dchidxi
            * (
                momentumWall * momentumPlasma * gammaPlasma**2 * dvdChi
                + momentumWall * energyPlasma * dTemperaturedChi / temperature
                + 1 / 2 * dMsqdChi * uwBaruPl
            )
        )

        ##### liouville operator #####
        # Given in the LHS of Eq. (5) in 2204.13120, with further details given
        # by the second line of Eq. (32).
        identityParticles = np.identity(len(particles))[
            :, None, None, None, :, None, None, None
        ]
        liouville = identityParticles * (
            dchidxi[:, :, :, :, None, None, None, None]
            * momentumWall[:, :, :, :, None, None, None, None]
            * derivMatrixChi[None, :, None, None, None, :, None, None]
            * intertwinerRzMat[None, None, :, None, None, None, :, None]
            * intertwinerRpMat[None, None, None, :, None, None, None, :]
            - dchidxi[:, :, :, :, None, None, None, None]
            * drzdpz[:, :, :, :, None, None, None, None]
            * (gammaWall / 2)
            * dMsqdChi[:, :, :, :, None, None, None, None]
            * intertwinerChiMat[None, :, None, None, None, :, None, None]
            * derivMatrixRz[None, None, :, None, None, None, :, None]
            * intertwinerRpMat[None, None, None, :, None, None, None, :]
        )
        """
        An alternative, but slower, implementation is given by the following:
        liouville = (
            np.einsum(
                "ijk, ia, jb, kc -> ijkabc",
                dchidxi * PWall,
                derivChi,
                TRzMat,
                TRpMat,
                optimize=True,
            )
            - np.einsum(
                "ijk, ia, jb, kc -> ijkabc",
                gammaWall / 2 * dchidxi * drzdpz * dmsqdChi,
                TChiMat,
                derivRz,
                TRpMat,
                optimize=True,
            )
        )
        """

        # including factored-out T^2 in collision integrals
        collision = self.collisionMultiplier * (
            (temperature**2)[:, :, :, :, None, None, None, None]
            * intertwinerChiMat[None, :, None, None, None, :, None, None]
            * self.collisionArray[:, None, :, :, :, None, :, :]
        )
        ##### total operator #####
        operator = liouville + collision

        # reshaping indices
        totalSize = (
            len(particles) * (self.grid.M - 1) * (self.grid.N - 1) * (self.grid.N - 1)
        )
        source = np.reshape(source, totalSize, order="C")
        operator = np.reshape(operator, (totalSize, totalSize), order="C")

        # returning results
        return operator, source, liouville, collision

    def loadCollisions(self, directoryPath: "pathlib.Path") -> None:
        """
        Loads collision files for use with the Boltzmann solver.

        Args:
            directoryPath (pathlib.Path): Directory containing the .hdf5 collision data.

        Returns:
            None

        Raises:
            CollisionLoadError
        """
        try:
            self.collisionArray = CollisionArray.newFromDirectory(
                directoryPath,
                self.grid,
                self.basisN,
                self.offEqParticles,
            )
            logging.debug("Loaded collision data from directory %s", directoryPath)
        except CollisionLoadError as e:
            raise

    @staticmethod
    def _checkBasis(basis: str) -> None:
        """
        Check that basis is recognised
        """
        bases = ["Cardinal", "Chebyshev"]
        assert basis in bases, f"BoltzmannSolver error: unkown basis {basis}"

    @staticmethod
    def _checkDerivatives(derivatives: str) -> None:
        """
        Check that derivative option is recognised
        """
        derivativesOptions = ["Spectral", "Finite Difference"]
        assert (
            derivatives in derivativesOptions
        ), f"BoltzmannSolver error: unkown derivatives option {derivatives}"

    @staticmethod
    def _feq(x: np.ndarray, statistics: int | np.ndarray) -> np.ndarray:
        """
        Thermal distribution functions, Bose-Einstein and Fermi-Dirac
        """
        x = np.asarray(x)
        return np.where(
            x > BoltzmannSolver.MAX_EXPONENT,
            0,
            1 / (np.exp(x) - statistics),
        )

    @staticmethod
    def _dfeq(x: np.ndarray, statistics: int | np.ndarray) -> np.ndarray:
        """
        Temperature derivative of thermal distribution functions
        """
        x = np.asarray(x)
        return np.where(
            x > BoltzmannSolver.MAX_EXPONENT,
            -0,
            -1 / (np.exp(x) - 2 * statistics + np.exp(-x)),
        )
    

class EWBGBoltzmannSolver:
    """
    Class for solving the Boltzmann equation in the context of electroweak baryogenesis.
    """

    # Static value holding of natural log of the maximum expressible float
    MAX_EXPONENT: typing.Final[float] = sys.float_info.max_exp * np.log(2)

    # Member variables
    grid: Grid
    offEqParticles: list[Particle]
    background: BoltzmannBackground
    collisionArray: CollisionArray
    truncationOption: ETruncationOption
    wallGoResults: WallGoResults
    helicities: np.ndarray
    etas: np.ndarray
    chargeBranches: np.ndarray
    kineticStates: np.ndarray

    def __init__(
        self,
        grid: Grid,
        basisM: str = "Cardinal",
        basisN: str = "Chebyshev",
        derivatives: str = "Spectral",
        collisionMultiplier: float = 1.0,
        truncationOption: ETruncationOption = ETruncationOption.AUTO,
        helicities: tuple[int, ...] = (-1, 1),
    ):
        """
        Initialisation of EWBGBoltzmannSolver

        Parameters
        ----------
        grid : Grid
            An object of the Grid class.
            integrals.
        basisM : str, optional
            The position polynomial basis type, either 'Cardinal' or 'Chebyshev'.
            Default is 'Cardinal'.
        basisN : str, optional
            The momentum polynomial basis type, either 'Cardinal' or 'Chebyshev'.
            Default is 'Chebyshev'.
        derivatives : {'Spectral', 'Finite Difference'}, optional
            Choice of method for computing derivatives. Default is 'Spectral'
            which is expected to be more accurate.
        collisionMultiplier : float, optional
            Factor by which the collision term is multiplied. Can be used for testing.
            Default is 1.0.
        truncationOption : ETruncationOption, optional
            Option for truncating the spectral expansion. Default is
            ETruncationOption.AUTO. Other options
            are ETruncationOption.NONE and ETruncationOption.THIRD.
        helicities : tuple[int, ...], optional
            Helicity eigenvalues to solve for. Each value must be either
            ``-1`` or ``1`` and values may not be repeated. The default solves
            both helicities in the order ``(-1, 1)``.

        Returns
        -------
        cls : EWBGBoltzmannSolver
            An object of the EWBGBoltzmannSolver class.
        """

        self.grid = grid
        EWBGBoltzmannSolver._checkDerivatives(derivatives)
        self.derivatives = derivatives
        EWBGBoltzmannSolver._checkBasis(basisM)
        EWBGBoltzmannSolver._checkBasis(basisN)
        if derivatives == "Finite Difference":
            assert (
                basisM == "Cardinal" and basisN == "Cardinal"
            ), "Must use Cardinal basis for Finite Difference method"

        # Position polynomial type
        self.basisM = basisM
        # Momentum polynomial type
        self.basisN = basisN

        self.collisionMultiplier = collisionMultiplier
        self.truncationOption = truncationOption
        if (
            not helicities
            or len(set(helicities)) != len(helicities)
            or any(helicity not in (-1, 1) for helicity in helicities)
        ):
            raise ValueError(
                "helicities must contain unique values chosen from (-1, 1)."
            )
        self.helicities = np.asarray(helicities, dtype=int)
        self.helicities.setflags(write=False)
        # eta = +1 denotes particles and eta = -1 antiparticles.  The
        # collision operator is currently CP symmetric and does not carry an
        # explicit eta axis, so charge-resolved distributions are reconstructed
        # from their CP-even and CP-odd components after solving.
        self.etas = np.asarray((1, -1), dtype=int)
        self.etas.setflags(write=False)
        self.chargeBranches = self.etas

        # These are set, and can be updated, by our member functions
        # TODO: are these None types the best way to go?
        self.background = None  # type: ignore[assignment]
        self.collisionArray = None  # type: ignore[assignment]
        self.offEqParticles = []
        self.usesChiralSpecies = False
        self.kineticStates = np.empty(
            (0, len(self.etas), len(self.helicities)), dtype=object
        )
        self.kineticStates.setflags(write=False)

    def getHelicityIndex(self, helicity: int) -> int:
        """Return the array index corresponding to helicity ``-1`` or ``1``."""
        matches = np.flatnonzero(self.helicities == helicity)
        if matches.size == 0:
            raise ValueError(
                f"Helicity {helicity} was not configured; available values are "
                f"{tuple(self.helicities)}."
            )
        return int(matches[0])

    def getEtaIndex(self, eta: int) -> int:
        """Return the array index for particle or antiparticle ``eta``."""
        matches = np.flatnonzero(self.etas == eta)
        if matches.size == 0:
            raise ValueError(
                f"Charge branch {eta} is not physical; available values are "
                f"{tuple(self.etas)}."
            )
        return int(matches[0])

    def getChargeIndex(self, eta: int) -> int:
        """Alias for :meth:`getEtaIndex`."""
        return self.getEtaIndex(eta)

    def getKineticState(
        self,
        particleIndex: int,
        eta: int,
        helicity: int,
    ) -> KineticState:
        r"""Return the state labelled by species index, :math:`\eta`, and helicity."""
        particleMatches = [
            index
            for index, particle in enumerate(self.offEqParticles)
            if particle.index == particleIndex
        ]
        if len(particleMatches) != 1:
            raise ValueError(
                f"Expected one particle with index {particleIndex}; "
                f"found {len(particleMatches)}."
            )
        particlePosition = particleMatches[0]
        etaPosition = self.getEtaIndex(eta)
        if self.usesChiralSpecies:
            state = self.kineticStates[particlePosition, etaPosition]
            if state.helicity != helicity:
                raise ValueError(
                    f"State {state.particle.name}, eta={eta} has physical helicity "
                    f"{state.helicity}, not {helicity}."
                )
            return state
        return self.kineticStates[
            particlePosition,
            etaPosition,
            self.getHelicityIndex(helicity),
        ]

    # this should be verified since we do not know the which frame is used in the wallGoResults
    def setBackground(self, background: BoltzmannBackground) -> None:
        """
        Setter for the BoltzmannBackground
        """
        self.background = deepcopy(
            background
        )  # do we need a deepcopy? Does this even work generally?
        self.background.boostToPlasmaFrame()

    def setCollisionArray(self, collisionArray: CollisionArray) -> None:
        """
        Setter for the CollisionArray
        """
        self.collisionArray = collisionArray

    def updateParticleList(self, offEqParticles: list[Particle]) -> None:
        """
        Setter for the list of out-of-equilibrium Particle objects
        """
        self.offEqParticles = offEqParticles
        self.usesChiralSpecies = any(
            isinstance(particle, ChiralParticle) for particle in offEqParticles
        )
        if self.usesChiralSpecies:
            invalidFermions = [
                particle.name
                for particle in offEqParticles
                if particle.statistics == "Fermion"
                and not isinstance(particle, ChiralParticle)
            ]
            if invalidFermions:
                raise TypeError(
                    "A chiral transport system requires every fermion to be a "
                    f"ChiralParticle; invalid species: {invalidFermions}."
                )
            kineticStates = np.empty(
                (len(self.offEqParticles), len(self.etas)), dtype=object
            )
            for particleIndex, particle in enumerate(self.offEqParticles):
                for etaIndex, eta in enumerate(self.etas):
                    helicity = (
                        int(eta * particle.chirality)
                        if isinstance(particle, ChiralParticle)
                        else 0
                    )
                    kineticStates[particleIndex, etaIndex] = KineticState(
                        particle, int(eta), helicity
                    )
            self.kineticStates = kineticStates
            self.kineticStates.setflags(write=False)
            return

        invalidParticles = [
            particle.name
            for particle in offEqParticles
            if not isinstance(particle, ComplexMassParticle)
        ]
        if invalidParticles:
            raise TypeError(
                "Non-chiral EWBG transport requires ComplexMassParticle objects; "
                f"invalid species: {invalidParticles}."
            )
        kineticStates = np.empty(
            (
                len(self.offEqParticles),
                len(self.etas),
                len(self.helicities),
            ),
            dtype=object,
        )
        for particleIndex, particle in enumerate(self.offEqParticles):
            for etaIndex, eta in enumerate(self.etas):
                for helicityIndex, helicity in enumerate(self.helicities):
                    kineticStates[particleIndex, etaIndex, helicityIndex] = (
                        KineticState(particle, int(eta), int(helicity))
                    )
        self.kineticStates = kineticStates
        self.kineticStates.setflags(write=False)

    def setWallGoResults(
        self,
        wallGoResults: WallGoResults,
    ) -> None:
        """
        Import a converged WallGo solution.

        The WallGo result supplies the scalar profiles, temperature profile,
        and plasma-velocity profile required by the EWBG Boltzmann equation.
        velocityMid (the average of the asymptotic plasma velocities, used for
        the boost to the plasma frame) is not stored on WallGoResults, so it
        must be supplied by the caller, e.g. via
        ``hydrodynamics.findHydroBoundaries(wallGoResults.wallVelocity)``.
        """
        if wallGoResults is None:
            raise ValueError("wallGoResults cannot be None.")

        self.wallGoResults = deepcopy(wallGoResults)

    def setBackground(self, velocityMid: float) -> None:
        if self.wallGoResults is None:
            raise ValueError("wallGoResults must be set before setting background.")

        background = BoltzmannBackground(
            velocityMid=velocityMid,
            velocityProfile=self.wallGoResults.velocityProfile,
            fieldProfiles=self.wallGoResults.fieldProfiles,
            temperatureProfile=self.wallGoResults.temperatureProfile,
            polynomialBasis=self.basisM, # this should be verified. Also, this is not from the WallGoResults.
        )

        bg = deepcopy(background)

        self.background = bg
        
        self.background.boostToPlasmaFrame()



    def getDeltas(
        self,
        deltaF: typing.Optional[np.ndarray] = None,
    ) -> BoltzmannResults:
        """
        Computes Deltas necessary for solving the Higgs equation of motion.

        These are defined in equation (15) of 2204.13120 [LC22]_.

        Parameters
        ----------
        deltaF : array_like, optional
            A five-dimensional legacy helicity distribution or chiral-species
            charge distribution, or a six-dimensional legacy distribution
            carrying both charge and helicity axes. The final three axes are
            always ``(z, pz, pp)``.

        Returns
        -------
        Deltas : BoltzmannDeltas
            The four moments defined in equation (15) of [LC22]_, plus the
            number-density moment ``Delta10``. The moments retain all
            non-momentum axes of ``deltaF``.
        """
        # checking if result pre-computed
        if deltaF is None:
            deltaF = self.solveBoltzmannEquations()
        if deltaF.ndim not in (5, 6):
            raise ValueError(
                "deltaF must have axes (particle, helicity, z, pz, pp) or "
                "(particle, eta, helicity, z, pz, pp)."
            )

        # checking spectral convergence
        deltaF, shapeTruncated, spectralPeaks = self.checkSpectralConvergence(deltaF)

        # getting (optimistic) estimate of truncation error
        truncationError = self.estimateTruncationError(
            deltaF, shapeTruncated
        )
        truncatedTail = (
            shapeTruncated[-3] != deltaF.shape[-3],
            shapeTruncated[-2] != deltaF.shape[-2],
            shapeTruncated[-1] != deltaF.shape[-1],
        )

        particles = self.offEqParticles
        arrayRank = deltaF.ndim - 3
        arrayBases = ("Array",) * arrayRank
        arrayDirections = ("Array",) * arrayRank

        # constructing Polynomial class from deltaF array
        deltaFPoly = Polynomial(
            deltaF,
            self.grid,
            arrayBases + (self.basisM, self.basisN, self.basisN),
            arrayDirections + ("z", "pz", "pp"),
            False,
        )
        deltaFPoly.changeBasis(
            arrayBases + ("Cardinal", "Cardinal", "Cardinal")
        )

        # Take all field-space points, but throw the boundary points away
        # TODO: LN: why throw away boundary points?
        field = self.background.fieldProfiles.takeSlice(
            1, -1, axis=self.background.fieldProfiles.overFieldPoints
        )

        # adding new axes, to make everything rank 3 like deltaF (z, pz, pp)
        # for fast multiplication of arrays, using numpy's broadcasting rules
        pz = self.grid.pzValues.reshape((1,) * arrayRank + (1, -1, 1))
        pp = self.grid.ppValues.reshape((1,) * arrayRank + (1, 1, -1))
        msqShape = (len(particles),) + (1,) * (arrayRank - 1) + (-1, 1, 1)
        msq = np.array(
            [particle.msqVacuum(field) for particle in particles]
        ).reshape(msqShape)
        # constructing energy with (z, pz, pp) axes
        energy = np.sqrt(msq + pz**2 + pp**2)

        _, dpzdrz, dppdrp = self.grid.getCompactificationDerivatives()
        dpzdrz = dpzdrz.reshape((1,) * arrayRank + (1, -1, 1))
        dppdrp = dppdrp.reshape((1,) * arrayRank + (1, 1, -1))

        # base integrand, for '00'
        integrand = dpzdrz * dppdrp * pp / (4 * np.pi**2 * energy)

        momentumAxes = (deltaF.ndim - 2, deltaF.ndim - 1)
        Delta00 = deltaFPoly.integrate(  # pylint: disable=invalid-name
            momentumAxes, integrand
        )
        Delta02 = deltaFPoly.integrate(  # pylint: disable=invalid-name
            momentumAxes, pz**2 * integrand
        )
        Delta20 = deltaFPoly.integrate(  # pylint: disable=invalid-name
            momentumAxes, energy**2 * integrand
        )
        Delta11 = deltaFPoly.integrate(  # pylint: disable=invalid-name
            momentumAxes, energy * pz * integrand
        )
        Delta10 = deltaFPoly.integrate(  # pylint: disable=invalid-name
            momentumAxes, energy * integrand
        )

        Deltas = BoltzmannDeltas(  # pylint: disable=invalid-name
            Delta00=Delta00,
            Delta02=Delta02,
            Delta20=Delta20,
            Delta11=Delta11,
            Delta10=Delta10,
        )

        # returning results
        return BoltzmannResults(
            deltaF=deltaF,
            Deltas=Deltas,
            truncationError=truncationError,
            truncatedTail=truncatedTail,
            spectralPeaks=spectralPeaks,
        )

    def solveBoltzmannEquations(
        self,
        sourceType: EWBGSourceType = EWBGSourceType.ODD,
    ) -> np.ndarray:
        r"""
        Solves Boltzmann equation for :math:`\delta f`, equation (32) of [LC22].

        The Boltzmann equations are linearised and expressed in a spectral expansion,
        so that they take the form

        .. math::
            \left(\mathcal{L}[\alpha,\beta,\gamma;i,j,k]\delta_{ab} + \bar T_i(\chi^{(\alpha)})\mathcal{C}_{ab}[\beta,\gamma; j,k] \right) \delta f^b_{ijk} = \mathcal{S}_a[\alpha,\beta,\gamma],

        where :math:`\mathcal{L}` is the Lioville operator, :math:`\mathcal{C}`
        is the collision operator, and :math:`\mathcal{S}` is the source.

        As regards the indicies,

            - :math:`\alpha, \beta, \gamma` denote points on the coordinate lattice :math:`\{\xi^{(\alpha)},p_{z}^{(\beta)},p_{\Vert}^{(\gamma)}\}`,

            - :math:`i, j, k` denote elements of the basis of spectral functions :math:`\{\bar{T}_i, \bar{T}_j, \tilde{T}_k\}`,

            - :math:`a, b` denote particle species.

        For more details see the WallGo paper.

        Parameters
        ----------
        sourceType : EWBGSourceType, optional
            Select ``EVEN``, ``ODD``, or their ``TOTAL``. The default is
            ``ODD`` so that the returned distribution remains the
            baryogenesis-relevant CP-odd perturbation.

        Returns
        -------
        delta_f : array_like
            For a legacy Dirac basis, axes are ``(particle, helicity, z, pz,
            pp)``. For a chiral-species basis, axes are ``(particle, eta, z,
            pz, pp)`` and physical helicity is fixed by the corresponding
            :class:`KineticState`.

        References
        ----------
        .. [LC22] B. Laurent and J. M. Cline, First principles determination
            of bubble wall velocity, Phys. Rev. D 106 (2022) no.2, 023501
            doi:10.1103/PhysRevD.106.023501
        """

        # contructing the various terms in the Boltzmann equation
        operator, source, _, _ = self.buildLinearEquations(sourceType)

        # solving the linear system: operator.deltaF = source
        deltaF = np.linalg.solve(operator, source)

        # np.linalg.solve treats the transport-branch index as a set of
        # right-hand sides. For a chiral species basis these are charge
        # branches; for the legacy Dirac basis they are helicities.
        branchCount = (
            len(self.etas) if self.usesChiralSpecies else len(self.helicities)
        )
        deltaFShape = (
            len(self.offEqParticles),
            self.grid.M - 1,
            self.grid.N - 1,
            self.grid.N - 1,
            branchCount,
        )
        deltaF = np.reshape(deltaF, deltaFShape, order="C")
        deltaF = np.moveaxis(deltaF, -1, 1)

        return deltaF

    def reconstructChargeBranches(
        self,
        deltaFEven: np.ndarray,
        deltaFOdd: np.ndarray,
    ) -> np.ndarray:
        r"""Construct :math:`\delta f_{\eta h}` from CP-even and CP-odd solutions.

        The returned axes are ``(particle, eta, helicity, z, pz, pp)`` and

        .. math::
            \delta f_{\eta h}
            = \delta f_h^{\mathrm{even}}+\eta\delta f_h^{\mathrm{odd}}.

        This reconstruction assumes a CP-symmetric collision operator.  The
        current collision tensor has no explicit charge or helicity axes.
        """
        if self.usesChiralSpecies:
            raise ValueError(
                "ChiralParticle species already carry explicit eta branches; "
                "reconstruction is only defined for a legacy Dirac basis."
            )
        if deltaFEven.shape != deltaFOdd.shape:
            raise ValueError(
                "CP-even and CP-odd distributions must have equal shapes."
            )
        if deltaFEven.ndim != 5:
            raise ValueError(
                "CP-sector distributions must have axes "
                "(particle, helicity, z, pz, pp)."
            )

        eta = self.etas.reshape(1, -1, 1, 1, 1, 1)
        return deltaFEven[:, None, ...] + eta * deltaFOdd[:, None, ...]

    def solveBoltzmannEquationsByCharge(
        self,
        sourceType: EWBGSourceType = EWBGSourceType.TOTAL,
    ) -> np.ndarray:
        r"""Solve the charge-resolved Boltzmann equation.

        For a legacy Dirac basis, ``EVEN`` duplicates the CP-even solution on
        both charge branches and ``ODD`` returns
        :math:`\eta\delta f_h^{\mathrm{odd}}`. For a chiral species the source
        sign is evaluated directly from :math:`\eta h=\chi`, with
        :math:`h=\eta\chi`.

        The current Liouville and collision operators are diagonal in the
        explicit transport branches, while the source is evaluated for each
        configured :class:`KineticState`. A legacy Dirac basis returns axes
        ``(particle, eta, helicity, z, pz, pp)``. A chiral-species basis
        returns ``(particle, eta, z, pz, pp)`` because helicity is fixed by
        chirality and charge.
        """
        operator, source, _, _ = self.buildLinearEquations(
            sourceType,
            resolveChargeBranches=True,
        )
        deltaF = np.linalg.solve(operator, source)
        if self.usesChiralSpecies:
            deltaFShape = (
                len(self.offEqParticles),
                self.grid.M - 1,
                self.grid.N - 1,
                self.grid.N - 1,
                len(self.etas),
            )
            deltaF = np.reshape(deltaF, deltaFShape, order="C")
            return np.moveaxis(deltaF, -1, 1)
        deltaFShape = (
            len(self.offEqParticles),
            self.grid.M - 1,
            self.grid.N - 1,
            self.grid.N - 1,
            len(self.etas),
            len(self.helicities),
        )
        deltaF = np.reshape(deltaF, deltaFShape, order="C")
        return np.moveaxis(deltaF, (-2, -1), (1, 2))

    def estimateTruncationError(
        self, deltaF: np.ndarray, shapeTruncated: tuple[int, ...]
    ) -> float:
        r"""
        Quick estimate of the polynomial truncation error using
        John Boyd's Rule-of-thumb-2: the last coefficient of a Chebyshev
        polynomial expansion is the same order-of-magnitude as the truncation
        error.

        Parameters
        ----------
        deltaF : array_like
            The helicity-resolved solution with axes
            ``(particle, helicity, z, pz, pp)``.

        Returns
        -------
        truncationError : float
            Estimate of the relative truncation error.
        """
        if deltaF.ndim not in (5, 6):
            raise ValueError("deltaF must be helicity- or charge-resolved.")
        arrayRank = deltaF.ndim - 3
        arrayBases = ("Array",) * arrayRank

        # constructing Polynomial
        basisTypes = arrayBases + (self.basisM, self.basisN, self.basisN)
        basisNames = arrayBases + ("z", "pz", "pp")
        deltaFPoly = Polynomial(deltaF, self.grid, basisTypes, basisNames, False)

        # sum(|deltaF|) as the norm
        deltaFPoly.changeBasis(
            arrayBases + ("Chebyshev", "Chebyshev", "Chebyshev")
        )
        truncatedSlices = tuple(slice(0, size) for size in shapeTruncated)
        deltaFTuncated = deltaFPoly.coefficients[truncatedSlices]
        deltaFSumAbs = np.sum(np.abs(deltaFTuncated), axis=(-3, -2, -1))

        # estimating truncation errors in each direction
        truncationErrors = []
        with np.errstate(divide="ignore", invalid="ignore"):
            for axis in (-3, -2, -1):
                boundary = np.take(deltaFTuncated, -1, axis=axis)
                numerator = np.sum(np.abs(boundary), axis=(-2, -1))
                truncationErrors.append(
                    np.divide(
                        numerator,
                        deltaFSumAbs,
                        out=np.zeros_like(numerator),
                        where=deltaFSumAbs != 0,
                    )
                )

        # estimating the total truncation error as the maximum of these three
        return max(float(np.max(error)) for error in truncationErrors)

    def checkSpectralConvergence(
        self, deltaF: np.ndarray
    ) -> tuple[np.ndarray, tuple[int, ...], tuple[int, int, int]]:
        """
        Check for spectral convergence.

        Fits to the exponential slope of the last 1/3 of coefficients in the
        Chebyshev basis, and truncates if they are increasing. Also returns the
        positions of the spectral peaks of the distribution in each dimension.

        Parameters
        ----------
        deltaF : array_like
            The helicity-resolved solution with axes
            ``(particle, helicity, z, pz, pp)``.

        Returns
        -------
        deltaFTruncated : np.ndarray
            Potentially truncated version of ``deltaF``, padded with zeros if
            truncated and therefore retaining the input shape.
        shapeTruncated : tuple[int, int, int, int, int]
            Shape of truncated array.
        spectralPeaks : tuple[int, int, int]
            Indices of the peaks in the (potentially truncated) spectral expansion.
        """
        # constructing Polynomial
        if deltaF.ndim not in (5, 6):
            raise ValueError("deltaF must be helicity- or charge-resolved.")
        arrayRank = deltaF.ndim - 3
        arrayBases = ("Array",) * arrayRank
        basisTypes = arrayBases + (self.basisM, self.basisN, self.basisN)
        basisNames = arrayBases + ("z", "pz", "pp")
        deltaFPoly = Polynomial(deltaF, self.grid, basisTypes, basisNames, False)
        truncatedShape = list(deltaF.shape)

        # changing to Chebyshev basis
        deltaFPoly.changeBasis(
            arrayBases + ("Chebyshev", "Chebyshev", "Chebyshev")
        )

        # looking at convergence of spectral expansion
        def spectralCoefficients(axis: int) -> np.ndarray:
            sumAxes = tuple(i for i in range(deltaF.ndim) if i != axis)
            return np.sum(np.abs(deltaFPoly.coefficients), axis=sumAxes)

        spectralCoeffsChi = spectralCoefficients(deltaF.ndim - 3)
        spectralCoeffsPz = spectralCoefficients(deltaF.ndim - 2)
        spectralCoeffsPp = spectralCoefficients(deltaF.ndim - 1)

        # how much to cut, if truncating
        cutSpatial = -((self.grid.M - 1) // 3)
        cutMomentum = -((self.grid.N - 1) // 3)

        # checking spectral convergence of spatial direction
        chiConvergenceTailInfo = SpectralConvergenceInfo(
            spectralCoeffsChi[cutSpatial:],
            # weightPower=0,
            offset=self.grid.M - 1 + cutSpatial,
        )

        # checking spectral convergence of pz direction
        pzConvergenceTailInfo = SpectralConvergenceInfo(
            spectralCoeffsPz[cutMomentum:],
            # weightPower=1,  # removed as max(pz) only grows as log(N)
            offset=self.grid.N - 1 + cutMomentum,
        )

        # checking spectral convergence of pp direction
        ppConvergenceTailInfo = SpectralConvergenceInfo(
            spectralCoeffsPp[cutMomentum:],
            # weightPower=2,  # removed as max(pp) only grows as log(N)
            offset=self.grid.N - 1 + cutMomentum,
        )

        allTailsConverging = (
            chiConvergenceTailInfo.apparentConvergence and
            pzConvergenceTailInfo.apparentConvergence and
            ppConvergenceTailInfo.apparentConvergence
        )

        # Deciding what to do based on truncationOption
        if self.truncationOption == ETruncationOption.AUTO:
            # if the slope is not definitely negative, we will truncate
            if not chiConvergenceTailInfo.apparentConvergence:
                slices = [slice(None)] * deltaF.ndim
                slices[-3] = slice(cutSpatial, None)
                deltaFPoly.coefficients[tuple(slices)] = 0
                truncatedShape[-3] = deltaF.shape[-3] + cutSpatial
            if not pzConvergenceTailInfo.apparentConvergence:
                slices = [slice(None)] * deltaF.ndim
                slices[-2] = slice(cutMomentum, None)
                deltaFPoly.coefficients[tuple(slices)] = 0
                truncatedShape[-2] = deltaF.shape[-2] + cutMomentum
            if not ppConvergenceTailInfo.apparentConvergence:
                slices = [slice(None)] * deltaF.ndim
                slices[-1] = slice(cutMomentum, None)
                deltaFPoly.coefficients[tuple(slices)] = 0
                truncatedShape[-1] = deltaF.shape[-1] + cutMomentum
        elif self.truncationOption == ETruncationOption.THIRD:
            # truncating regardless
            for axis, cut in zip((-3, -2, -1), (cutSpatial, cutMomentum, cutMomentum)):
                slices = [slice(None)] * deltaF.ndim
                slices[axis] = slice(cut, None)
                deltaFPoly.coefficients[tuple(slices)] = 0
            truncatedShape[-3:] = [
                deltaF.shape[-3] + cutSpatial,
                deltaF.shape[-2] + cutMomentum,
                deltaF.shape[-1] + cutMomentum,
            ]
            if allTailsConverging:
                logging.info(
                    "Tails of spectral expansions converging but truncated, consider changing truncation option."
                )
        else:
            # not truncating regardless
            if not allTailsConverging:
                logging.info(
                    "Tails of spectral expansions not converging, consider changing truncation option, or changing grid parameters."
                )

        # checking spectral convergence of z direction
        chiConvergenceInfo = SpectralConvergenceInfo(
            spectralCoeffsChi[:truncatedShape[-3]], weightPower=0
        )

        # checking spectral convergence of pz direction
        pzConvergenceInfo = SpectralConvergenceInfo(
            spectralCoeffsPz[:truncatedShape[-2]], weightPower=1
        )

        # checking spectral convergence of pp direction
        ppConvergenceInfo = SpectralConvergenceInfo(
            spectralCoeffsPp[:truncatedShape[-1]], weightPower=2
        )

        # putting together the spectral peaks
        spectralPeaks = (
            chiConvergenceInfo.spectralPeak,
            pzConvergenceInfo.spectralPeak,
            ppConvergenceInfo.spectralPeak,
        )

        if self.truncationOption == ETruncationOption.NONE:
            return deltaF, tuple(truncatedShape), spectralPeaks

        # changing back to original basis
        deltaFPoly.changeBasis(basisTypes)

        return deltaFPoly.coefficients, tuple(truncatedShape), spectralPeaks

    @staticmethod
    def _smoothTruncation(length: int, cut: int, sharp: float = 3) -> np.ndarray:
        """
        Internal function to smooth the truncation of the spectral expansion. """
        x = np.arange(length)
        return 1 / (1 + np.exp(sharp * (x - cut)))

    def checkLinearization(
        self, deltaF: typing.Optional[np.ndarray] = None
    ) -> tuple[float, float]:
        r"""
        Compute two criteria to verify the validity of the linearisation of the
        Boltzmann equation: :math:`\delta f/f_{eq}` and
        :math:`\delta f_2/(f_{eq}+\delta f)`, with :math:`\delta f_2` the first-order
        correction due to nonlinearities.
        To be valid, at least one of the two criteria must be small for each particle.

        Parameters
        ----------
        deltaF : array-like, optional
            Solution of the Boltzmann equation. The default is None.

        Returns
        -------
        deltaFCriterion : tuple
        collCriterion : tuple
            Criteria for the validity of the linearization.

        """
        if deltaF is None:
            deltaF = self.solveBoltzmannEquations()
        if deltaF.ndim == 6:
            criteria = [
                self.checkLinearization(deltaF[:, etaIndex])
                for etaIndex in range(len(self.etas))
            ]
            return tuple(np.max(criteria, axis=0))  # type: ignore[return-value]
        if deltaF.ndim != 5:
            raise ValueError("deltaF must be helicity- or charge-resolved.")

        particles = self.offEqParticles

        # constructing Polynomial class from deltaF array
        deltaFPoly = Polynomial(
            deltaF,
            self.grid,
            ("Array", "Array", self.basisM, self.basisN, self.basisN),
            ("Array", "Array", "z", "pz", "pp"),
            False,
        )
        deltaFPoly.changeBasis(
            ("Array", "Array", "Cardinal", "Cardinal", "Cardinal")
        )

        # Computing \delta f^2
        deltaFSqPoly = deltaFPoly * deltaFPoly
        deltaFSqPoly.changeBasis(
            ("Array", "Array", self.basisM, self.basisN, self.basisN)
        )

        operator, _, _, collision = self.buildLinearEquations()
        source = np.einsum(
            "aijkblmn,bhlmn->ahijk",
            collision,
            deltaFSqPoly.coefficients,
            optimize=True,
        )

        # Computing the correction from nonlinear terms
        totalSize = operator.shape[0]
        source = np.moveaxis(source, 1, -1)
        branchCount = (
            len(self.etas) if self.usesChiralSpecies else len(self.helicities)
        )
        source = np.reshape(source, (totalSize, branchCount), order="C")
        deltaNonlin = np.linalg.solve(operator, source)
        deltaNonlinShape = (
            len(self.offEqParticles),
            self.grid.M - 1,
            self.grid.N - 1,
            self.grid.N - 1,
            branchCount,
        )
        deltaNonlin = np.reshape(deltaNonlin, deltaNonlinShape, order="C")
        deltaNonlin = np.moveaxis(deltaNonlin, -1, 1)
        deltaNonlinPoly = Polynomial(
            coefficients=deltaNonlin,
            grid=self.grid,
            basis=("Array", "Array", self.basisM, self.basisN, self.basisN),
            direction=("Array", "Array", "z", "pz", "pp"),
            endpoints=False,
        )
        deltaNonlinPoly.changeBasis(
            ("Array", "Array", "Cardinal", "Cardinal", "Cardinal")
        )

        msqFull = np.array(
            [
                particle.msqVacuum(self.background.fieldProfiles)
                for particle in particles
            ]
        )

        msqPoly = Polynomial(
            msqFull,
            self.grid,
            ("Array", "Cardinal"),
            "z",
            True,
        )
        dmsqdChi = msqPoly.derivative(axis=1).coefficients[:, 1:-1, None, None]

        thetaFull = np.array(
            [
                particle.phase(self.background.fieldProfiles)
                if isinstance(particle, ComplexMassParticle)
                else np.zeros_like(particle.msqVacuum(self.background.fieldProfiles))
                for particle in particles
            ]
        )

        thetaPoly = Polynomial(
            thetaFull,
            self.grid,
            ("Array", "Cardinal"),
            "z",
            True,
        )

        dThetadChi = thetaPoly.derivative(1).coefficients[:, 1:-1, None, None]
        ddThetadChi2 = thetaPoly.derivative(1).derivative(1).coefficients[:, 1:-1, None, None]

        # adding new axes, to make everything rank 3 like deltaF (z, pz, pp)
        # for fast multiplication of arrays, using numpy's broadcasting rules
        pz = self.grid.pzValues[None, None, :, None]
        pp = self.grid.ppValues[None, None, None, :]
        msq = msqFull[:, 1:-1, None, None]
        # constructing energy with (z, pz, pp) axes
        energy = np.sqrt(msq + pz**2 + pp**2)

        temperature = self.background.temperatureProfile[None, 1:-1, None, None]
        statistics = np.array(
            [-1 if particle.statistics == "Fermion" else 1 for particle in particles]
        )[:, None, None, None]

        fEq = EWBGBoltzmannSolver._feq(energy / temperature, statistics)
        fEqPoly = Polynomial(
            fEq,
            self.grid,
            ("Array", "Cardinal", "Cardinal", "Cardinal"),
            ("z", "z", "pz", "pp"),
            False,
        )

        _, dpzdrz, dppdrp = self.grid.getCompactificationDerivatives()
        dpzdrz = dpzdrz[None, None, :, None]
        dppdrp = dppdrp[None, None, None, :]

        dofs = np.array([particle.totalDOFs for particle in particles])[
            :, None, None, None
        ]
        integrand = dofs * dmsqdChi * dpzdrz * dppdrp * pp / (4 * np.pi**2 * energy)
        # ``totalDOFs`` includes the two explicit transport branches: the two
        # helicities for a legacy Dirac particle or particle/antiparticle for
        # a chiral species. Each branch carries half of that multiplicity.
        integrandPerBranch = integrand[:, None, ...] / 2

        # Computing the pressure contributions of the equilibrium part, the linear
        # out-of-equilibrium part and the first-order correction due to nonlinearities.
        pressureEq = np.sum(fEqPoly.integrate((1, 2, 3), integrand).coefficients)
        pressureDeltaF = np.sum(
            deltaFPoly.integrate(
                (2, 3, 4), integrandPerBranch
            ).coefficients
        )
        pressureNonlin = np.sum(
            deltaNonlinPoly.integrate(
                (2, 3, 4), integrandPerBranch
            ).coefficients
        )

        # Computing the 2 linearisation criteria
        criterion1 = abs(pressureDeltaF / pressureEq)
        criterion2 = abs(pressureNonlin / (pressureEq + pressureDeltaF))

        return criterion1, criterion2

    def buildLinearEquations(
        self,
        sourceType: EWBGSourceType = EWBGSourceType.ODD,
        resolveChargeBranches: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Construct the matrix and selected source for the Boltzmann equation.

        Note, we make extensive use of numpy's broadcasting rules.

        Parameters
        ----------
        sourceType : EWBGSourceType, optional
            Select the CP-even, CP-odd, or total source. The default is the
            CP-odd source used for baryogenesis.
        resolveChargeBranches : bool, optional
            Construct the source for every explicit :class:`KineticState`.
            For a legacy Dirac basis, right-hand sides are ordered by
            ``(eta, helicity)``. A chiral-species basis always uses one
            right-hand side per ``eta`` branch.
        """
        if not isinstance(sourceType, EWBGSourceType):
            raise TypeError("sourceType must be an EWBGSourceType.")

        particles = self.offEqParticles

        # coordinates
        xi, pz, pp = self.grid.getCoordinates()  # non-compact
        # adding new axes, to make everything rank 3 like deltaF, (z, pz, pp)
        # for fast multiplication of arrays, using numpy's broadcasting rules
        xi = xi[None, :, None, None]
        pz = pz[None, None, :, None]
        pp = pp[None, None, None, :]

        # compactified coordinates
        # chi, rz, rp = self.grid.getCompactCoordinates(endpoints=False)

        # background profiles
        #
        temperatureFull = self.background.temperatureProfile
        vFull = self.background.velocityProfile
        msqFull = np.array(
            [
                particle.msqVacuum(self.background.fieldProfiles)
                for particle in particles
            ]
        )

        thetaFull = np.array(
            [
                particle.phase(self.background.fieldProfiles)
                if isinstance(particle, ComplexMassParticle)
                else np.zeros_like(particle.msqVacuum(self.background.fieldProfiles))
                for particle in particles
            ]
        )

        velocityWall = self.background.velocityWall

        # expanding to be rank 3 arrays, like deltaF
        temperature = self.background.temperatureProfile[None, 1:-1, None, None]
        v = vFull[None, 1:-1, None, None]
        msq = msqFull[:, 1:-1, None, None]
        theta = thetaFull[:, 1:-1, None, None]
        energy = np.sqrt(msq + pz**2 + pp**2)
        energyZ = np.sqrt(msq + pz**2)

        # fluctuation mode
        statistics = np.array(
            [-1 if particle.statistics == "Fermion" else 1 for particle in particles]
        )[:, None, None, None]

        # building parts which depend on the 'derivatives' argument
        if self.derivatives == "Spectral":
            # fit the background profiles to polynomials
            temperaturePoly = Polynomial(
                temperatureFull,
                self.grid,
                "Cardinal",
                "z",
                True,
            )
            vPoly = Polynomial(vFull, self.grid, "Cardinal", "z", True)
            msqPoly = Polynomial(
                msqFull, self.grid, ("Array", "Cardinal"), ("Array", "z"), True
            )

            #
            thetaPoly = Polynomial(
                thetaFull, self.grid, ("Array", "Cardinal"), "z", True
            )

            # intertwiner matrices
            intertwinerChiMat = temperaturePoly.matrix(self.basisM, "z")
            intertwinerRzMat = temperaturePoly.matrix(self.basisN, "pz")
            intertwinerRpMat = temperaturePoly.matrix(self.basisN, "pp")
            # derivative matrices
            derivMatrixChi = temperaturePoly.derivMatrix(self.basisM, "z")[1:-1]
            derivMatrixRz = temperaturePoly.derivMatrix(self.basisN, "pz")[1:-1]
            # spatial derivatives of profiles
            dTemperaturedChi = temperaturePoly.derivative(0).coefficients[
                None, 1:-1, None, None
            ]
            dvdChi = vPoly.derivative(0).coefficients[None, 1:-1, None, None]
            dMsqdChi = msqPoly.derivative(1).coefficients[:, 1:-1, None, None]
            dThetadChi = thetaPoly.derivative(1).coefficients[:, 1:-1, None, None]
            ddThetadChi2 = thetaPoly.derivative(1).derivative(1).coefficients[:, 1:-1, None, None]

        else:  # self.derivatives == "Finite Difference"
            # intertwiner matrices are simply unit matrices
            # as we are in the (Cardinal, Cardinal) basis
            intertwinerChiMat = np.identity(self.grid.M - 1)
            intertwinerRzMat = np.identity(self.grid.N - 1)
            intertwinerRpMat = np.identity(self.grid.N - 1)
            # derivative matrices
            chiFull, rzFull, _ = self.grid.getCompactCoordinates(endpoints=True)
            derivOperatorChi = findiff.FinDiff((0, chiFull, 1), acc=2)
            derivOperatorChi2 = findiff.FinDiff((0, chiFull, 2), acc=2) # second order derivative for CPV source term
            derivMatrixChi = derivOperatorChi.matrix((self.grid.M + 1,))
            derivMatrixChi2 = derivOperatorChi2.matrix((self.grid.M + 1,))
            derivOperatorRz = findiff.FinDiff((0, rzFull, 1), acc=2)
            derivMatrixRz = derivOperatorRz.matrix((self.grid.N + 1,))
            # spatial derivatives of profiles, endpoints used for taking
            # derivatives but then dropped as deltaF fixed at 0 at endpoints
            dTemperaturedChi = (derivMatrixChi @ temperatureFull)[
                None, 1:-1, None, None
            ]
            dvdChi = (derivMatrixChi @ vFull)[None, 1:-1, None, None]
            # the following is equivalent to:
            # dMsqdChiEinsum = np.einsum(
            #   "ij,aj->ai", derivMatrixChi.toarray(), msqFull
            # )[:, 1:-1, None, None]
            dMsqdChi = np.sum(
                derivMatrixChi.toarray()[None, :, :] * msqFull[:, None, :],
                axis=-1,
            )[:, 1:-1, None, None]
            dThetadChi = np.sum(
                derivMatrixChi.toarray()[None, :, :] * thetaFull[:, None, :], axis=-1
            )[:, 1:-1, None, None]
            ddThetadChi2 = np.sum(
                derivMatrixChi2.toarray()[None, :, :] * thetaFull[:, None, :], axis=-1
            )[:, 1:-1, None, None]
            # restructuring derivative matrices to appropriate forms for
            # Liouville operator
            derivMatrixChi = derivMatrixChi.toarray()[1:-1, 1:-1]
            derivMatrixRz = derivMatrixRz.toarray()[1:-1, 1:-1]
            derivMatrixChi2 = derivMatrixChi2.toarray()[1:-1, 1:-1]

        # dot products with wall velocity
        gammaWall = 1 / np.sqrt(1 - velocityWall**2)
        momentumWall = gammaWall * (pz - velocityWall * energy)

        # dot products with plasma profile velocity
        gammaPlasma = 1 / np.sqrt(1 - v**2)
        energyPlasma = gammaPlasma * (energy - v * pz)
        momentumPlasma = gammaPlasma * (pz - v * energy)

        # dot product of velocities
        uwBaruPl = gammaWall * gammaPlasma * (velocityWall - v)

        # (exact) derivatives of compactified coordinates
        dxidchi, dpzdrz, _ = self.grid.getCompactificationDerivatives()
        dchidxi = 1 / dxidchi[None, :, None, None]
        drzdpz = 1 / dpzdrz[None, None, :, None]

        d2xidchi2, d2rz2, d2rp2 = self.grid.getCompactificationSecondDerivatives()
        d2xidchi2xx = d2xidchi2[None, :, None, None] # naming is bit weird, so need to be fixed
        
        # derivative of equilibrium distribution
        dfEq = EWBGBoltzmannSolver._dfeq(energyPlasma / temperature, statistics)
        d2feq = EWBGBoltzmannSolver._d2feq(energyPlasma / temperature, statistics)

        ##### CP-even source term #####
        # This is -(P_w d_xi + F_even d_p) f_eq. It is the same source used
        # by BoltzmannSolver for wall friction and is independent of helicity
        # at the order retained here.
        sourceEvenBase = (
            (dfEq / temperature)
            * dchidxi
            * (
                momentumWall * momentumPlasma * gammaPlasma**2 * dvdChi
                + momentumWall * energyPlasma * dTemperaturedChi / temperature
                + 0.5 * dMsqdChi * uwBaruPl
            )
        )

        ##### CP-odd source term #####

        gammaParallel = energy / energyZ
        sp = gammaParallel * pz / np.sqrt(pz**2 + pp**2)

        cpGradient = (
            dThetadChi * dMsqdChi * dchidxi**2
            + ddThetadChi2 * msq * dchidxi**2
            - msq * dchidxi**3 * d2xidchi2xx * dThetadChi
        )
        forceOdd = 0.5 * sp * cpGradient / energyZ
        deltaEnergy = 0.5 * sp * dThetadChi * msq * dchidxi / energyZ / energy
        ddeltaEnergydxi = 0.5 * sp * cpGradient / energyZ / energy

        sourceOddBase = -dfEq * gammaPlasma * v / temperature * forceOdd
        sourceOddBase = sourceOddBase - momentumWall * d2feq * (
            -(
                momentumPlasma * gammaPlasma**2 * dvdChi
                + energyPlasma * dTemperaturedChi / temperature
            )
            / temperature
        ) * dchidxi * gammaPlasma / temperature * deltaEnergy
        sourceOddBase = sourceOddBase - momentumWall * dfEq * (
            gammaPlasma**3 * v * dvdChi * dchidxi / temperature
            - gammaPlasma * dTemperaturedChi * dchidxi / temperature**2
        ) * deltaEnergy
        sourceOddBase = (
            sourceOddBase
            - momentumWall
            * dfEq
            * gammaPlasma
            / temperature
            * ddeltaEnergydxi
        )
        sourceOddBase = (
            sourceOddBase
            - 0.5
            * dMsqdChi
            * dchidxi
            * d2feq
            * gammaPlasma**2
            * v
            / temperature**2
            * deltaEnergy
        )

        if self.usesChiralSpecies:
            if self.kineticStates.shape != (len(particles), len(self.etas)):
                raise RuntimeError(
                    "Kinetic states are inconsistent with the chiral particle list."
                )
            cpSigns = np.asarray(
                [state.cpSign for state in self.kineticStates.flat],
                dtype=int,
            ).reshape(self.kineticStates.shape)
            sourceEven = np.broadcast_to(
                sourceEvenBase[:, None, ...],
                (
                    len(particles),
                    len(self.etas),
                    *sourceEvenBase.shape[1:],
                ),
            )
            sourceOdd = (
                sourceOddBase[:, None, ...]
                * cpSigns[:, :, None, None, None]
            )
        elif resolveChargeBranches:
            if self.kineticStates.shape != (
                len(particles), len(self.etas), len(self.helicities)
            ):
                raise RuntimeError(
                    "Kinetic states are inconsistent with the particle list."
                )
            cpSigns = np.asarray(
                [state.cpSign for state in self.kineticStates.flat],
                dtype=int,
            ).reshape(self.kineticStates.shape)
            sourceEven = np.broadcast_to(
                sourceEvenBase[:, None, None, ...],
                (
                    len(particles),
                    len(self.etas),
                    len(self.helicities),
                    *sourceEvenBase.shape[1:],
                ),
            )
            sourceOdd = (
                sourceOddBase[:, None, None, ...]
                * cpSigns[:, :, :, None, None, None]
            )
        else:
            sourceEven = np.repeat(
                sourceEvenBase[:, None, ...], len(self.helicities), axis=1
            )
            sourceOdd = (
                sourceOddBase[:, None, ...]
                * self.helicities[None, :, None, None, None]
            )

        if sourceType == EWBGSourceType.EVEN:
            source = sourceEven
        elif sourceType == EWBGSourceType.ODD:
            source = sourceOdd
        else:
            source = sourceEven + sourceOdd

        ##### liouville operator #####
        # Given in the LHS of Eq. (5) in 2204.13120, with further details given
        # by the second line of Eq. (32).
        identityParticles = np.identity(len(particles))[
            :, None, None, None, :, None, None, None
        ]
        liouville = identityParticles * (
            dchidxi[:, :, :, :, None, None, None, None]
            * momentumWall[:, :, :, :, None, None, None, None]
            * derivMatrixChi[None, :, None, None, None, :, None, None]
            * intertwinerRzMat[None, None, :, None, None, None, :, None]
            * intertwinerRpMat[None, None, None, :, None, None, None, :]
            - dchidxi[:, :, :, :, None, None, None, None]
            * drzdpz[:, :, :, :, None, None, None, None]
            * (gammaWall / 2)
            * dMsqdChi[:, :, :, :, None, None, None, None]
            * intertwinerChiMat[None, :, None, None, None, :, None, None]
            * derivMatrixRz[None, None, :, None, None, None, :, None]
            * intertwinerRpMat[None, None, None, :, None, None, None, :]
        )
        """
        An alternative, but slower, implementation is given by the following:
        liouville = (
            np.einsum(
                "ijk, ia, jb, kc -> ijkabc",
                dchidxi * PWall,
                derivChi,
                TRzMat,
                TRpMat,
                optimize=True,
            )
            - np.einsum(
                "ijk, ia, jb, kc -> ijkabc",
                gammaWall / 2 * dchidxi * drzdpz * dmsqdChi,
                TChiMat,
                derivRz,
                TRpMat,
                optimize=True,
            )
        )
        """

        # including factored-out T^2 in collision integrals
        collision = self.collisionMultiplier * (
            (temperature**2)[:, :, :, :, None, None, None, None]
            * intertwinerChiMat[None, :, None, None, None, :, None, None]
            * self.collisionArray[:, None, :, :, :, None, :, :]
        )
        ##### total operator #####
        operator = liouville + collision

        # reshaping indices
        totalSize = (
            len(particles) * (self.grid.M - 1) * (self.grid.N - 1) * (self.grid.N - 1)
        )
        if self.usesChiralSpecies:
            source = np.moveaxis(source, 1, -1)
            source = np.reshape(source, (totalSize, len(self.etas)), order="C")
        elif resolveChargeBranches:
            source = np.moveaxis(source, (1, 2), (-2, -1))
            source = np.reshape(
                source,
                (totalSize, len(self.etas) * len(self.helicities)),
                order="C",
            )
        else:
            source = np.moveaxis(source, 1, -1)
            source = np.reshape(
                source, (totalSize, len(self.helicities)), order="C"
            )
        operator = np.reshape(operator, (totalSize, totalSize), order="C")

        # returning results
        return operator, source, liouville, collision

    def loadCollisions(self, directoryPath: "pathlib.Path") -> None:
        """
        Loads collision files for use with the Boltzmann solver.

        Args:
            directoryPath (pathlib.Path): Directory containing the .hdf5 collision data.

        Returns:
            None

        Raises:
            CollisionLoadError
        """
        try:
            self.collisionArray = CollisionArray.newFromDirectory(
                directoryPath,
                self.grid,
                self.basisN,
                self.offEqParticles,
            )
            logging.debug("Loaded collision data from directory %s", directoryPath)
        except CollisionLoadError as e:
            raise

    @staticmethod
    def _checkBasis(basis: str) -> None:
        """
        Check that basis is recognised
        """
        bases = ["Cardinal", "Chebyshev"]
        assert basis in bases, f"EWBGBoltzmannSolver error: unkown basis {basis}"

    @staticmethod
    def _checkDerivatives(derivatives: str) -> None:
        """
        Check that derivative option is recognised
        """
        derivativesOptions = ["Spectral", "Finite Difference"]
        assert (
            derivatives in derivativesOptions
        ), f"EWBGBoltzmannSolver error: unkown derivatives option {derivatives}"

    @staticmethod
    def _feq(x: np.ndarray, statistics: int | np.ndarray) -> np.ndarray:
        """
        Thermal distribution functions, Bose-Einstein and Fermi-Dirac
        """
        x = np.asarray(x)
        return np.where(
            x > EWBGBoltzmannSolver.MAX_EXPONENT,
            0,
            1 / (np.exp(x) - statistics),
        )

    @staticmethod
    def _dfeq(x: np.ndarray, statistics: int | np.ndarray) -> np.ndarray:
        """
        Temperature derivative of thermal distribution functions
        """
        x = np.asarray(x)
        return np.where(
            x > EWBGBoltzmannSolver.MAX_EXPONENT,
            -0,
            -1 / (np.exp(x) - 2 * statistics + np.exp(-x)),
        )
    
    @staticmethod
    def _d2feq(x: np.ndarray, statistics: int | np.ndarray) -> np.ndarray:
        """
        Second temperature derivative of thermal distribution functions
        """
        x = np.asarray(x)
        exp_x = np.exp(x)
        return np.where(
            x > EWBGBoltzmannSolver.MAX_EXPONENT,
            0,
            exp_x * (exp_x + statistics) / (exp_x - statistics) ** 3,
        )
