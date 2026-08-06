r"""
This Python script, singletStandardModelZ2.py,
implements a minimal Standard Model extension via
a scalar singlet and incorporating a Z2 symmetry.
TopL, TopR, BotL, and Higgs are out of equilibrium. The singlet and gauge
bosons remain in the equilibrium bath.

Introducing the dimension-5 operator which gives cp violation in the top quark sector

.. math::
    \mathcal{L} \supset i \frac{y_t}{\Lambda} s \bar{t}_L h t_R + h.c.

Features:
- Definition of the extended model parameters including the singlet scalar field.
- Definition of the out-of-equilibrium particles.
- Implementation of the one-loop thermal potential, without high-T expansion.

Usage:
- This script is intended to compute the wall speed of the model.

Dependencies:
- NumPy for numerical calculations
- the WallGo package
- CollisionIntegrals in read-only mode using the default path for the collision
integrals as the "CollisonOutput" directory

Note:
This benchmark model was used to compare against the results of
B. Laurent and J. M. Cline, First principles determination
of bubble wall velocity, Phys. Rev. D 106 (2022) no.2, 023501
doi:10.1103/PhysRevD.106.023501
As a consequence, we overwrite the default WallGo thermal functions
Jb/Jf.
"""

import os
import sys
import pathlib
import argparse
import logging
from typing import TYPE_CHECKING
import numpy as np

# WallGo imports
import WallGo  # Whole package, in particular we get WallGo._initializeInternal()
from WallGo import Fields, GenericModel, Particle
from WallGo.particle import ChiralParticle
from WallGo.interpolatableFunction import EExtrapolationType

from WallGo.PotentialTools import EffectivePotentialNoResum, EImaginaryOption

# Add the Models folder to the path; need to import the base example
# template
modelsBaseDir = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(modelsBaseDir))

from wallGoExampleBase import WallGoExampleBase  # pylint: disable=C0411, C0413, E0401
from wallGoExampleBase import ExampleInputPoint  # pylint: disable=C0411, C0413, E0401

from SingletStandardModel_Z2.singletStandardModelZ2 import (
    EffectivePotentialxSMZ2,
    SingletSMZ2,
)

if TYPE_CHECKING:
    import WallGoCollision


class SingletSM_CPVdim5(SingletSMZ2):
    """
    Z2 symmetric SM + singlet model. V = muHsq |phi|^2 + lHH (|phi|^2)^2 + 1/2 muSsq S^2 + 1/4 lSS S^4 + 1/2 lHS |phi|^2 S^2
    """

    def __init__(self, allowOutOfEquilibriumGluon=False):

        self.modelParameters = {}

        self.effectivePotential = EffectivePotentialCPVdim5(self)

        self.defineParticles(allowOutOfEquilibriumGluon)
        self.bIsGluonOffEq = allowOutOfEquilibriumGluon

    def defineParticles(self, includeGluon):
        self.clearParticles()

        if includeGluon:
            raise ValueError(
                "This EWBG setup keeps gluons in equilibrium; "
                "includeGluon must be False."
            )

        def topMsqVacuum(fields):
            h = fields.getField(0)
            s = fields.getField(1)

            yt = self.modelParameters["yt"]
            Lam = self.modelParameters["Lambda"]

            return 0.5 * (h**2) * yt**2 * (1 + (s / Lam) ** 2)

        def topMsqDerivative(fields):
            h = fields.getField(0)
            s = fields.getField(1)

            yt = self.modelParameters["yt"]
            Lam = self.modelParameters["Lambda"]

            return np.transpose(
                [
                    0.5 * yt**2 * (1 + (s / Lam) ** 2) * 2 * h,
                    0.5 * yt**2 * h**2 * 2 * s / Lam**2,
                ]
            )

        def topMCPPhase(fields):
            h = fields.getField(0)
            s = fields.getField(1)

            Lam = self.modelParameters["Lambda"]

            return np.arctan2(s, Lam)

        topQuarkL = ChiralParticle(
            "TopL",
            index=0,
            msqVacuum=topMsqVacuum,
            msqDerivative=topMsqDerivative,
            phase=topMCPPhase,
            chirality=-1,
            totalDOFs=6,
        )

        topQuarkR = ChiralParticle(
            "TopR",
            index=1,
            msqVacuum=topMsqVacuum,
            msqDerivative=topMsqDerivative,
            phase=topMCPPhase,
            chirality=1,
            totalDOFs=6,
        )

        def masslessMsqVacuum(fields):
            return np.zeros_like(fields.getField(0))

        def masslessMsqDerivative(fields):
            return np.zeros_like(fields)

        def zeroPhase(fields):
            return np.zeros_like(fields.getField(0))

        bottomQuarkL = ChiralParticle(
            "BotL",
            index=2,
            msqVacuum=masslessMsqVacuum,
            msqDerivative=masslessMsqDerivative,
            phase=zeroPhase,
            chirality=-1,
            totalDOFs=6,
        )

        def higgsMsqVacuum(fields):
            h = fields.getField(0)
            s = fields.getField(1)
            return (
                self.modelParameters["muHsq"]
                + 3 * self.modelParameters["lHH"] * h**2
                + 0.5 * self.modelParameters["lHS"] * s**2
            )

        def higgsMsqDerivative(fields):
            h = fields.getField(0)
            s = fields.getField(1)
            return np.transpose(
                [
                    6 * self.modelParameters["lHH"] * h,
                    self.modelParameters["lHS"] * s,
                ]
            )

        higgs = Particle(
            "Higgs",
            index=3,
            msqVacuum=higgsMsqVacuum,
            msqDerivative=higgsMsqDerivative,
            statistics="Boson",
            totalDOFs=4,
        )

        self.addParticle(topQuarkL)
        self.addParticle(topQuarkR)
        self.addParticle(bottomQuarkL)
        self.addParticle(higgs)

    def calculateLagrangianParameters(self, inputParameters):
        params = super().calculateLagrangianParameters(inputParameters)
        params["Lambda"] = inputParameters["Lambda"]
        return params


class EffectivePotentialCPVdim5(EffectivePotentialxSMZ2):

    def fermionInformation(self, fields):

        v = fields.getField(0)
        x = fields.getField(1)

        yt = self.modelParameters["yt"]
        Lam = self.modelParameters["Lambda"]

        mtsq = 0.5 * v**2 * yt**2 * (1 + x**2 / Lam**2)

        massSq = np.stack((mtsq,), axis=-1)
        degreesOfFreedom = np.array([12])
        c = np.array([3 / 2])
        rgScale = np.array([self.modelParameters["RGScale"]])

        return massSq, degreesOfFreedom, c, rgScale


class SingletSMZ2_EWBG(WallGoExampleBase):

    def __init__(self) -> None:

        self.bShouldRecalculateCollisions = False

        self.bShouldRecalculateMatrixElements = False

        self.matrixElementFile = pathlib.Path(
            self.exampleBaseDirectory
            / "MatrixElements/matrixElements.ewbg_chiral.json"
        )
        self.matrixElementInput = pathlib.Path(
            self.exampleBaseDirectory / "MatrixElements/model.m"
        )

        # ~ Begin WallGoExampleBase interface

    def initCommandLineArgs(self) -> argparse.ArgumentParser:
        """Non-abstract override to add a SM + singlet specific command line option"""

        argParser: argparse.ArgumentParser = super().initCommandLineArgs()
        argParser.add_argument(
            "--outOfEquilibriumGluon",
            help="Treat the SU(3) gluons as out-of-equilibrium particle species",
            action="store_true",
        )
        return argParser

    def initWallGoModel(self) -> "WallGo.GenericModel":
        """
        Initialize the model. This should run after cmdline argument parsing
        so safe to use them here.
        """
        return SingletSM_CPVdim5(self.cmdArgs.outOfEquilibriumGluon)

    def initCollisionModel(
        self, wallGoModel: "SingletSM_CPVdim5"
    ) -> "WallGoCollision.PhysicsModel":
        """Initialize the Collision model and set the seed."""

        import WallGoCollision  # pylint: disable = C0415

        # Collision integrations utilize Monte Carlo methods, so RNG is involved.
        # We can set the global seed for collision integrals as follows.
        # This is optional; by default the seed is 0.
        WallGoCollision.setSeed(0)

        # This example comes with a very explicit example function on how to setup and
        # configure the collision module. It is located in a separate module
        # (same directory) to avoid bloating this file. Import and use it here.
        from exampleCollisionDefs import setupCollisionModel_EWBG  # pylint: disable=C0415

        collisionModel = setupCollisionModel_EWBG(wallGoModel.modelParameters)

        return collisionModel

    def updateCollisionModel(
        self,
        inWallGoModel: "SingletSM_CPVdim5",
        inOutCollisionModel: "WallGoCollision.PhysicsModel",
    ) -> None:
        """Propagate couplings and thermal masses to the collision model."""
        import WallGoCollision  # pylint: disable = C0415

        changedParams = WallGoCollision.ModelParameters()

        gs = inWallGoModel.modelParameters["g3"]
        gw = inWallGoModel.modelParameters["g2"]
        yt = inWallGoModel.modelParameters["yt"]
        lHH = inWallGoModel.modelParameters["lHH"]
        lHS = inWallGoModel.modelParameters["lHS"]
        lSS = inWallGoModel.modelParameters["lSS"]

        changedParams.addOrModifyParameter("gs", gs)
        changedParams.addOrModifyParameter("gw", gw)
        changedParams.addOrModifyParameter("yt", yt)
        changedParams.addOrModifyParameter("lHH", lHH)
        changedParams.addOrModifyParameter("lHS", lHS)
        changedParams.addOrModifyParameter("lSS", lSS)
        changedParams.addOrModifyParameter("mq2", gs**2 / 6.0)
        changedParams.addOrModifyParameter("mg2", 2.0 * gs**2)
        changedParams.addOrModifyParameter("mW2", 3.0 * gw**2 / 5.0)
        changedParams.addOrModifyParameter(
            "mH2",
            3.0 * gw**2 / 16.0 + lHH / 2.0 + yt**2 / 4.0 + lHS / 24.0,
        )
        changedParams.addOrModifyParameter("mS2", lHS / 6.0 + lSS / 4.0)

        inOutCollisionModel.updateParameters(changedParams)

    def configureCollisionIntegration(
        self, inOutCollisionTensor: "WallGoCollision.CollisionTensor"
    ) -> None:
        """Non-abstract override"""

        import WallGoCollision  # pylint: disable = C0415

        """Configure the integrator. Default settings should be reasonably OK so you
        can modify only what you need, or skip this step entirely. Here we set
        everything manually to show how it's done.
        """
        integrationOptions = WallGoCollision.IntegrationOptions()
        integrationOptions.calls = 50000
        integrationOptions.maxTries = 50
        # collision integration momentum goes from 0 to maxIntegrationMomentum.
        # This is in units of temperature
        integrationOptions.maxIntegrationMomentum = 20
        integrationOptions.absoluteErrorGoal = 1e-8
        integrationOptions.relativeErrorGoal = 1e-1

        inOutCollisionTensor.setIntegrationOptions(integrationOptions)

        """We can also configure various verbosity settings that are useful when
        you want to see what is going on in long-running integrations. These 
        include progress reporting and time estimates, as well as a full result dump
        of each individual integral to stdout. By default these are all disabled. 
        Here we enable some for demonstration purposes.
        """
        verbosity = WallGoCollision.CollisionTensorVerbosity()
        verbosity.bPrintElapsedTime = (
            True  # report total time when finished with all integrals
        )

        """Progress report when this percentage of total integrals (approximately)
        have been computed. Note that this percentage is per-particle-pair, ie. 
        each (particle1, particle2) pair reports when this percentage of their
        own integrals is done. Note also that in multithreaded runs the 
        progress tracking is less precise.
        """
        verbosity.progressReportPercentage = 0.25

        # Print every integral result to stdout? This is very slow and
        # verbose, intended only for debugging purposes
        verbosity.bPrintEveryElement = False

        inOutCollisionTensor.setIntegrationVerbosity(verbosity)

    def configureManager(self, inOutManager: "WallGo.WallGoManager") -> None:
        """We load the configs from a file for this example."""
        inOutManager.config.loadConfigFromFile(
            pathlib.Path(self.exampleBaseDirectory / "singletStandardModelZ2Config.ini")
        )
        super().configureManager(inOutManager)

    def updateModelParameters(
        self, model: "SingletSM_CPVdim5", inputParameters: dict[str, float]
    ) -> None:
        """Convert SM + singlet inputs to Lagrangian params and update internal
        model parameters. This example is constructed so that the effective
        potential and particle mass functions refer to model.modelParameters,
        so be careful not to replace that reference here.
        """

        # oldParams = model.modelParameters.copy()

        model.updateModel(inputParameters)

        """Collisions integrals for this example depend only on the QCD coupling,
        if it changes we must recompute collisions before running the wall solver.
        The bool flag here is inherited from WallGoExampleBase and temperatureed
        in runExample(). But since we want to keep the example simple, we skip
        this check and assume the existing data is OK.
        (FIXME?)
        """
        self.bShouldRecalculateCollisions = False

        """
        newParams = model.modelParameters
        if not oldParams or newParams["g3"] != oldParams["g3"]:
            self.bNeedsNewCollisions = True
        """

    def getBenchmarkPoints(self) -> list[ExampleInputPoint]:
        """
        Input parameters, phase info, and settings for the effective potential and
        wall solver for the xSM benchmark point.
        """
        output: list[ExampleInputPoint] = []
        output.append(
            ExampleInputPoint(
                {
                    "RGScale": 125.0,
                    "v0": 246.0,
                    "MW": 80.379,
                    "MZ": 91.1876,
                    "Mt": 173.0,
                    "g3": 1.2279920495357861,
                    "mh1": 125.0,
                    "mh2": 120.0,
                    "lHS": 0.9,
                    "lSS": 1.0,
                    "Lambda": 1000.0,
                },
                WallGo.PhaseInfo(
                    temperature=100.0,  # nucleation temperature
                    phaseLocation1=WallGo.Fields([0.0, 200.0]),
                    phaseLocation2=WallGo.Fields([246.0, 0.0]),
                ),
                WallGo.VeffDerivativeSettings(
                    temperatureVariationScale=10.0,
                    fieldValueVariationScale=[10.0, 10.0],
                ),
                WallGo.WallSolverSettings(
                    # we actually do both cases in the common example
                    bIncludeOffEquilibrium=True,
                    meanFreePathScale=50.0,  # In units of 1/Tnucl
                    wallThicknessGuess=5.0,  # In units of 1/Tnucl
                ),
            )
        )

        return output

    # ~ End WallGoExampleBase interface


def initCommandLineArgs() -> argparse.ArgumentParser:
    """Command line arguments for main()."""
    argParser = argparse.ArgumentParser()

    argParser.add_argument(
        "--momentumGridSize",
        type=int,
        default=0,
        help="""Basis size N override for the momentum grid. Values less than
        or equal to 0 are ignored and we fall back to the default (N=11),
        including for the collision data directory name.""",
    )

    argParser.add_argument(
        "-v",
        "--verbose",
        type=int,
        default=logging.DEBUG,
        help="""Set the verbosity level. Must be an int: DEBUG=10, INFO=20,
        WARNING=30, ERROR=40. Default is DEBUG.""",
    )

    argParser.add_argument(
        "--recalculateCollisions",
        action="store_true",
        help="""Forces full recalculation of the relevant collision integrals
        instead of loading the provided data files for this example. This is
        very slow and disabled by default. The resulting collision data will
        be written to a directory labeled _UserGenerated; the default
        provided data will not be overwritten.""",
    )

    return argParser


# main function to run the example
def main():

    argParser = initCommandLineArgs()
    cmdArgs = argParser.parse_args()

    momentumGridSize = cmdArgs.momentumGridSize if cmdArgs.momentumGridSize > 0 else 11

    manager = WallGo.EWBGWallGoManager()
    manager.setVerbosity(cmdArgs.verbose)

    # Change the amount of grid points in the spatial coordinates
    # for faster computations
    manager.config.configGrid.spatialGridSize = 20
    if cmdArgs.momentumGridSize > 0:
        manager.config.configGrid.momentumGridSize = cmdArgs.momentumGridSize
    # Increase the number of iterations in the wall solving to
    # ensure convergence
    manager.config.configEOM.maxIterations = 25
    # Decrease error tolerance for phase tracing to ensure stability
    manager.config.configThermodynamics.phaseTracerTol = 1e-8

    model = SingletSM_CPVdim5(allowOutOfEquilibriumGluon=False)
    manager.registerModel(model)

    inputParameters = {
                    "RGScale": 125.0,
                    "v0": 246.0,
                    "MW": 80.379,
                    "MZ": 91.1876,
                    "Mt": 173.0,
                    "g3": 1.2279920495357861,
                    "mh1": 125.0,
                    "mh2": 120.0,
                    "lHS": 0.9,
                    "lSS": 1.0,
                    "Lambda": 1000.0,
    }

    model.updateModel(inputParameters)

    exampleHelper = SingletSMZ2_EWBG()

    if cmdArgs.recalculateCollisions:
        newCollisionDir = pathlib.Path(__file__).resolve().parent / (
            f"CollisionOutput_N{momentumGridSize}_UserGenerated"
        )
        manager.setPathToCollisionData(newCollisionDir)

        collisionModel = exampleHelper.initCollisionModel(model)

        if not collisionModel.loadMatrixElements(
            str(exampleHelper.matrixElementFile), True
        ):
            print("FATAL: Failed to load matrix elements")
            return exit(1)

        collisionTensor = collisionModel.createCollisionTensor(momentumGridSize)
        exampleHelper.configureCollisionIntegration(collisionTensor)

        print(
            "Entering collision integral computation, this may take long",
            flush=True,
        )
        collisionResults = collisionTensor.computeIntegralsAll()
        collisionResults.writeToIndividualHDF5(
            str(manager.getCurrentCollisionDirectory())
        )
    else:
        exampleDir = pathlib.Path(__file__).resolve().parent
        pathtoCollisions = exampleDir / f"CollisionOutput_N{momentumGridSize}"
        pathtoUserGeneratedCollisions = exampleDir / (
            f"CollisionOutput_N{momentumGridSize}_UserGenerated"
        )

        if not pathtoCollisions.exists() and pathtoUserGeneratedCollisions.exists():
            pathtoCollisions = pathtoUserGeneratedCollisions

        if not pathtoCollisions.exists():
            print(
                f"Collision data not found at {pathtoCollisions} or "
                f"{pathtoUserGeneratedCollisions}. Please run with "
                "--recalculateCollisions, or check --momentumGridSize."
            )
            return exit(1)

        manager.setPathToCollisionData(pathtoCollisions)

    manager.setupThermodynamicsHydrodynamics(
        WallGo.PhaseInfo(
            temperature=100.0,  # nucleation temperature
            phaseLocation1=WallGo.Fields([0.0, 200.0]),
            phaseLocation2=WallGo.Fields([246.0, 0.0]),
        ),
        WallGo.VeffDerivativeSettings(
            temperatureVariationScale=10.0,
            fieldValueVariationScale=[10.0, 10.0],
        ),
 
    )

    # ---- Solve wall speed in Local Thermal Equilibrium (LTE) approximation
    vwLTE = manager.wallSpeedLTE()
    print(f"LTE wall speed:    {vwLTE:.6f}")

    solverSettings = WallGo.WallSolverSettings(
        bIncludeOffEquilibrium=True,
        meanFreePathScale=50.0,  # In units of 1/Tnucl
        wallThicknessGuess=5.0,  # In units of 1/Tnucl
    )


    solver: WallGo.WallSolver = manager.setupWallSolver(solverSettings)

 
    results = solver.eom.findWallVelocityDeflagrationHybrid(
        solver.initialWallThickness
    )

    print(
        f"Wall velocity without out-of-equilibrium contributions {results.wallVelocity:.6f}"
    )
    
    # _, _, _, _, velocityMid = manager.hydrodynamics.findHydroBoundaries(
    #     results.wallVelocity
    # )

    # velocityProfile = results.velocityProfile
    # temperatureProfile = results.temperatureProfile
    # wallVelocity = results.wallVelocity
    # fieldProfile = results.fieldProfiles
    # spatialcoordinatePolynomialbasis = solver.boltzmannSolver.basisM

    # background : BoltzmannBackground = BoltzmannBackground(        
    #     velocityMid, 
    #     velocityProfile, 
    #     temperatureProfile, 
    #     fieldProfile, 
    #     spatialcoordinatePolynomialbasis
    # )

    print(
        f"Wall velocity with out-of-equilibrium contributions {results.wallVelocity:.6f}"
    )

    # EWBG input
    ewbgSolver = manager.setupEWBGSolver(solver, results)
    deltaF = manager.solveBoltzmannEWBG(
        ewbgSolver,
        sourceType=WallGo.EWBGSourceType.ODD,
        resolveChargeBranches=True,
    )
    boltzmannResults = manager.getDeltas(ewbgSolver, deltaF)

    z = ewbgSolver.grid.xiValues
    delta10 = boltzmannResults.Deltas.Delta10
    assert delta10 is not None

    import matplotlib.pyplot as plt

    transportSolver = ewbgSolver.EWBGBoltzmannSolver
    _, sourceEvenFlat, _, _ = transportSolver.buildLinearEquations(
        WallGo.EWBGSourceType.EVEN,
        resolveChargeBranches=True,
    )
    _, sourceOddFlat, _, _ = transportSolver.buildLinearEquations(
        WallGo.EWBGSourceType.ODD,
        resolveChargeBranches=True,
    )

    sourceGridShape = (
        len(transportSolver.offEqParticles),
        ewbgSolver.grid.M - 1,
        ewbgSolver.grid.N - 1,
        ewbgSolver.grid.N - 1,
        len(transportSolver.etas),
    )
    sourceEven = np.moveaxis(
        np.reshape(sourceEvenFlat, sourceGridShape, order="C"), -1, 1
    )
    sourceOdd = np.moveaxis(
        np.reshape(sourceOddFlat, sourceGridShape, order="C"), -1, 1
    )

    topLPosition = next(
        index
        for index, particle in enumerate(transportSolver.offEqParticles)
        if particle.name == "TopL"
    )
    particleBranchIndex = transportSolver.getEtaIndex(1)
    _, pzIndex, ppIndex = np.unravel_index(
        np.argmax(np.abs(sourceOdd[topLPosition, particleBranchIndex])),
        sourceOdd[topLPosition, particleBranchIndex].shape,
    )
    pzValue = ewbgSolver.grid.pzValues[pzIndex]
    ppValue = ewbgSolver.grid.ppValues[ppIndex]

    sourceFigure, sourceAxes = plt.subplots(
        1, 2, figsize=(11, 4), sharex=True
    )
    for state in transportSolver.kineticStates[topLPosition].flat:
        chargeIndex = transportSolver.getEtaIndex(state.eta)
        sourceAxes[0].plot(
            z,
            sourceEven[topLPosition, chargeIndex, :, pzIndex, ppIndex],
            label=rf"$\eta={state.eta:+d},\ h={state.helicity:+d}$",
        )
        sourceAxes[1].plot(
            z,
            sourceOdd[topLPosition, chargeIndex, :, pzIndex, ppIndex],
            label=rf"$\eta={state.eta:+d},\ h={state.helicity:+d}$",
        )

    sourceAxes[0].set_title("CP-even source")
    sourceAxes[1].set_title("CP-odd source")
    for sourceAxis in sourceAxes:
        sourceAxis.set_xlabel(r"$z$")
        sourceAxis.set_ylabel(
            r"$S_h(z,p_z^\star,p_\parallel^\star)$"
        )
        sourceAxis.grid(alpha=0.3)
        sourceAxis.legend()
    sourceFigure.suptitle(
        rf"TopL sources at $p_z={pzValue:.3g}$, "
        rf"$p_\parallel={ppValue:.3g}$"
    )
    sourceFigure.tight_layout()

    deltaFigure, deltaAxes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for particlePosition, deltaAxis in enumerate(deltaAxes.flat):
        particle = transportSolver.offEqParticles[particlePosition]
        for state in transportSolver.kineticStates[particlePosition].flat:
            chargeIndex = transportSolver.getEtaIndex(state.eta)
            deltaAxis.plot(
                z,
                delta10.coefficients[particlePosition, chargeIndex],
                label=rf"$\eta={state.eta:+d},\ h={state.helicity:+d}$",
            )
        deltaAxis.set_xlabel(r"$z$")
        deltaAxis.set_ylabel(r"$\Delta_{10,\eta}$")
        deltaAxis.set_title(particle.name)
        deltaAxis.grid(alpha=0.3)
        deltaAxis.legend()
    deltaFigure.suptitle("Charge-resolved density moments")
    deltaFigure.tight_layout()

    plt.show()

if __name__ == "__main__":
    main()
