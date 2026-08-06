"""
Module with Particle class to hold particle information
"""

from __future__ import annotations

from dataclasses import dataclass
import typing
import numpy as np
from .fields import Fields, FieldPoint


class Particle:  # pylint: disable=too-few-public-methods
    """Particle configuration

    A simple class holding attributes of an out-of-equilibrium particle as
    relevant for calculations of Boltzmann equations.
    """

    STATISTICS_OPTIONS: typing.Final[list[str]] = ["Fermion", "Boson"]

    def __init__(
        self,
        name: str,
        index: int,
        msqVacuum: typing.Callable[[Fields | FieldPoint], np.ndarray],
        msqDerivative: typing.Callable[[Fields | FieldPoint], np.ndarray],
        statistics: str,
        totalDOFs: int,
    ) -> None:
        r"""Initialisation

        Parameters
        ----------
        name : string
            A string naming the particle species.
        index : int
            Integer identifier for the particle species. Must be unique
            and match the intended particle index in matrix elements.
        msqVacuum : function
            Function :math:`m^2_0(\phi)`, should take a Fields or FieldPoint object and
            return an array of length Fields.NumPoints(). The background field dependent
            but temperature independent part of the effective mass squared.
        msqDerivative : function
            Function :math:`d(m_0^2)/d(\phi)`, should take a Fields or FieldPoints
            object and return an array of shape Fields.shape.
        statistics : {\"Fermion\", \"Boson\"}
            Particle statistics.
        totalDOFs : int
            Total number of degrees of freedom (should include the multiplicity
            factor).


        Returns
        -------
        cls : Particle
            An object of the Particle class.
        """
        Particle._validateInput(
            name,
            index,
            msqVacuum,
            msqDerivative,
            statistics,
            totalDOFs,
        )
        self.name = name
        self.index = index
        self.msqVacuum = msqVacuum
        self.msqDerivative = msqDerivative
        self.statistics = statistics
        self.totalDOFs = totalDOFs

    @staticmethod
    def _validateInput(  # pylint: disable=unused-argument
        name: str,
        index: int,
        msqVacuum: typing.Callable[[Fields], np.ndarray],
        msqDerivative: typing.Callable[[Fields], np.ndarray],
        statistics: str,
        totalDOFs: int,
    ) -> None:
        """
        Checks that the input fits expectations
        """
        # fields = np.array([1, 1])
        # assert isinstance(msqVacuum(fields), float), \
        #    f"msqVacuum({fields}) must return float"

        if statistics not in Particle.STATISTICS_OPTIONS:
            raise ValueError(f"{statistics=} not in {Particle.STATISTICS_OPTIONS}")
        assert isinstance(totalDOFs, int), "totalDOFs must be an integer"
        assert isinstance(index, int), "index must be an integer"

class ComplexMassParticle(Particle):
    def __init__(self, 
        name: str,
        index: int,
        msqVacuum: typing.Callable[[Fields | FieldPoint], np.ndarray],
        msqDerivative: typing.Callable[[Fields | FieldPoint], np.ndarray],
        phase: typing.Callable[[Fields | FieldPoint], np.ndarray],
        statistics: str,
        totalDOFs: int,
    ) -> None:
        r"""Initialisation
        
        Parameters
        ----------
        name : string
            A string naming the particle species.
        index : int
            Integer identifier for the particle species. Must be unique
            and match the intended particle index in matrix elements.
        msqVacuum : function
            Function :math:`m^2_0(\phi)`, should take a Fields or FieldPoint object and
            return an array of length Fields.NumPoints(). The background field dependent
            but temperature independent part of the effective mass squared.
        msqDerivative : function
            Function :math:`d(m_0^2)/d(\phi)`, should take a Fields or FieldPoints
            object and return an array of shape Fields.shape.
        phase : function
            Function :math: `\theta(\phi)`, should take a Fields or FieldPoints
            object and return an array of shape Fields.shape.
        statistics : {\"Fermion\", \"Boson\"}
            Particle statistics.
        totalDOFs : int
            Total number of degrees of freedom (should include the multiplicity
            factor.
        
        Returns
        -------
        cls : ComplexMassParticle
            An object of the ComplexMassParticle class.
    
        """

        ComplexMassParticle._validateInput(
            name,
            index,
            msqVacuum,
            msqDerivative,
            phase,
            statistics,
            totalDOFs,
        )

        super().__init__(
            name,
            index,
            msqVacuum,
            msqDerivative,
            statistics,
            totalDOFs
        )

        self.phase = phase

    @staticmethod
    def _validateInput(  # pylint: disable=unused-argument
        name: str,
        index: int,
        msqVacuum: typing.Callable[[Fields], np.ndarray],
        msqDerivative: typing.Callable[[Fields], np.ndarray],
        phase: typing.Callable[[Fields], np.ndarray],
        statistics: str,
        totalDOFs: int,
    ) -> None:
        """
        Checks that the input fits expectations
        """

        Particle._validateInput(
            name,
            index,
            msqVacuum,
            msqDerivative,
            statistics,
            totalDOFs,
        )

        if not callable(phase):
            raise TypeError("phase must be callable")


class ChiralParticle(ComplexMassParticle):
    r"""A Weyl species with a fixed chirality and complex Dirac-mass phase.

    ``chirality=-1`` denotes a left-handed species and ``chirality=+1`` a
    right-handed species. For a charge branch ``eta``, the physical helicity
    is ``helicity = eta * chirality`` in the ultrarelativistic limit.
    """

    def __init__(
        self,
        name: str,
        index: int,
        msqVacuum: typing.Callable[[Fields | FieldPoint], np.ndarray],
        msqDerivative: typing.Callable[[Fields | FieldPoint], np.ndarray],
        phase: typing.Callable[[Fields | FieldPoint], np.ndarray],
        chirality: int,
        totalDOFs: int,
    ) -> None:
        if chirality not in (-1, 1):
            raise ValueError("chirality must be -1 (left) or +1 (right).")
        self.chirality = chirality
        super().__init__(
            name=name,
            index=index,
            msqVacuum=msqVacuum,
            msqDerivative=msqDerivative,
            phase=phase,
            statistics="Fermion",
            totalDOFs=totalDOFs,
        )


@dataclass(frozen=True)
class KineticState:
    r"""A charge- and helicity-resolved transport state.

    For ``ChiralParticle`` objects, helicity is fixed by
    :math:`h=\eta\chi`. A bosonic state uses ``helicity=0``.
    """

    particle: Particle
    eta: int
    helicity: int

    def __post_init__(self) -> None:
        if self.eta not in (-1, 1):
            raise ValueError("eta must be +1 (particle) or -1 (antiparticle).")
        if isinstance(self.particle, ChiralParticle):
            expectedHelicity = self.eta * self.particle.chirality
            if self.helicity != expectedHelicity:
                raise ValueError(
                    "helicity must equal eta * chirality for a chiral particle."
                )
        elif isinstance(self.particle, ComplexMassParticle):
            if self.helicity not in (-1, 1):
                raise ValueError("helicity must be either -1 or +1.")
        elif self.particle.statistics == "Boson":
            if self.helicity != 0:
                raise ValueError("a bosonic kinetic state must have helicity 0.")
        else:
            raise TypeError(
                "fermionic kinetic states require ComplexMassParticle or "
                "ChiralParticle."
            )
        if self.helicity not in (-1, 0, 1):
            raise ValueError("helicity must be -1, 0, or +1.")

    @property
    def cpSign(self) -> int:
        r"""Return the branch sign :math:`\eta h` of the CP-odd source."""
        return self.eta * self.helicity

    @property
    def isParticle(self) -> bool:
        """Whether this is the particle rather than antiparticle branch."""
        return self.eta == 1

    def phase(self, fields: Fields | FieldPoint) -> np.ndarray:
        r"""Return the charge-conjugated phase :math:`\theta_\eta=\eta\theta`."""
        if isinstance(self.particle, ComplexMassParticle):
            return self.eta * self.particle.phase(fields)
        return np.zeros_like(self.particle.msqVacuum(fields))
