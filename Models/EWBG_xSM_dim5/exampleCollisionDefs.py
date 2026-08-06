"""Collision-model definitions for the chiral singlet-SM EWBG setup."""

import WallGoCollision


def setupCollisionModel_EWBG(
    modelParameters: dict[str, float],
) -> WallGoCollision.PhysicsModel:
    """Build the collision model matching ``MatrixElements/model.m``.

    TopL, TopR, BotL, and Higgs carry independent perturbations. Singlet,
    Gluon, and W are equilibrium bath species.
    """

    modelDefinition = WallGoCollision.ModelDefinition()
    parameters = WallGoCollision.ModelParameters()

    parameters.addOrModifyParameter("gs", modelParameters["g3"])
    parameters.addOrModifyParameter("gw", modelParameters["g2"])
    parameters.addOrModifyParameter("yt", modelParameters["yt"])
    parameters.addOrModifyParameter("lHH", modelParameters["lHH"])
    parameters.addOrModifyParameter("lHS", modelParameters["lHS"])
    parameters.addOrModifyParameter("lSS", modelParameters["lSS"])

    def quarkThermalMassSquared(
        p: WallGoCollision.ModelParameters,
    ) -> float:
        return p["gs"] ** 2 / 6.0

    def gluonThermalMassSquared(
        p: WallGoCollision.ModelParameters,
    ) -> float:
        return 2.0 * p["gs"] ** 2

    def wThermalMassSquared(
        p: WallGoCollision.ModelParameters,
    ) -> float:
        return 3.0 * p["gw"] ** 2 / 5.0

    def higgsThermalMassSquared(
        p: WallGoCollision.ModelParameters,
    ) -> float:
        return (
            3.0 * p["gw"] ** 2 / 16.0
            + p["lHH"] / 2.0
            + p["yt"] ** 2 / 4.0
            + p["lHS"] / 24.0
        )

    def singletThermalMassSquared(
        p: WallGoCollision.ModelParameters,
    ) -> float:
        return p["lHS"] / 6.0 + p["lSS"] / 4.0

    parameters.addOrModifyParameter("mq2", quarkThermalMassSquared(parameters))
    parameters.addOrModifyParameter("mg2", gluonThermalMassSquared(parameters))
    parameters.addOrModifyParameter("mW2", wThermalMassSquared(parameters))
    parameters.addOrModifyParameter("mH2", higgsThermalMassSquared(parameters))
    parameters.addOrModifyParameter("mS2", singletThermalMassSquared(parameters))
    modelDefinition.defineParameters(parameters)

    def defineParticle(
        name: str,
        index: int,
        particleType: WallGoCollision.EParticleType,
        inEquilibrium: bool,
        massSqFunction,
    ) -> None:
        particle = WallGoCollision.ParticleDescription()
        particle.name = name
        particle.index = index
        particle.type = particleType
        particle.bInEquilibrium = inEquilibrium
        particle.bUltrarelativistic = True
        particle.massSqFunction = massSqFunction
        modelDefinition.defineParticleSpecies(particle)

    fermion = WallGoCollision.EParticleType.eFermion
    boson = WallGoCollision.EParticleType.eBoson

    defineParticle("TopL", 0, fermion, False, quarkThermalMassSquared)
    defineParticle("TopR", 1, fermion, False, quarkThermalMassSquared)
    defineParticle("BotL", 2, fermion, False, quarkThermalMassSquared)
    defineParticle("Higgs", 3, boson, False, higgsThermalMassSquared)
    defineParticle("Singlet", 4, boson, True, singletThermalMassSquared)
    defineParticle("Gluon", 5, boson, True, gluonThermalMassSquared)
    defineParticle("W", 6, boson, True, wThermalMassSquared)

    return WallGoCollision.PhysicsModel(modelDefinition)
