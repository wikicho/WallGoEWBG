(* ::Package:: *)

(*
Minimal WallGoMatrix model for electroweak-baryogenesis collisions in the
Z2-symmetric real-singlet extension of the Standard Model.

Field content:
    Q3L = (tL, bL), tR, H, S, gluons, and SU(2) gauge bosons.

TopL and TopR are interaction-basis labels used to generate chiral matrix
elements. They are mapped downstream to separate ChiralParticle species.

The dimension-five CP-violating singlet-top interaction is not included in
the collision amplitudes. It enters the complex top mass and semiclassical
source through ChiralParticle.
*)


If[$InputFileName == "",
    SetDirectory[NotebookDirectory[]],
    SetDirectory[DirectoryName[$InputFileName]]
];

WallGo`WallGoMatrix`$GroupMathMultipleModels = True;
WallGo`WallGoMatrix`$LoadGroupMath = True;

Check[
    Get["WallGo`WallGoMatrix`"],
    Message[
        Get::noopen,
        "WallGo`WallGoMatrix` at " <> ToString[$UserBaseDirectory] <> "/Applications"
    ];
    Abort[];
];


(* ::Chapter:: *)
(*Minimal QCD + weak + top-Yukawa + singlet model*)


(* ::Section:: *)
(*Gauge and matter representations*)


Group = {"SU3", "SU2"};
RepAdjoint = {{1, 1}, {2}};
CouplingName = {gs, gw};


(* One third-generation quark doublet and one right-handed top. *)
RepQ3L = {{{1, 0}, {1}}, "L"};
RepTopR = {{{1, 0}, {0}}, "R"};
RepFermion = {RepQ3L, RepTopR};


HiggsDoublet = {{{0, 0}, {1}}, "C"};
RealSinglet = {{{0, 0}, {0}}, "R"};
RepScalar = {HiggsDoublet, RealSinglet};


{
    gvvv,
    gvff,
    gvss,
    \[Lambda]1,
    \[Lambda]3,
    \[Lambda]4,
    \[Mu]ij,
    \[Mu]IJ,
    \[Mu]IJC,
    Ysff,
    YsffC
} = AllocateTensors[
    Group,
    RepAdjoint,
    CouplingName,
    RepFermion,
    RepScalar
];


(* ::Section:: *)
(*Z2-symmetric scalar potential*)


HiggsMassInvariant = CreateInvariant[
    Group,
    RepScalar,
    {{1, 1}, {True, False}}
][[1]] // Simplify // FullSimplify;

SingletMassInvariant = CreateInvariant[
    Group,
    RepScalar,
    {{2, 2}, {True, True}}
][[1]] // Simplify // FullSimplify;


(*
V = muHsq (H^\[Dagger] H) + muSsq S^2/2
    + lHH (H^\[Dagger] H)^2 + lSS S^4/4
    + lHS (H^\[Dagger] H) S^2/2.
*)
ScalarMassPotential =
    muHsq HiggsMassInvariant
    + muSsq SingletMassInvariant/2;

ScalarQuarticPotential =
    lHH HiggsMassInvariant^2
    + lSS SingletMassInvariant^2/4
    + lHS HiggsMassInvariant SingletMassInvariant/2;

\[Mu]ij = GradMass[ScalarMassPotential] // Simplify // SparseArray;
\[Lambda]4 = GradQuartic[ScalarQuarticPotential];

(* Z2 symmetry forbids scalar tadpoles and cubic interactions. *)
\[Lambda]1 = 0 \[Lambda]1;
\[Lambda]3 = 0 \[Lambda]3;


(* ::Section:: *)
(*Top Yukawa interaction*)


TopYukawaInvariant = CreateInvariantYukawa[
    Group,
    RepScalar,
    RepFermion,
    {{1, 1, 2}, {False, False, True}}
] // Simplify;

Ysff = -yt GradYukawa[TopYukawaInvariant[[1]]];
YsffC = SparseArray[
    Simplify[
        Conjugate[Ysff] // Normal,
        Assumptions -> {yt > 0}
    ]
];


ImportModel[
    Group,
    gvvv,
    gvff,
    gvss,
    \[Lambda]1,
    \[Lambda]3,
    \[Lambda]4,
    \[Mu]ij,
    \[Mu]IJ,
    \[Mu]IJC,
    Ysff,
    YsffC,
    Verbose -> False
];


(* ::Section:: *)
(*Interaction-basis particles*)


(*
The singlet background is set to zero in the collision amplitudes. Its
wall-dependent value remains part of the ChiralParticle source sector.
*)
vev = {0, v, 0, 0, 0};
SymmetryBreaking[vev];


(* Chiral components of Q3L and tR. *)
RepTopLParticle = CreateParticle[{{1, 1}}, "F", mq2, "TopL"];
RepBotLParticle = CreateParticle[{{1, 2}}, "F", mq2, "BotL"];
RepTopRParticle = CreateParticle[{{2, 1}}, "F", mq2, "TopR"];

(* Gauge bosons. *)
RepGluon = CreateParticle[{1}, "V", mg2, "Gluon"];
RepW = CreateParticle[{{2, 1}}, "V", mW2, "W"];

(* Higgs-doublet and real-singlet excitations. *)
RepHiggs = CreateParticle[{1}, "S", mH2, "Higgs"];
RepSinglet = CreateParticle[{2}, "S", mS2, "Singlet"];


(* Species with independent kinetic perturbations. *)
ParticleList = {
    RepTopLParticle,
    RepTopRParticle,
    RepBotLParticle,
    RepHiggs
};

(* Equilibrium bath species: they enter scattering amplitudes but have no
independent incoming perturbation. *)
LightParticleList = {
    RepSinglet,
    RepGluon,
    RepW
};


(* ::Section:: *)
(*Matrix-element export*)


OutputFile = "matrixElements.ewbg_chiral";

MatrixElements = ExportMatrixElements[
    OutputFile,
    ParticleList,
    LightParticleList,
    {
        (* Keep finite Yukawa and scalar-contact processes. In particular,
        lHS-mediated scattering is what allows Singlet to remain an
        equilibrium bath species while relaxing the Higgs perturbation. *)
        TruncateAtLeadingLog -> False,
        Format -> {"json", "txt"}
    }
];

MatrixElements
