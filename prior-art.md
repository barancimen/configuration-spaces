# Prior art: configuration spaces of mechanisms

Researched 2026-11-12 and 2026-11-13. As far as I found, no tool does everything cspace is aiming at:
arbitrary solids plus constraints, an exact configuration space, proofs about it, and
large scale. Several tools each cover part of it.

| Tool | What it does | What's missing |
|---|---|---|
| **Sacks & Joskowicz, configuration-space method / HIPAIR** | Exact configuration spaces of *contact* pairs: gears, cams, ratchets, escapements. Kinematic tolerance analysis finds play, interference and jamming when part dimensions vary. Book: *The Configuration Space Method for Kinematic Design of Mechanisms* (MIT Press, 2010). | Planar, pair by pair, 1990s research software with no current download found. **Closest prior work, especially for watches.** |
| **CuikSuite** (IRI Barcelona, GPLv3) | Computes configuration spaces and singularities of closed-chain mechanisms from a mechanism file. | Numerical (branch-and-prune boxes), no collisions or claims, tens of bodies. |
| **Bertini, HomotopyContinuation.jl** | Numerical algebraic geometry: all solutions, components, certification; used for linkage synthesis. | Small systems, no shapes, no general statements. |
| **CAD constraint solvers** (e.g. Siemens D-Cubed DCM) | Large assemblies: find one consistent pose, report remaining freedoms, split into independent groups. | Numerical, one pose at a time, no proofs. |
| **Computational mechanism design** (Disney/ETH: Coros, Thomaszewski) | Optimise linkages and gear trains to produce target motions, then fabricate. | No verification; small planar mechanisms. |
| **Formal verification of kinematics** (HOL4, Coq) | Machine-checked theorems about robot kinematics. | Proofs guided by hand; nothing automatic for mechanisms. |
| **Discrete formal models** (Alloy, clingo, an SMT solver used directly) | Describe parts and "locks to" relations as a finite model; find counterexamples to invariants, enumerate valid assemblies, check a build order for deadlock. | The geometry is assumed: that two parts lock is an axiom, and that is the thing needing proof. For the helix tower's joint the obvious axiom is false: a segment's own two pins do not hold it. |
| **Stability of rigid assemblies** (Mattikalli, Baraff & Khosla; Whiting et al.; Wang, Song & Pauly) | Decide without simulation whether an assembly stands: look for contact forces that balance gravity, by linear or quadratic programming. *Interlocking* designs hold every part but one key piece by geometry alone. | Numerical, one pose at a time; with friction the test is conservative. No motion, no claims over a configuration space. |
| **Watch industry** | Simulation: Ansys transient analysis (Audemars Piguet), Maple for barrel springs (Ulysse Nardin), finite-element models of lever escapements, an EPFL thesis on escapement synthesis. | Dynamics of single components; no global configuration-space reasoning. |

## Why nothing does it all at scale

The barrier is complexity, not neglect. Exact real arithmetic gets dramatically more expensive
with each variable, and linkages can encode essentially any algebraic set. Every tool that
scales gives something up: proofs (CAD solvers), generality (planar pairs), or global claims
(cspace's local prover).

## What a watch actually needs

- **Scale is modest.** Patek Philippe's Calibre 89 has 1,728 parts and Vacheron Constantin's
  ref. 57260 has 2,826. Millions of parts is aircraft scale.
- **A movement is essentially 2.5D:** parallel axles and planar parts. A planar engine is
  much cheaper than general 3D.
- **Gear trains are easy:** fixed axles and exact ratio couplings, which compose trivially.
- **The hard parts are contact pairs:** escapement, cams, jumpers, ratchets, keyless works.
  cspace doesn't model contact yet; this is exactly the Sacks–Joskowicz territory.
- **Useful claims are mostly local to one module:** no jamming, no interference within
  tolerance, safety actions hold. That's where cspace's decomposition scales.
- **"Best" is mostly dynamics:** isochronism, efficiency, power reserve. Kinematic
  verification is a constraint inside an optimiser driven by dynamic simulation, not the
  whole answer.

## Suggested direction

1. A planar engine with contact pairs computed from part outlines.
2. Tolerance-aware claims checked per module.
3. An optimiser plus dynamic simulation that uses the verifier as a hard constraint.

## Sources

- Sacks & Joskowicz book (library record): https://library.usi.edu/record/1412373
- Configuration space computation for mechanism design: https://cris.huji.ac.il/en/publications/configuration-space-computation-for-mechanism-design/
- HIPAIR (AAAI-94): https://cdn.aaai.org/AAAI/1994/AAAI94-259.pdf
- A Catalog of Higher Pairs and Their Configuration Spaces: https://cs.princeton.edu/research/techreps/126
- Kinematic tolerance analysis: https://cris.huji.ac.il/en/publications/kinematic-tolerance-analysis-13/
- CuikSuite documentation: https://www.iri.upc.edu/research/webprojects/cuikweb/CuikSuite3-Doc/html/index.html
- HomotopyContinuation.jl: https://arxiv.org/pdf/1711.10911
- Siemens D-Cubed 3D DCM: https://www.siemens.com/en-gb/products/plm-components/d-cubed/3d-dcm/
- Computational Design of Mechanical Characters: https://eth-cdl.github.io/publications/2013/CompDesignMechCharacters
- Computational Design of Linkage-Based Characters: https://ri.cmu.edu/publications/computational-design-of-linkage-based-characters
- Formal analysis of the kinematic Jacobian (HOL4): https://pdxscholar.library.pdx.edu/ece_fac/475
- Alloy: https://alloytools.org/
- clingo: https://potassco.org/clingo/
- Finding All Stable Orientations of Assemblies with Friction: https://doi.org/10.1109/70.488948
- Procedural Modeling of Structurally-Sound Masonry Buildings: https://dspace.mit.edu/handle/1721.1/73112
- State of the Art on Computational Design of Assemblies with Rigid Parts: https://diglib.eg.org/handle/10.1111/cgf142660
- EPFL thesis on free escapements: https://infoscience.epfl.ch/record/103745?ln=en
- Lever escapement FEA (Virginia Tech): https://vtechworks.lib.vt.edu/handle/10919/112753
- Ulysse Nardin / Maple: https://www.scientific-computing.com/news/watchmaker-uses-mathematical-software-lengthen-running-time-watches
