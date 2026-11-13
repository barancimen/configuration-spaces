# cspace

Describe rigid solids and the constraints between them; get the **configuration space** of
the mechanism as exact polynomial equations and inequalities, and prove or refute claims
about it. Nothing is simulated: no motion, no forces.

```
python3 -m cspace examples/fourbar.yaml               # writes fourbar.json
python3 -m cspace examples/bennett.yaml -o - --check  # print it; also measure the dimension
python3 -m cspace examples/cube_on_cube.yaml --prove  # decide the file's claims
```

Requires Python 3.10+, SymPy, PyYAML, NumPy, SciPy and z3.

## What comes out

A semialgebraic set: a list of variables, polynomial equations `= 0`, and a boolean
combination of polynomial inequalities, written as JSON with SymPy-readable strings:

| Key | Content |
|---|---|
| `parameters`, `assumptions` | symbolic parameters and what the file assumes about them |
| `variables` | the coordinates of the space |
| `equations` | each polynomial that must vanish (unit-circle and unit-quaternion relations included) |
| `inequalities` | each `g` with `g >= 0`, from joint ranges and cones |
| `formulas` | `no_overlap` conditions |
| `poses` | each body's 4×4 world pose as polynomials in the variables |
| `coordinates` | every joint's coordinates (cosine, sine, displacement) as polynomials, including joints that close loops |
| `shapes` | each body's solid as a predicate in body coordinates `x, y, z` |

The `--check` diagnostic samples real points numerically. At each point it reports the
number of variables minus the rank of the Jacobian, which is the local dimension at
smooth points. It sits next to the naive count (6 per body minus each constraint's
codimension), and the two often disagree: Bennett has a naive count of −2 but moves
with 1 freedom.

## Viewing configurations

```
python3 -m cspace examples/fourbar.yaml --view          # writes fourbar.html
python3 -m cspace.viewer -o explorer.html examples/fourbar.yaml:a=2,b=5,c=4,d=6 examples/gears.yaml
```

The page draws a mechanism at random valid poses. Open it in any browser; no server is needed.
- **Sample anywhere:** a random start, pulled onto the equations by Levenberg–Marquardt.
- **Sample nearby:** a random step along the tangent space at the current pose, pulled back
  and accepted only if it stayed close.
- **Random walk:** repeats nearby steps.
- **Parameters** are editable; `path:name=value` sets their starting values.
- A body may set `color: "#rrggbb"` for drawing; otherwise colours come from a palette.

Each mechanism is compiled into the page as JavaScript: equations, exact Jacobian, limits,
poses, joint coordinates and shapes. Every shown pose satisfies the equations to about 1e-13,
the joint limits, and the quantifier-free `no_overlap` conditions.

**Replaying a simulation.** `--replay run.npz` makes the page play back a motion recorded from
MuJoCo instead of sampling:

```
python3 -m cspace.viewer --replay run.npz examples/helixturm.yaml -o helixturm.html
```

The file holds the arrays `time` and `qpos`, one row per frame, from a model that `cspace.mjcf`
exported from the first mechanism listed. The page gets a play button and a time slider. Hinges,
slides and free bodies are read back; a mechanism with a ball joint in its tree cannot be replayed.

Limitations:
- `no_overlap` conditions are only checked, never used to steer sampling, so tight collision
  constraints may need many tries. Quantified ones are not checked at all.
- Samples are random but not uniformly distributed over the space.
- `where` shapes are not drawn, and differences are drawn without the cut.
- Meant for mechanisms up to tens of variables.
- Drawn with the browser's 2D canvas (no WebGL or external libraries), so it works everywhere;
  overlapping solids are depth-sorted face by face, which can misorder interpenetrating parts.

## Simulating in MuJoCo

```
python3 -m cspace.mjcf examples/stewart.yaml -o stewart.xml --motors leg1,leg2,leg3,leg4,leg5,leg6
python3 examples/simulate_mujoco.py stewart.xml --wiggle --view      # needs: pip install mujoco
```

`cspace.mjcf` writes MuJoCo's MJCF format.

| cspace | MuJoCo |
|---|---|
| spanning tree | body tree |
| revolute, prismatic, spherical, free | hinge, slide, ball, freejoint |
| universal, cylindrical, planar | several hinges and slides in one body, axes computed at the sample |
| joint ranges and limits | joint `range` |
| loop-closing spherical / fixed / revolute | `connect` / `weld` / two `connect`s on the axis |
| two-joint linear `gear` / `couple` | joint equality |
| box, ball, cylinder, hull, unions | geoms (hull becomes a mesh) |
| bodies without a shape | thin capsules between their frames |
| `no_overlap` pairs | the only contact pairs |

**Starting pose.** The model is written at a valid sampled pose, with each hinge and slide's
`ref` set to its value there. MuJoCo derives loop-closing anchors from that reference pose, so
every loop starts exactly closed. `--keyframes N` adds more valid samples as keyframes.

**Motors.** `--motors` adds position actuators. By default the gain is set so gravity moves each
joint by about 1% of the mechanism's size, with near-critical damping; `--kp` overrides it.

**Mass.** Mass comes from geom volumes at `--density` (default 1000), in the file's length units.

Not exported, reported as warnings:
- loop-closing prismatic, cylindrical, planar and universal joints (list them so they become tree
  joints);
- the implicit constraints (`point_on_line`, `point_on_plane`, `distance`, `parallel`);
- couplings of more than two joints;
- cone limits;
- non-convex shapes (`difference`, `where`).

MuJoCo enforces constraints softly, so loops drift slightly during simulation, and contact
replaces `no_overlap`.

**Simulation-only keys.** None of these change the configuration space, the prover or the viewer:

```yaml
constraints:
  - revolute: [world.holdPivot, holdPawl]
    name: holdPawl
    spring: {stiffness: 0.02, rest: 0.35, damping: 0.001}   # joint stiffness and springref
  - contact: [holdPawl, ratchet]       # these bodies collide in MuJoCo
    friction: 0.2
start:                                 # initial pose: listed joints held exactly, loops closed
  holdPawl: -0.25
simulation:                            # defaults for the exporter; command-line flags override
  gravity: false
  timestep: 0.0005
  density: 600                         # for the bodies' mass, instead of --density
  solref: [0.004, 1]                   # how stiff contacts and joint stops are: MuJoCo's time
                                       # constant (seconds) and damping ratio; solimp: likewise
  motors: {lever: {kp: 5, damping: 0.05}}
```

`contact` differs from `no_overlap`. `no_overlap` restricts the configuration space and is exact,
but for toothed parts its formulas are far too large. `contact` only tells MuJoCo which shapes
push on each other.

`examples/ratchet_drive.yaml` combines all of this in 10 bodies:
- a motor-driven lever with a spring-loaded drive pawl;
- a 12-tooth ratchet held by a spring-loaded latch pawl;
- a 3-gear train;
- a slider-crank;
- a detent catch that clicks into a V-notch on the slider.

The tests check that its start pose has no contact pair interpenetrating.

**How it is tested.** `tests/test_mjcf.py` needs no MuJoCo: it uses a small forward-kinematics
interpreter that follows MuJoCo's documented conventions. For every keyframe of nine models, it
reproduces the compiler's body poses to 1e-8. Every connect, weld and joint equality also holds
in every keyframe. In MuJoCo itself (3.17), the exported four-bar and Stewart platform load and
run with their loops closed at the start (worst loop error over two seconds of driving: 1.7e-4
and 2.4e-3 in the files' length units), and the helix tower below is built and checked by
`tests/test_helixturm.py`.

## The helix tower

`examples/helixturm.yaml` is a tower that builds itself upward from stored segments, after Konrad
Zuse's Helixturm (HT1, 1993). It is described in cspace's format and simulated in MuJoCo with
gravity, contact, friction, springs and one motor; `examples/helixturm_lock.yaml` proves how its
segments lock.

```
python3 examples/make_helixturm.py > examples/helixturm.yaml        # the generator; --turns, --columns
python3 -m cspace.mjcf examples/helixturm.yaml -o helixturm.xml
python3 examples/run_helixturm.py helixturm.xml --push 10 --record run.npz   # build it, check it
python3 -m cspace.viewer --replay run.npz examples/helixturm.yaml -o helixturm.html   # watch it
python3 -m cspace examples/helixturm_lock.yaml --prove              # the lock, decided by z3
```

**The machine.** 50 bodies: a hexagonal tower with six columns, around a fixed stator.
- **Rotor:** a staircase cam, turned by the only motor. Per revolution it has six flats joined by
  six short ramps, then one steep return.
- **Lifter and bolt** (one pair per column): the lifter slides vertically and its foot rides the cam.
  It carries a spring-loaded latch bolt that reaches into a segment's window and lifts it by the
  lintel. The bolt is flat on top and bevelled below, so on the way down it is pushed back, passes
  the next segment and snaps into that one's window.
- **Magazine** (one per column): a spring-loaded plate pushes a queue of segments toward the tower.
- **Segments** (18, free bodies): two legs and a lintel around a window, a finger with a pin at
  each end, and two slots for neighbours' pins.
- **Head:** one turn of segment stubs under a cap; its underside is the helix.

**One step** is a sixth of a revolution and adds one segment:
1. All bolts rest on flats. The column whose bottom has reached one segment height gets the next
   segment pushed under it by its magazine.
2. That column's lifter sets it down on the new segment, drops to it and latches.
3. All six lifters rise by a sixth of a segment height.

Segment *i* goes into column *i* mod 6, so each sits a sixth of a height below the one before it.

**What the simulation shows** (18 segments, one every 1.6 s, 34 kg in all, the head 9 kg):

| Check | Result |
|---|---|
| Segments within 2 mm and 0.5° of their place | 18 of 18 |
| Head rise | 720.02 mm (three turns are 720 mm), tilt 0.01°, 0.75 mm off axis |
| Rotor stopped, tower on its bolts | nothing moves more than 0.12 mm in 1.5 s |
| 10 N sideways on the head for a second | the head gives 1.8 mm; all segments stay within 5 mm |
| Motor | peak torque on the cam 27 N m; 156 J of work, of which 111 J became height |

A 20 N push makes the head give 8.5 mm: the joints have play, and the push starts to tip the tower.

**What the prover shows.** A segment's own two pins slide into slots of earlier segments along its
own radial direction. Two later segments then pin it, each coming in along a direction turned by
60°: segment *i*+1 at its right end, segment *i*+5 at its lower left corner.

| Claim | Model | Answer |
|---|---|---|
| `held`: within 6 mm of its place, the segment is within 1 mm (2 mm vertically) | real shapes, no turning | proved |
| `comesOutLast`: with both later segments drawn out, it slides out | real shapes | proved |
| `neverComesOut` (false on purpose) | real shapes | refuted |
| `pinned…` (5 claims): within 0.6 mm and 0.3° when all four pins sit in their channels | plane motion with turning | proved |
| `swingsWithoutLowerPin`: before segment *i*+5 is in, the left end can swing out | plane motion | proved |
| `slidesOnItsOwnPins`: on its own two pins it still slides straight out | plane motion | proved |

The plane-motion model treats a round pin between two rails as a point in a channel as wide as
the clearance. With the real shapes and turning together, z3 did not finish in six minutes.

**What it is not.**
- **It only extends.** The latch bolts work one way. Zuse's lifting arms were moved in and out by
  a second cam, which is what lets his tower also retract.
- **It is not Zuse's geometry.** His segments were slanted sheet-metal parts and the motion was
  continuous, with eight magazines. Here the segments are boxes, the motion is stepwise, and the
  generator allows 4 to 6 columns, because the joint has to fit in the corner between two walls.
- **The lock is proved for one segment and its neighbours,** not for the whole tower, and not
  against tilting out of the wall's plane. The simulation covers those.
- **The machine has to be stopped** when the magazines are empty; the run script does that.
- Contacts in MuJoCo are slightly soft (a 4 ms time constant), and the clearances are 0.2 to
  2 mm on a 240 mm segment.

## Proving claims

```
python3 -m cspace examples/cube_on_cube.yaml --prove
python3 examples/make_snake.py 1000 > snake.yaml && python3 -m cspace snake.yaml --prove -v
```

Claims go in the mechanism file under `claims:` (or in a separate file passed with `--claims`):

```yaml
claims:
  - name: neverConvex
    always: not convex(base, slider)
  - name: neverHangsOver
    always: "forall(p, implies(inside(p, slider), inside([p.x, p.y, A/2], base)))"
  - name: cornerReachable
    possible: slideX.d == (A - a)/2 and slideY.d == (A - a)/2
```

`always: F` means F holds in every configuration. `possible: F` means F holds in at least
one configuration. Both are checked for every parameter value that satisfies `assume:`.

Two per-claim options:
- `assume: "a <= b and ..."` adds conditions on the parameters for this claim only.
- `given: [joint, ...]` (for `possible` claims) means: for every value of these joints'
  coordinates allowed by their own limits, some configuration exists that also satisfies
  F. For example, `given: [crank]` with `possible: "True"` says the crank can turn through
  its whole range. This puts a second quantifier in front of the configuration, so it is
  much harder for the solver. It works for small or open systems; on closed loops such as
  the four-bar it times out.

Each claim gets one of four answers. All of them come from z3, an automatic theorem prover
with an exact decision procedure for polynomial arithmetic over the reals.

| Answer | Meaning |
|---|---|
| `proved` | The claim holds. |
| `refuted` | It fails; the counterexample is a real configuration. |
| `unknown` | Neither could be established within the limits. |
| `error` | The claim could not be read. |

**Formula language.** The syntax is Python-like, but claims are parsed, never executed.
- **Arithmetic and logic:** `+ - * / ** ^`, comparisons (chains allowed), `and`, `or`, `not`,
  `implies(F, G)`, `iff(F, G)`. Division is only allowed by nonzero numbers.
- **Joint coordinates:** `joint.d`, `joint.c`, `joint.s` (the cosine and sine of a hinge
  angle), `joint.q0`, …
- **Positions:** `pos(body)` or `pos(body.frame)`, `point(body, x, y, z)`, `axis(body.frame)`,
  `[x, y, z]`, `p.x`, `dot`, `cross`, `norm2`, `dist2`.
- **Solids:**
  - `inside(p, B)`: p is in the closed solid; `interior(p, B)`: p is strictly inside
  - `overlap(A, B)`: the interiors intersect; `meet(A, B)`: the solids touch or intersect
  - `convex(A, B, …)`: the union of the solids is convex
- **Quantifiers:** `forall(p, F)` and `exists(p, F)` over points; `forall_real(t, F)` and
  `exists_real(t, F)` over numbers.

A name that is both a body and a joint resolves by context. `.c`, `.d` and the other
coordinate names refer to the joint, `.frame` refers to the body, and functions such as
`overlap` and `pos` take the body.

**How it scales.** A claim only needs the bodies it mentions and whatever constrains them.
1. **Independent groups (exact).** Bodies connected to each other only through the fixed
   world move independently. A claim only needs the groups of the bodies it mentions.
2. **Dangling bodies (exact).** A body hanging off by a single joint, with nothing else
   attached, can always follow the rest, so it is removed. This repeats until nothing
   dangles.
3. **Neighbourhoods.** If a group is still larger than `--max-bodies` (default 12), the
   prover works on the bodies within 0, 1, 2, … steps of the mentioned ones, two ways:
   - **relaxed:** constraints to the rest are dropped, so there are *more* configurations.
     A proof of an `always` claim, or a refutation of a `possible` claim, carries over.
   - **restricted:** the joints of the rest are held at zero, so there are *fewer*
     configurations. A witness for a `possible` claim, or a counterexample to an `always`
     claim, is real.
   - Any other outcome is inconclusive, and the prover widens the neighbourhood.

On a 1000-link chain (`examples/make_snake.py`), 7 of 8 local claims about links in the
middle are decided in about 30 s in total. The eighth, `staysInPlane`, is true but stays
unknown, because proving it requires following the chain all the way to the world.

**Limits.**
- Statements about the whole of a large connected mechanism stay out of reach. Real
  arithmetic with quantifiers is decidable, but the cost can grow doubly exponentially
  with the number of variables.
- Constants must be algebraic: `sqrt(2)` and `cos(pi/12)` are fine, but `pi` itself and
  the `--trig` view are not.
- The prover trusts z3's answers.
- Each solver call gets a z3 context of its own. Without that, how long z3 took on a claim
  depended on which claims the same run had asked before it.

Options: `--timeout` (per solver call), `--budget` (per claim), `--max-bodies`, and `-v`
to see each attempt.

## Coordinates

- An angle is a pair `(c, s)` with `c^2 + s^2 == 1`, so everything stays polynomial.
- Variables are named role + joint name: `cElbow`, `sElbow` (hinge `elbow`), `dSlide`
  (slider `slide`), `q0Ball`..`q3Ball` (ball joint), `xPuck`.. (free body).
- **reduced** (default): joints on a spanning tree from `world` (file order) carry
  coordinates; every other constraint becomes equations on the resulting poses (loop
  closure). Bodies that no tree joint reaches get a free joint (position + unit quaternion).
- **maximal** (`--coords maximal`): every body gets position + unit quaternion, and every
  constraint is an equation between two poses.
- **trig** (`--trig`): tree hinges get an angle `thElbow` with `Cos`/`Sin` instead of
  `(c, s)`. This is the only mode for angle–length couplings (screws, rack and pinion) and
  ranges wider than one turn.
- Equations are reduced modulo the circle/quaternion relations, have their rational
  content removed and are deduplicated.

## Input format (YAML)

```yaml
name: two_rods
parameters: {l1: symbolic, l2: 3}           # symbolic, or an exact value; override with --set l2=4
assume: [l1 > 0]
bodies:
  world:                                    # predefined, fixed; may list frames, no shape
    frames: {o: {}}
  rod1:
    shape: {box: [l1, 1/10, 1/10], at: [l1/2, 0, 0]}
    frames:
      a: {at: [0, 0, 0]}
      b: {at: [l1, 0, 0], z: [0, 0, 1], x: [1, 0, 0]}   # origin, joint axis, zero direction
  rod2: {frames: {a: {}}}
constraints:
  - fixed: [world.o, rod1.a]
  - revolute: [rod1.b, rod2.a]
    name: elbow
    range: [-pi/2, pi/2]
```

All numbers are exact: `0.1` means 1/10, `pi/3` stays symbolic. A reference is `body.frame`
or `body`, which means the body's own frame. Names are letters and digits. `x`, `y`, `z`,
`pi`, `sqrt`, `sin`, `cos` and `tan` are reserved.

### Pairwise constraints

Each acts on the pose of frame 2 relative to frame 1. Axes are the frames' z axes.

| Kind | Freedoms | Meaning |
|---|---|---|
| `fixed` | 0 | frames coincide |
| `revolute` | 1 | hinge about z |
| `prismatic` | 1 | slide along z |
| `cylindrical` | 2 | hinge + slide along z |
| `planar` | 3 | slide in the xy plane, turn about z |
| `spherical` | 3 | ball joint: origins coincide |
| `universal` | 2 | about frame 1's x axis, then frame 2's y axis |
| `free` | 6 | no constraint (coordinates only) |
| `point_on_line` | 4 | frame 2's origin on frame 1's z axis |
| `point_on_plane` | 5 | frame 2's origin in frame 1's xy plane |
| `distance` | 5 | origins at `length: L` |
| `parallel` | 4 | z axes coincide in direction |

The last four have no coordinates of their own and always close loops.

Options:
- `name:` used in variable names and couplings (default `j1`, `j2`, …).
- `range: [lo, hi]` on the joint's main coordinate (angle, or displacement for `prismatic`).
- `limits: {angle: [..], d: [..], angle1: [..], angle2: [..]}` for joints with several coordinates.
- `cone: A` (ball joints): frame 2's z axis stays within angle A of frame 1's z axis.
- `closing: true` keeps a joint out of the spanning tree.

### Couplings (n-ary)

```yaml
- gear: [pinion, wheel]
  teeth: [20, 40]          # external mesh; internal: true for a ring gear; phase: angle
- couple: {h1: 1, h2: -2}  # sum of coefficient * coordinate = equals
  equals: 0
- couple: {slide.d: 1, crank.angle: -pitch/(2*pi)}   # needs --trig
```

An angle coupling `Σ a_i θ_i = φ` with rational `a_i` becomes `Π z_i^(a_i) = e^(iφ)`, with
`z = c + i s` and the `a_i` scaled to coprime integers. This is exactly the set of poses the
real relation reaches. A coupling may refer to a loop-closing joint's coordinate.

### Shapes and `no_overlap`

Leaves:
- `box: [lx, ly, lz]` (centred)
- `ball: r`
- `cylinder: [r, h]` (axis z, centred)
- `hull: [[x, y, z], …]` (numeric vertices)
- `where: ["x^2 + y^2 <= 1", "z >= 0"]` (polynomial inequalities)

Combinators are `union`, `intersection` and `difference` (the first child minus the others).
Any node may carry `at`, `z`, `x`.

`no_overlap: [A, B]` means no point is interior to both bodies:
- Unions of boxes, hulls and balls give an exact quantifier-free formula: separating
  axes for polytope pairs, centre distance for ball pairs.
- Anything else, including a ball against a polytope, gives an exact condition
  quantified over all points, which the prover handles directly.

## Examples

| File | Shows |
|---|---|
| `two_rods` | the linkage from the request; symbolic lengths, hinge range |
| `fourbar` | a planar loop; naive count −2, true dimension 1 |
| `slider_crank` | prismatic joint, displacement range |
| `bennett` | overconstrained spatial loop that still moves |
| `rssr` | ball joints, a cone limit, an idle freedom (dimension 2) |
| `gears` | a gear coupling between two hinges |
| `screw` | angle–length coupling (`--trig` only) |
| `tethered_puck` | implicit constraints on a free body |
| `sliding_boxes` | exact box/box condition (\|s\| ≥ 2) and a `ForAll` fallback |
| `door_stop` | collision turns a hinge's circle into an arc |
| `stewart` | 14-body Stewart platform: six closed spatial loops, 37 variables, 6 freedoms |
| `cube_on_cube` | claims: never convex, never hangs over, a deliberately false claim |
| `make_snake.py` | generates an N-link chain with local claims (scaling test) |
| `ratchet_drive` | springs, contacts and a start pose for MuJoCo (from `make_ratchet_drive.py`) |
| `helixturm` | a self-building tower: 50 bodies, 19 of them free, contacts only (from `make_helixturm.py`; run with `run_helixturm.py`) |
| `helixturm_lock` | the tower's joints alone, with claims about how they lock |

## Are pairwise constraints enough?

For joints, yes: every joint, range and collision is a condition on one relative pose.
Couplings are the exception. A gear ratio relates two joint angles, which involves three
or four bodies, and differentials and cable runs involve more. So `couple` and `gear` take
any number of joints. The space is the intersection of all constraints, and its dimension
is a global property: the `--check` output above shows it disagreeing with the naive count.

## Limits

- Gröbner bases work over the complex numbers; the real dimension can be lower.
- Unit quaternions cover each rotation twice.
- Angle–length couplings are not algebraic, so they need `--trig`.
- Rolling without slipping in 3D constrains velocities, not poses, so it cannot be stated here.
- `hull` vertices must be numeric. The fallback `ForAll` formulas are exact, but resolving
  them can be slow.
- Spec strings are evaluated by SymPy's parser: only compile files you trust.
- Not included: mesh import, the two-body rolling gear law, cams, a quantifier-free test
  for a ball against a polytope.

## Tests

The installed pytest 5.2 needs plugin autoload off on Python 3.10:

```
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q -p no:cacheprovider --assert=plain tests/
```

`tests/test_helixturm.py` takes about a minute. Its MuJoCo run (two steps of a small tower) is
skipped when MuJoCo is not installed.
