# cspace

Compile a description of rigid solids and the constraints between them into the
**configuration space** of the mechanism, written as exact polynomial equations and
inequalities you can manipulate symbolically (for example in Mathematica). Nothing is
simulated: no motion, no forces.

```
python3 -m cspace examples/fourbar.yaml            # writes fourbar.wl
python3 -m cspace examples/fourbar.yaml -o - --format json
python3 -m cspace examples/bennett.yaml --check    # also measure the dimension numerically
```

Requires Python 3.10+, SymPy, PyYAML, NumPy, SciPy (z3 for the tests).

## What comes out

A semialgebraic set: a list of variables, polynomial equations `== 0`, and a negation-free
boolean combination of polynomial inequalities. The `.wl` file defines one association:

| Key | Content |
|---|---|
| `Parameters`, `Assumptions` | symbolic parameters and what the file assumes about them |
| `Variables` | the coordinates of the space |
| `Equations` | `p == 0` for each polynomial (unit-circle and unit-quaternion relations included) |
| `Inequalities` | `g >= 0` from joint ranges and cones |
| `Constraints` | inequalities plus `no_overlap` conditions |
| `Formula` | everything, one formula |
| `Poses` | each body's 4×4 world pose as polynomials in the variables |
| `Coordinates` | every joint's coordinates (cosine, sine, displacement) as polynomials, including joints that close loops |
| `Shapes` | each body's solid as `Function[{x, y, z}, ...]` in body coordinates |

How you might use it in Mathematica (not run here, since this machine has no Mathematica):

```mathematica
Get["fourbar.wl"];
GroebnerBasis[cspace["Equations"], cspace["Variables"]]
Eliminate[cspace["Equations"], {cKnee, sKnee, cElbow, sElbow}]     (* crank angle vs parameters *)
Reduce[cspace["Formula"] && cspace["Assumptions"], cspace["Variables"], Reals]
Resolve[cspace["Constraints"], Reals]                             (* removes ForAll from no_overlap *)
cspace["Poses"]["coupler"] . {b/2, 1, 0, 1}                        (* a coupler point, for its curve *)
```

The `--check` diagnostic samples real points numerically. At each point it reports the
number of variables minus the rank of the Jacobian, which is the local dimension at
smooth points. It sits next to the naive count (6 per body minus each constraint's
codimension), and the two often disagree: Bennett has a naive count of −2 but moves
with 1 freedom.

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
or `body`, which means the body's own frame. Names are letters and digits. `x`, `y`, `z`
and Mathematica's one-letter built-ins (`C D E I K N O`) are reserved.

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
- Anything else, including a ball against a polytope, gives an exact
  `ForAll[{x, y, z}, …]` for `Resolve`.

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
