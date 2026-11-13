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

Limitations:
- `no_overlap` conditions are only checked, never used to steer sampling, so tight collision
  constraints may need many tries. Quantified ones are not checked at all.
- Samples are random but not uniformly distributed over the space.
- `where` shapes are not drawn, and differences are drawn without the cut.
- Meant for mechanisms up to tens of variables.
- Drawn with the browser's 2D canvas (no WebGL or external libraries), so it works everywhere;
  overlapping solids are depth-sorted face by face, which can misorder interpenetrating parts.

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
