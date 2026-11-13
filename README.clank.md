# cspace

cspace works out the configuration space of a mechanism, which is the set of all poses its
parts can take. You describe rigid bodies and the constraints between them in a YAML file.
cspace turns the description into polynomial equations and inequalities with exact
coefficients. It can also prove or disprove statements about the result.

cspace does not simulate motion or forces. It can export a mechanism to MuJoCo, which does.

```
python3 -m cspace examples/fourbar.yaml               # writes fourbar.json
python3 -m cspace examples/bennett.yaml -o - --check  # prints it and also measures the dimension
python3 -m cspace examples/cube_on_cube.yaml --prove  # decides the claims in the file
```

It needs Python 3.10 or later, SymPy, PyYAML, NumPy, SciPy and z3.

## Output

The output is a semialgebraic set written as JSON. It has a list of variables, polynomial
equations that must equal zero, and a boolean combination of polynomial inequalities. All
expressions are strings that SymPy can read.

| Key | Content |
|---|---|
| `parameters`, `assumptions` | the symbolic parameters and what the file assumes about them |
| `variables` | the coordinates of the space |
| `equations` | the polynomials that must be zero, including the unit-circle and unit-quaternion relations |
| `inequalities` | each `g` that must satisfy `g >= 0`, from joint ranges and cones |
| `formulas` | the `no_overlap` conditions |
| `poses` | the 4×4 world pose of each body, as polynomials in the variables |
| `coordinates` | the coordinates of every joint (cosine, sine, displacement) as polynomials, including joints that close loops |
| `shapes` | the solid of each body, as a condition on the body coordinates `x, y, z` |

`--check` samples points of the set numerically. At each point it reports the number of
variables minus the rank of the Jacobian. At a smooth point this is the local dimension. Next
to it, it prints the simple count, which is 6 per body minus the number of freedoms each
constraint removes. The two numbers often differ. For the Bennett linkage the simple count is
−2, but the linkage moves with one degree of freedom.

## Viewing configurations

```
python3 -m cspace examples/fourbar.yaml --view          # writes fourbar.html
python3 -m cspace.viewer -o explorer.html examples/fourbar.yaml:a=2,b=5,c=4,d=6 examples/gears.yaml
```

The page draws a mechanism at random valid poses. It opens in any browser and needs no server.

- The "Sample anywhere" button starts from a random point and moves it onto the equations
  with the Levenberg–Marquardt method.
- The "Sample nearby" button takes a random step along the tangent space at the current pose
  and moves the result back onto the equations. The new pose is accepted only if it stayed
  close to the old one.
- The "Random walk" button repeats nearby steps.
- Parameters can be edited on the page. `path:name=value` on the command line sets their
  starting values.
- A body can set `color: "#rrggbb"`. Bodies without one get colours from a palette.

Each mechanism is compiled into the page as JavaScript. This includes its equations, the
exact Jacobian, limits, poses, joint coordinates and shapes. Every pose shown satisfies the
equations to about 1e-13. It also satisfies the joint limits and those `no_overlap`
conditions that have no quantifiers.

### Replaying a MuJoCo recording

With `--replay run.npz` the page plays back a motion recorded in MuJoCo and does not sample
poses.

```
python3 -m cspace.viewer --replay run.npz examples/helixturm.yaml -o helixturm.html
```

The file must contain the arrays `time` and `qpos` with one row per frame. They must come
from a model that `cspace.mjcf` exported from the first mechanism on the command line. The
page then has a play button and a time slider. Hinges, slides and free bodies can be
replayed. A mechanism with a ball joint in its tree cannot.

### Limitations of the viewer

- `no_overlap` conditions are checked after sampling but do not guide it. A mechanism with
  tight collision constraints may therefore need many tries. Conditions with quantifiers are
  not checked.
- Samples are random but not uniformly distributed over the space.
- `where` shapes are not drawn. A `difference` is drawn without the cut.
- The viewer is meant for mechanisms with up to a few tens of variables.
- The page draws with WebGL and uses no external libraries. In a browser without WebGL it
  falls back to a 2D canvas and paints the faces from back to front. The fallback sometimes
  draws faces in the wrong order where parts touch or overlap, and the order can change
  from one frame to the next.

## Simulating in MuJoCo

```
python3 -m cspace.mjcf examples/stewart.yaml -o stewart.xml --motors leg1,leg2,leg3,leg4,leg5,leg6
python3 examples/simulate_mujoco.py stewart.xml --wiggle --view      # needs: pip install mujoco
```

`cspace.mjcf` writes a model in MuJoCo's MJCF format. The table shows what each cspace
feature becomes.

| cspace | MuJoCo |
|---|---|
| spanning tree | body tree |
| revolute, prismatic, spherical, free | hinge, slide, ball, freejoint |
| universal, cylindrical, planar | several hinges and slides in one body, with axes computed at the sample |
| joint ranges and limits | joint `range` |
| loop-closing spherical, fixed and revolute joints | `connect`, `weld`, and two `connect`s on the axis |
| a linear `gear` or `couple` of two joints | joint equality |
| box, ball, cylinder, hull, unions | geoms (a hull becomes a mesh) |
| bodies without a shape | thin capsules between their frames |
| `no_overlap` pairs | the only contact pairs |

The model is written at a valid sampled pose, and each hinge and slide gets its value at that
pose as its `ref`. MuJoCo computes the anchors of loop-closing constraints from this
reference pose, so every loop is closed at the start. `--keyframes N` adds N more valid poses
as keyframes.

`--motors` adds position actuators to the listed joints. The default gain lets gravity move
each joint by about 1% of the mechanism's size, and the damping is close to critical. `--kp`
sets the gain.

Mass is computed from the volume of each shape at `--density`. The default density is 1000,
in the file's length units.

The exporter leaves out the following and prints a warning for each:

- prismatic, cylindrical, planar and universal joints that close a loop. Reorder the
  constraints so that these joints end up in the spanning tree.
- the implicit constraints `point_on_line`, `point_on_plane`, `distance` and `parallel`
- couplings of more than two joints
- cone limits
- shapes that are not convex, which are `difference` and `where`

MuJoCo enforces constraints approximately, so closed loops drift a little during a
simulation. Contact between shapes takes the place of `no_overlap`.

### Keys used only for simulation

The keys below are read only by the exporter. They do not change the configuration space,
the prover or the viewer.

```yaml
constraints:
  - revolute: [world.holdPivot, holdPawl]
    name: holdPawl
    spring: {stiffness: 0.02, rest: 0.35, damping: 0.001}   # joint stiffness and springref
  - contact: [holdPawl, ratchet]       # these bodies collide in MuJoCo
    friction: 0.2
start:                                 # start pose: the listed joints get these values, loops are closed
  holdPawl: -0.25
simulation:                            # exporter defaults, which command-line flags override
  gravity: false
  timestep: 0.0005
  density: 600                         # used for mass in place of --density
  solref: [0.004, 1]                   # stiffness of contacts and joint stops, as MuJoCo's time
                                       # constant in seconds and damping ratio. solimp works the same way
  motors: {lever: {kp: 5, damping: 0.05}}
```

`contact` and `no_overlap` are different things. `no_overlap` restricts the configuration
space and is exact, but its formulas become far too large for parts with teeth. `contact`
only tells MuJoCo which shapes can push on each other.

`examples/ratchet_drive.yaml` uses all of these keys. It has 10 bodies:

- a lever driven by a motor, with a spring-loaded pawl that drives the ratchet
- a ratchet with 12 teeth, which a second spring-loaded pawl stops from turning back
- a train of three gears
- a slider-crank
- a spring-loaded catch that clicks into a V-shaped notch on the slider

The tests check that no pair of contacting bodies overlaps in its start pose.

### How the exporter is tested

`tests/test_mjcf.py` does not need MuJoCo. It contains a small forward-kinematics routine that
follows MuJoCo's documented conventions. For every keyframe of nine models it reproduces the
body poses that cspace computed, to within 1e-8. It also checks that every connect, weld and
joint equality holds in every keyframe.

In MuJoCo 3.17 itself, the exported four-bar linkage and Stewart platform load and run with
their loops closed at the start. Over two seconds of driving, the largest loop error was
1.7e-4 for the four-bar and 2.4e-3 for the Stewart platform, in the files' length units.
`tests/test_helixturm.py` builds and checks the helix tower described below.

## The helix tower

`examples/helixturm.yaml` describes a tower that builds itself upward from stored segments. It
is based on Konrad Zuse's Helixturm (HT1, 1993). The tower is written in cspace's format and
simulated in MuJoCo with gravity, contact, friction, springs and one motor. A second file,
`examples/helixturm_lock.yaml`, holds the proofs of how the segments lock together.

```
python3 examples/make_helixturm.py > examples/helixturm.yaml        # generate the file (options: --turns, --columns)
python3 -m cspace.mjcf examples/helixturm.yaml -o helixturm.xml     # export it to MuJoCo
python3 examples/run_helixturm.py helixturm.xml --push 10 --record run.npz   # build the tower and check it
python3 -m cspace.viewer --replay run.npz examples/helixturm.yaml -o helixturm.html   # replay it in a browser
MUJOCO_GL=egl python3 examples/render_helixturm.py helixturm.xml run.npz --rewind -o helixturm.gif   # render a GIF
python3 -m cspace examples/helixturm_lock.yaml --prove              # prove the lock claims
```

### Parts

The default tower has 50 bodies. It is hexagonal, with six columns of segments around a fixed
stator.

- The rotor is a staircase-shaped cam, and the only motor turns it. In one revolution the cam
  has six flat sections joined by six short ramps, followed by one steep drop.
- Each column has a lifter that slides vertically, with a foot that rides on the cam. The
  lifter carries a spring-loaded latch bolt. The bolt reaches into the window of a segment and
  lifts the segment by the lintel above the window. The bolt is flat on top and bevelled
  underneath. When the lifter moves down, the bevel pushes the bolt back, so the bolt passes
  the next segment and then snaps into that segment's window.
- Each column has a magazine, which is a spring-loaded plate that pushes a queue of segments
  toward the tower.
- There are 18 segments, and they are free bodies. A segment has two legs and a lintel around
  a window. At each end it has a finger with a pin, and it has two slots that take the pins of
  neighbouring segments.
- The head is the top of the tower. It is a cap with one turn of short segment stubs below
  it, and the lower edge of these stubs follows the helix.

### How one step works

One step is a sixth of a revolution of the rotor and adds one segment.

1. All bolts rest on flat sections of the cam. One column has risen far enough that its lower
   edge is one segment height above the floor. The magazine of that column pushes the next
   segment under it.
2. The lifter of that column lowers the column onto the new segment. The lifter then drops
   further, and its bolt latches into the new segment's window.
3. All six lifters rise by a sixth of a segment height.

Segment *i* goes into column *i* mod 6. Each segment therefore sits a sixth of a segment
height lower than the one before it, and the segments form a helix.

### Simulation results

The default run adds 18 segments, one every 1.6 s. All parts together weigh 34 kg, and the
head accounts for 9 kg of that.

| Check | Result |
|---|---|
| Segments within 2 mm and 0.5° of their intended position | 18 of 18 |
| Rise of the head | 720.02 mm (three turns are 720 mm), with 0.01° of tilt and 0.75 mm off the axis |
| Movement after the rotor stops, with the tower resting on its bolts | at most 0.12 mm in 1.5 s |
| A sideways force of 10 N on the head for one second | the head moves 1.8 mm, and every segment stays within 5 mm |
| Motor | 27 N m peak torque on the cam, and 156 J of work, of which 111 J went into lifting |

With a 20 N push the head moves 8.5 mm. The joints have some play, and at that force the
tower begins to tip.

### What the prover shows

When a segment is pushed in, its two pins slide into slots in earlier segments. Both pins
move along the segment's own radial direction, so on their own they would let the segment
slide back out. Two later segments hold it in place. Each of them is pushed in along a
direction that differs by 60° from the first segment's. Segment *i*+1 pins the right end, and
segment *i*+5 pins the lower left corner.

| Claim | Model | Answer |
|---|---|---|
| `held`: if the segment is within 6 mm of its position, it is within 1 mm of it (2 mm vertically) | real shapes, no rotation | proved |
| `comesOutLast`: once both later segments are pulled out, the segment can slide out | real shapes | proved |
| `neverComesOut`, which is deliberately false | real shapes | refuted |
| five `pinned…` claims: with all four pins in their channels, the segment is within 0.6 mm and 0.3° of its position | motion in the plane, with rotation | proved |
| `swingsWithoutLowerPin`: before segment *i*+5 is in place, the left end can swing outward | motion in the plane | proved |
| `slidesOnItsOwnPins`: held only by its own two pins, the segment can slide straight out | motion in the plane | proved |

The planar model replaces each round pin between two rails with a point in a channel that is
as wide as the clearance. When z3 was given the real shapes together with rotation, it did
not finish within six minutes.

### A taller tower

`make_helixturm.py --turns 6` puts six segments in each magazine. This gives 36 segments and a
total mass of 48 kg. The same machine builds this tower as well. The head rises 1440.00 mm in
58 s with 0.00° of tilt, and after the rotor stops nothing moves more than 0.17 mm. The peak
torque on the cam is 38 N m. The motor does 415 J of work, of which 309 J goes into lifting.

The play in the joints adds up as the tower gets taller. At the end the top is rotated by
0.4° about the axis and is 1.2 mm off it. Because of this, two segments in the first turn are
2.2 mm and 2.3 mm sideways from their intended positions. `run_helixturm.py` allows 2 mm for
every segment, so it counts 34 of 36 and reports that the run failed.

### Rendering a GIF

`examples/render_helixturm.py` renders a recording with MuJoCo's renderer and writes an
animated GIF. It needs Pillow and ffmpeg. Setting `MUJOCO_GL=egl` lets it render without a
display.

- The GIF shows two views side by side on a white background, with no text. The left view
  shows the whole tower. The right view shows the base from closer up.
- Every part is grey. Each kind of part has its own shade, so all segments have the same
  shade. The segment stubs of the head count as segments here, and only its cap has a shade
  of its own. The colours in the YAML file are ignored.
- Black outlines are drawn around every body and along every edge. MuJoCo renders the shaded
  image. The script then finds the outlines from the depth of each pixel and the body that
  the pixel belongs to.
- In the right view, the walls between the camera and the tower's axis become transparent
  during the first half turn of the rotor. Their shading fades to 5% opacity and their
  outlines stay. This makes the cam, the lifters and the latch bolts visible. The walls that
  fade are the segments, magazines, guides and floor of three columns, and the same side of
  the head.
- `--speed` sets the playback speed as a multiple of real time. The default is 4. The end of
  the recording, where the tower stands still, is left out.
- `--rewind` plays the frames backwards after the build, without a pause. The GIF then loops
  smoothly, and the walls become solid again at the end of the loop. The backwards half is
  the same recording in reverse. It is not a simulation of the tower retracting.

### Limitations of the tower

- The tower can only extend, because the latch bolts work in one direction. In Zuse's machine
  a second cam moved the lifting arms in and out, which allowed his tower to retract as well.
- The geometry is not Zuse's. His segments were slanted sheet-metal parts, the motion was
  continuous, and there were eight magazines. Here the segments are boxes and the motion is
  stepwise. The generator allows 4 to 6 columns, because the joint has to fit into the corner
  between two walls.
- The lock is proved for one segment and its neighbours. It is not proved for the whole
  tower, and it is not proved against a segment tilting out of the plane of its wall. Only
  the simulation covers those cases.
- The rotor has to be stopped when the magazines are empty. The run script does this.
- Contacts in MuJoCo are slightly soft, with a time constant of 4 ms. The clearances are
  between 0.2 and 2 mm on a segment that is 240 mm tall.

## Proving claims

```
python3 -m cspace examples/cube_on_cube.yaml --prove
python3 examples/make_snake.py 1000 > snake.yaml && python3 -m cspace snake.yaml --prove -v
```

Claims are written under `claims:` in the mechanism file, or in a separate file given with
`--claims`.

```yaml
claims:
  - name: neverConvex
    always: not convex(base, slider)
  - name: neverHangsOver
    always: "forall(p, implies(inside(p, slider), inside([p.x, p.y, A/2], base)))"
  - name: cornerReachable
    possible: slideX.d == (A - a)/2 and slideY.d == (A - a)/2
```

`always: F` states that F is true in every configuration. `possible: F` states that F is true
in at least one configuration. Both are checked for every parameter value that satisfies the
file's `assume:` conditions.

A claim can have two options.

- `assume: "a <= b and ..."` adds conditions on the parameters for that claim only.
- `given: [joint, ...]` applies to `possible` claims. The claim then states that for every
  value of these joints' coordinates that their own limits allow, there is a configuration
  that also satisfies F. For example, `given: [crank]` with `possible: "True"` states that the
  crank can turn through its whole range. This adds a second quantifier, which makes the
  problem much harder for the solver. It works for small systems and for systems without
  closed loops. On closed loops such as the four-bar linkage it times out.

All answers come from z3, an automatic theorem prover with an exact decision procedure for
polynomial arithmetic over the real numbers. Each claim gets one of four answers.

| Answer | Meaning |
|---|---|
| `proved` | The claim is true. |
| `refuted` | The claim is false. The counterexample that is printed is a real configuration. |
| `unknown` | z3 could not decide the claim within the time limits. |
| `error` | The claim could not be parsed. |

The options are `--timeout` for the time limit of one solver call, `--budget` for the time
limit of one claim, `--max-bodies`, and `-v` to print each attempt.

### Formula language

The syntax looks like Python, but claims are only parsed and never executed.

- Arithmetic and logic: `+ - * / ** ^`, comparisons (which can be chained), `and`, `or`,
  `not`, `implies(F, G)`, `iff(F, G)`. Division is allowed only by nonzero numbers.
- Joint coordinates: `joint.d`, `joint.c` and `joint.s` (the cosine and sine of a hinge
  angle), `joint.q0`, …
- Positions: `pos(body)` or `pos(body.frame)`, `point(body, x, y, z)`, `axis(body.frame)`,
  `[x, y, z]`, `p.x`, `dot`, `cross`, `norm2`, `dist2`.
- Solids:
  - `inside(p, B)` means p is in the closed solid. `interior(p, B)` means p is strictly inside.
  - `overlap(A, B)` means the interiors intersect. `meet(A, B)` means the solids touch or
    intersect.
  - `convex(A, B, …)` means the union of the solids is convex.
- Quantifiers: `forall(p, F)` and `exists(p, F)` range over points. `forall_real(t, F)` and
  `exists_real(t, F)` range over numbers.

A name can belong to both a body and a joint. The context decides which one is meant. `.c`,
`.d` and the other coordinate names refer to the joint. `.frame` refers to the body.
Functions such as `overlap` and `pos` take the body.

### Large mechanisms

To decide a claim, the prover needs only the bodies that the claim mentions and whatever
constrains them. It reduces the mechanism in three steps.

1. Independent groups. Bodies that are connected to each other only through the fixed world
   move independently. The prover keeps only the groups that contain bodies mentioned in the
   claim. This step is exact.
2. Dangling bodies. A body that hangs on a single joint, with nothing else attached to it,
   can always follow the rest of the mechanism. The prover removes such bodies until none is
   left. This step is also exact.
3. Neighbourhoods. If a group still has more than `--max-bodies` bodies (12 by default), the
   prover works with the bodies within 0, 1, 2, … joints of the mentioned ones. It tries two
   versions of each neighbourhood.
   - In the relaxed version, the constraints to the rest of the mechanism are dropped. This
     allows more configurations than the real mechanism has. A proof of an `always` claim
     therefore still holds for the whole mechanism, and so does a refutation of a `possible`
     claim.
   - In the restricted version, the joints of the rest of the mechanism are held at zero.
     This allows fewer configurations. A configuration found for a `possible` claim is
     therefore real, and so is a counterexample to an `always` claim.
   - Any other result is inconclusive, and the prover tries a larger neighbourhood.

`examples/make_snake.py` generates a chain of 1000 links with eight local claims about links
in the middle. Seven of them are decided in about 30 s in total. The eighth, `staysInPlane`,
is true but gets the answer `unknown`, because a proof has to follow the chain all the way to
the world.

### Limits of the prover

- Claims about the whole of a large connected mechanism usually cannot be decided. Real
  arithmetic with quantifiers is decidable, but the cost can grow doubly exponentially with
  the number of variables.
- Constants must be algebraic numbers. `sqrt(2)` and `cos(pi/12)` are allowed. `pi` itself is
  not allowed, and neither is the `--trig` mode.
- The prover trusts z3's answers and does not check them independently.
- Each solver call uses its own z3 context. Before this change, the time z3 needed for a
  claim depended on which claims had been checked before it in the same run.

## Coordinates

- An angle is stored as a pair `(c, s)` with `c^2 + s^2 == 1`. This keeps all equations
  polynomial.
- A variable is named after its role and its joint. The hinge `elbow` has `cElbow` and
  `sElbow`, the slider `slide` has `dSlide`, a ball joint `ball` has `q0Ball` to `q3Ball`, and
  a free body `puck` has `xPuck` and so on.
- The default mode is called reduced. cspace builds a spanning tree of joints starting from
  `world`, in file order, and the joints in the tree get coordinates. Every other constraint
  becomes equations on the resulting poses, which close the loops. A body that no tree joint
  reaches gets a free joint, with a position and a unit quaternion.
- With `--coords maximal`, every body gets a position and a unit quaternion, and every
  constraint is an equation between two poses.
- With `--trig`, hinges in the tree get an angle variable such as `thElbow`, and the equations
  use `Cos` and `Sin` in place of `(c, s)`. This mode is required for couplings between an
  angle and a length, such as screws and rack-and-pinion drives, and for ranges wider than
  one turn.
- Equations are reduced modulo the circle and quaternion relations. Their rational content is
  removed, and duplicates are dropped.

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

All numbers are exact. `0.1` means 1/10, and `pi/3` stays symbolic. A reference is written
`body.frame`, or just `body` for the body's own frame. Names consist of letters and digits.
The names `x`, `y`, `z`, `pi`, `sqrt`, `sin`, `cos` and `tan` are reserved.

### Pairwise constraints

Each of these constrains the pose of frame 2 relative to frame 1. The axis of a joint is the
z axis of its frames.

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

The last four are the implicit constraints. They have no coordinates of their own, so they
always close loops.

A constraint can have these options.

- `name:` is used in variable names and in couplings. The default names are `j1`, `j2`, and
  so on.
- `range: [lo, hi]` limits the joint's main coordinate. This is the angle, or the
  displacement for `prismatic`.
- `limits: {angle: [..], d: [..], angle1: [..], angle2: [..]}` sets limits on joints with
  several coordinates.
- `cone: A` applies to ball joints. It keeps frame 2's z axis within the angle A of frame 1's
  z axis.
- `closing: true` keeps a joint out of the spanning tree.

### Couplings

A coupling relates the coordinates of any number of joints.

```yaml
- gear: [pinion, wheel]
  teeth: [20, 40]          # external mesh; internal: true for a ring gear; phase: angle
- couple: {h1: 1, h2: -2}  # sum of coefficient * coordinate = equals
  equals: 0
- couple: {slide.d: 1, crank.angle: -pitch/(2*pi)}   # needs --trig
```

An angle coupling `Σ a_i θ_i = φ` with rational `a_i` becomes `Π z_i^(a_i) = e^(iφ)`, where
`z = c + i s` and the `a_i` are scaled to coprime integers. The solutions of this equation
are the same poses that the original relation allows. A coupling may use the coordinate of a
joint that closes a loop.

### Shapes and `no_overlap`

The basic shapes are:

- `box: [lx, ly, lz]`, centred
- `ball: r`
- `cylinder: [r, h]`, centred, with its axis along z
- `hull: [[x, y, z], …]`, with numeric vertices
- `where: ["x^2 + y^2 <= 1", "z >= 0"]`, given by polynomial inequalities

Shapes can be combined with `union`, `intersection` and `difference`. A `difference` is the
first child minus the others. Any shape can be placed with `at`, `z` and `x`.

`no_overlap: [A, B]` states that no point is in the interior of both bodies.

- For unions of boxes, hulls and balls, cspace writes an exact formula without quantifiers.
  It uses separating axes for pairs of polytopes and the distance between centres for pairs
  of balls.
- For every other case, including a ball against a polytope, it writes an exact condition
  that is quantified over all points. The prover works with this condition directly.

## Examples

| File | Shows |
|---|---|
| `two_rods` | two rods joined by a hinge, with symbolic lengths and a limited hinge range |
| `fourbar` | a planar closed loop, where the simple count gives −2 and the true dimension is 1 |
| `slider_crank` | a prismatic joint with a limited displacement |
| `bennett` | an overconstrained spatial loop that still moves |
| `rssr` | ball joints, a cone limit, and an idle degree of freedom (dimension 2) |
| `gears` | a gear coupling between two hinges |
| `screw` | a coupling between an angle and a length (needs `--trig`) |
| `tethered_puck` | implicit constraints on a free body |
| `sliding_boxes` | the exact condition for two boxes (\|s\| ≥ 2), and a quantified condition as the fallback |
| `door_stop` | a collision that limits a hinge to part of its circle |
| `stewart` | a Stewart platform with 14 bodies, six closed spatial loops, 37 variables and 6 freedoms |
| `cube_on_cube` | claims that the cubes are never convex and never overhang, and one claim that is deliberately false |
| `make_snake.py` | generates a chain of N links with local claims, as a test of scaling |
| `ratchet_drive` | springs, contacts and a start pose for MuJoCo (generated by `make_ratchet_drive.py`) |
| `helixturm` | the self-building tower, with 50 bodies, 19 of them free and moved only by contact (generated by `make_helixturm.py`, run with `run_helixturm.py`) |
| `helixturm_lock` | only the tower's joints, with claims about how they lock |

## Constraints on more than two bodies

Every joint, joint range and collision condition involves two bodies. It is a condition on
the pose of one body relative to the other. Couplings are different. A gear ratio relates two
joint angles, and those two joints involve three or four bodies. Differentials and cable runs
involve more. For this reason `couple` and `gear` accept any number of joints.

The configuration space is the set of poses that satisfy all constraints at once. Its
dimension depends on the whole mechanism and cannot be found by counting constraints one at
a time. The `--check` output described above shows this.

## General limits

- Gröbner bases are computed over the complex numbers. The dimension of the real solution set
  can be lower.
- Unit quaternions represent each rotation twice.
- Couplings between an angle and a length are not algebraic, so they need `--trig`.
- Rolling without slipping in 3D is a constraint on velocities and not on poses, so it cannot
  be expressed.
- The vertices of a `hull` must be numbers. The quantified formulas used as a fallback are
  exact, but deciding them can be slow.
- Strings in a mechanism file are evaluated by SymPy's parser. Only compile files that you
  trust.
- These are not supported: importing meshes, the rolling law for two gear bodies, cams as a
  constraint, and a quantifier-free test for a ball against a polytope.

## Tests

The pytest installed here is version 5.2. Under Python 3.10 it runs only with plugin
autoloading turned off.

```
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q -p no:cacheprovider --assert=plain tests/
```

`tests/test_helixturm.py` takes about a minute. Its MuJoCo test, which runs two steps of a
small tower, is skipped if MuJoCo is not installed. Its GIF test is also skipped if Pillow,
ffmpeg or an OpenGL context is missing. There is no automated test of the viewer's WebGL
drawing. It was checked by eye in headless Chromium against the 2D fallback.
