"""Write a large test mechanism: a planar chain ("snake") of N links on hinges.

    python3 examples/make_snake.py 1000 > snake.yaml
    python3 -m cspace snake.yaml --prove

Each link is a 1 x 1/10 x 1/10 bar. Link 1 hinges on the world; link i+1 hinges on the far
end of link i, with each hinge limited to +-60 degrees. The claims are about links in the
middle of the chain.
"""
import sys


def snake(n, mid=None):
    mid = mid or n // 2
    lines = [f"name: snake{n}", "bodies:"]
    for i in range(1, n + 1):
        lines.append(f"  link{i}: {{shape: {{box: [1, 1/10, 1/10], at: [1/2, 0, 0]}}, "
                     f"frames: {{a: {{}}, b: {{at: [1, 0, 0]}}}}}}")
    lines.append("constraints:")
    lines.append("  - revolute: [world, link1.a]\n    name: h1\n    range: [-pi/3, pi/3]")
    for i in range(1, n):
        lines.append(f"  - revolute: [link{i}.b, link{i + 1}.a]\n    name: h{i + 1}\n"
                     f"    range: [-pi/3, pi/3]")
    a, b = mid, mid + 2
    # pos(link_i.a) is the hinge at the start of link i; link{a}.a and link{b}.a are the two ends
    # of links {a} and {a}+1, so their squared distance is 2 + 2 cos(bend), between 3 and 4.
    lines += ["claims:",
              f"  - name: linksTwoApartWithinReach      # triangle inequality",
              f"    always: dist2(pos(link{a}.a), pos(link{b}.a)) <= 4",
              f"  - name: bendLimitsTheGap               # hinge at most 60 degrees: squared gap >= 3",
              f"    always: dist2(pos(link{a}.a), pos(link{b}.a)) >= 3",
              f"  - name: canStraighten",
              f"    possible: dist2(pos(link{a}.a), pos(link{b}.a)) == 4",
              f"  - name: canBendFully",
              f"    possible: dist2(pos(link{a}.a), pos(link{b}.a)) == 3",
              f"  - name: canFoldTight                   # false: needs a bend over 60 degrees",
              f"    possible: dist2(pos(link{a}.a), pos(link{b}.a)) < 3",
              f"  - name: linksTwoApartNeverOverlap",
              f"    always: not overlap(link{a}, link{b})",
              f"  - name: staysInPlane                   # true, but needs the whole chain to the world",
              f"    always: pos(link{a}.b).z == 0",
              f"  - name: neverBendsLeft                 # false: refuted with a real configuration",
              f"    always: h{b}.s <= 0"]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    sys.stdout.write(snake(int(sys.argv[1]) if len(sys.argv) > 1 else 1000))
