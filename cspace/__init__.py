"""cspace: compile rigid bodies and joint constraints into an algebraic configuration space."""
from .compile import ALL_KINDS, System, compile_spec
from .export import summary, to_json
from .spec import SpecError, load


def build(path, coords="reduced", trig=False, overrides=None):
    """Load a mechanism file and compile it in one step."""
    return compile_spec(load(path, ALL_KINDS, overrides), coords, trig)
