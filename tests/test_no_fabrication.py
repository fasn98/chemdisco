"""The guard against fabricated numbers.

The predecessor project's central defect was not a bug anyone wrote on purpose.
It was drift: a plausible-looking scoring heuristic, then a line of noise added
to "simulate experimental variation", then a synthetic label so a model could be
trained when real data was thin, then reference compounds injected to make a
small dataset trainable. Each step was locally reasonable. The end state was a
platform reporting invented binding affinities to two decimal places.

Code review does not reliably catch drift, because each individual step looks
fine. A test does. This module walks the package's syntax tree and fails if a
module that produces or transforms a reported scientific value can reach a
random number generator at all.

Randomness has legitimate uses here -- shuffling a split, bootstrapping a
confidence interval, initialising a model. Those live in named modules on an
allowlist, each with a written justification. Adding a module to that allowlist
is a deliberate, reviewable act; adding ``import random`` to a scoring module is
a build failure.
"""

from __future__ import annotations

import ast
import pathlib
import unittest

PACKAGE_ROOT = pathlib.Path(__file__).resolve().parent.parent / "chemdisco"

#: Modules permitted to use a random number generator, each with the reason.
#: A path is matched as a POSIX-style suffix relative to the package root.
RANDOMNESS_ALLOWLIST: dict[str, str] = {
    "split/scaffold.py": (
        "random_split shuffles dataset indices to build the optimistic baseline "
        "comparison. The randomness selects which molecules go where; it never "
        "touches a measured or predicted value."
    ),
    "qsar/baselines.py": (
        "a permutation baseline shuffles labels to establish the score a model "
        "achieves by chance. The shuffling is the experiment."
    ),
    "qsar/evaluate.py": (
        "bootstrap resampling for confidence intervals. Resampling observed "
        "values is a standard estimator of uncertainty, not an invention of data."
    ),
    "generate/brics.py": (
        "random fragment recombination explores the enumeration space when "
        "exhaustive enumeration is intractable. It generates candidate "
        "structures to be evaluated, never evaluation results."
    ),
}

#: Names that reach a random number generator.
_RANDOM_MODULES = {"random", "numpy.random", "np.random", "secrets"}


def _iter_source_files() -> list[pathlib.Path]:
    return sorted(
        path
        for path in PACKAGE_ROOT.rglob("*.py")
        if "__pycache__" not in path.parts
    )


def _relative(path: pathlib.Path) -> str:
    return path.relative_to(PACKAGE_ROOT).as_posix()


def _is_allowlisted(relative_path: str) -> bool:
    return relative_path in RANDOMNESS_ALLOWLIST


class RandomnessVisitor(ast.NodeVisitor):
    """Collects every route to a random number generator in one module."""

    def __init__(self) -> None:
        self.findings: list[str] = []

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            root = alias.name.split(".")[0]
            if alias.name in _RANDOM_MODULES or root in _RANDOM_MODULES:
                self.findings.append(f"line {node.lineno}: import {alias.name}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        root = module.split(".")[0]
        if module in _RANDOM_MODULES or root in _RANDOM_MODULES:
            names = ", ".join(alias.name for alias in node.names)
            self.findings.append(f"line {node.lineno}: from {module} import {names}")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        # Catches np.random.normal(...) and numpy.random.uniform(...) even when
        # numpy is imported under an alias, which a plain import check misses.
        parts: list[str] = []
        current: ast.expr = node
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.append(current.id)
        dotted = ".".join(reversed(parts))
        if ".random." in f".{dotted}." or dotted.endswith(".random"):
            self.findings.append(f"line {node.lineno}: {dotted}")
        self.generic_visit(node)


class TestNoFabricatedValues(unittest.TestCase):
    def test_package_exists_and_is_being_scanned(self) -> None:
        files = _iter_source_files()
        self.assertGreater(
            len(files), 5, "the guard is not finding the package; it would pass vacuously"
        )

    def test_no_randomness_outside_the_allowlist(self) -> None:
        violations: list[str] = []
        for path in _iter_source_files():
            relative = _relative(path)
            if _is_allowlisted(relative):
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            visitor = RandomnessVisitor()
            visitor.visit(tree)
            for finding in visitor.findings:
                violations.append(f"{relative} {finding}")

        self.assertEqual(
            violations,
            [],
            "Randomness reached a module that reports scientific values.\n"
            "If the use is legitimate -- shuffling, resampling, seeding -- add "
            "the module to RANDOMNESS_ALLOWLIST in this test with a written "
            "justification. If it is noise added to a computed result, it is "
            "the defect this project was built to remove.\n"
            + "\n".join(f"  {violation}" for violation in violations),
        )

    def test_allowlist_has_no_stale_entries(self) -> None:
        # A stale entry silently widens the guard: the file may be renamed while
        # the permission lingers, ready to cover a future module at that path.
        existing = {_relative(path) for path in _iter_source_files()}
        stale = sorted(set(RANDOMNESS_ALLOWLIST) - existing)
        self.assertEqual(
            stale,
            [],
            f"allowlist entries point at files that do not exist: {stale}",
        )

    def test_every_allowlist_entry_is_justified(self) -> None:
        for path, reason in RANDOMNESS_ALLOWLIST.items():
            with self.subTest(path=path):
                self.assertGreater(
                    len(reason.strip()),
                    40,
                    f"{path} needs a real justification, not a placeholder",
                )

    def test_allowlisted_modules_actually_use_randomness(self) -> None:
        # An allowlist entry for a module that does not use randomness is a
        # permission granted for nothing, which future code can quietly inherit.
        for relative in RANDOMNESS_ALLOWLIST:
            path = PACKAGE_ROOT / relative
            if not path.exists():
                continue  # covered by test_allowlist_has_no_stale_entries
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            visitor = RandomnessVisitor()
            visitor.visit(tree)
            with self.subTest(path=relative):
                self.assertTrue(
                    visitor.findings,
                    f"{relative} is allowlisted for randomness but uses none; "
                    "remove the entry",
                )


class TestNoSilentDefaults(unittest.TestCase):
    """Catch the other half of the fabrication pattern: imputed fallbacks.

    The predecessor also returned mid-range defaults when a calculation failed
    -- ``props.get('molecular_weight', 300)``, ``props.get('logp', 2.0)`` -- so a
    molecule whose descriptors could not be computed entered the model as a
    typical drug. That is fabrication by default value, and it is harder to spot
    than added noise because it looks like defensive programming.
    """

    #: Descriptor-like keys that must never be given a numeric fallback.
    _SUSPECT_KEYS = {
        "molecular_weight",
        "mw",
        "logp",
        "alogp",
        "tpsa",
        "hbd",
        "hba",
        "rotatable_bonds",
        "aromatic_rings",
        "binding_affinity",
        "pactivity",
        "pchembl_value",
        "activity_value",
        "sp3_fraction",
        "num_heavy_atoms",
    }

    def test_no_numeric_fallback_for_descriptor_lookups(self) -> None:
        violations: list[str] = []
        for path in _iter_source_files():
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                if not (isinstance(func, ast.Attribute) and func.attr == "get"):
                    continue
                if len(node.args) != 2:
                    continue
                key, default = node.args
                if not (isinstance(key, ast.Constant) and isinstance(key.value, str)):
                    continue
                if key.value not in self._SUSPECT_KEYS:
                    continue
                if isinstance(default, ast.Constant) and isinstance(
                    default.value, (int, float)
                ) and not isinstance(default.value, bool):
                    violations.append(
                        f"{_relative(path)} line {node.lineno}: "
                        f".get({key.value!r}, {default.value!r})"
                    )

        self.assertEqual(
            violations,
            [],
            "A descriptor lookup supplies a numeric default, so a molecule with "
            "no computed value enters the pipeline wearing a typical drug's "
            "properties. Return None and let the caller handle the absence.\n"
            + "\n".join(f"  {violation}" for violation in violations),
        )


if __name__ == "__main__":
    unittest.main()
