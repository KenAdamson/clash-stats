"""Every trainer must carry the same NaN guards.

This exists because the same defect has now shipped three times, and each time
the fix was applied to the trainer that happened to fail rather than to all of
them:

  - wp_v4 went non-finite at epoch 11. The WP trainer was hardened.
  - tcn_v2 went non-finite at epoch 9, on the same dataset and the same GPU,
    because none of that hardening had been applied to the TCN trainer. Twelve
    days of a 1.89M-game run's worth of risk, discovered the expensive way.
  - The CVAE trainer was then found to have clip_grad_norm_ but to DISCARD its
    return value, which is worse than not clipping: a non-finite pre-clip norm
    makes clip_coef NaN and clipping then writes NaN into every parameter.

The bug each time was a sample; the population was "all trainers". A prose rule
did not prevent recurrence -- the repository already documented the failure --
so this asserts the invariant mechanically instead. A new trainer that steps an
optimizer without these guards fails here rather than at hour nine of a run.

Deliberately a source-level check, not a behavioural one. Provoking a real NaN
requires a GPU, a corpus, and hours; the guards are simple enough that their
presence is the property worth enforcing, and a source check costs milliseconds
and needs no fixtures.
"""

import ast
from pathlib import Path

import pytest

ML_DIR = Path(__file__).resolve().parents[1] / "ml"


def _trainer_modules() -> list[Path]:
    """Every module that steps an optimizer, discovered rather than listed.

    A hardcoded list would go stale the moment someone adds a trainer, which is
    exactly the failure mode this file exists to prevent.
    """
    found = []
    for path in sorted(ML_DIR.glob("*.py")):
        src = path.read_text()
        if ".step()" in src and "optimizer" in src and "backward()" in src:
            found.append(path)
    return found


def test_trainers_were_discovered():
    """Guard the guard: if discovery silently matches nothing, every other test
    in this file passes vacuously."""
    mods = _trainer_modules()
    assert len(mods) >= 3, (
        "expected at least the WP, TCN and CVAE trainers, found: %s"
        % [m.name for m in mods])


@pytest.mark.parametrize("path", _trainer_modules(), ids=lambda p: p.name)
def test_trainer_clips_gradients(path: Path):
    src = path.read_text()
    assert "clip_grad_norm_" in src, (
        "%s steps an optimizer without clipping gradients. An unclipped runaway "
        "update is what turned tcn_v2's weights to NaN at epoch 9." % path.name)


@pytest.mark.parametrize("path", _trainer_modules(), ids=lambda p: p.name)
def test_trainer_inspects_clip_norm(path: Path):
    """clip_grad_norm_'s return value must be checked, not discarded.

    The CVAE trainer discarded it. If the pre-clip norm is non-finite the clip
    coefficient becomes NaN, so clipping propagates the corruption into every
    parameter instead of bounding it -- the guard inverted into its opposite.
    """
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Expr):
            continue
        call = node.value
        if isinstance(call, ast.Call) and getattr(call.func, "attr", "") == "clip_grad_norm_":
            raise AssertionError(
                "%s calls clip_grad_norm_ as a bare statement, discarding the "
                "pre-clip norm. Assign it and skip the step when it is "
                "non-finite." % path.name)


@pytest.mark.parametrize("path", _trainer_modules(), ids=lambda p: p.name)
def test_trainer_checks_loss_finite(path: Path):
    src = path.read_text()
    assert "isfinite" in src, (
        "%s never tests torch.isfinite. A non-finite loss must be caught BEFORE "
        "backward(), while the weights are still clean." % path.name)


@pytest.mark.parametrize("path", _trainer_modules(), ids=lambda p: p.name)
def test_trainer_sets_adam_eps(path: Path):
    """Adam's default eps of 1e-8 lets a near-zero second-moment estimate blow an
    update up via lr*m/(sqrt(v)+eps). Every trainer raises the floor."""
    src = path.read_text()
    assert "eps=ADAM_EPS" in src or "eps=" in src, (
        "%s constructs an optimizer without an explicit eps." % path.name)


@pytest.mark.parametrize("path", _trainer_modules(), ids=lambda p: p.name)
def test_trainer_refuses_to_save_corrupt_weights(path: Path):
    """A saved NaN checkpoint destroys the only rollback target there is.

    tcn_v2's epoch-8 weights survived its NaN only by accident of IEEE 754 --
    the save is gated on val_loss < best and every comparison against NaN is
    false. That is the correct outcome reached for the wrong reason.
    """
    src = path.read_text()
    if "torch.save" not in src:
        pytest.skip("%s does not persist checkpoints" % path.name)
    assert "Refusing to save" in src or "isfinite" in src, (
        "%s can persist a checkpoint without verifying the weights are finite."
        % path.name)
