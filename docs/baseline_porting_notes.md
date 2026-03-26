# Baseline Porting Notes

This repository uses a unified local evaluation protocol for all methods:

- same CIFAR-style ResNet-18 family
- same source-training seeds
- same stream windowing
- same query-rate grid
- same contamination grid
- same metrics and result writing

Because the official external repositories target different datasets, backbones, and runners, several baselines were ported into the local harness rather than executed end to end from the original codebases.

## Tent

- Status: local unified-harness implementation.
- Source reference: `external/tent/tent-master/`.
- Deviation: adapted to the local CIFAR backbone and stream runner; behavior follows BN-only entropy minimization.

## COME-style zero-bit stable baseline

- Status: local unified-harness port.
- Source reference: `external/COME/COME-main/`.
- Main preserved idea: replace softmax entropy with a conservative entropy-of-opinion style objective.
- Main deviation: the artifact uses the local BN-only CIFAR stream harness rather than the official ImageNet-focused runner.

## BiTTA-style binary feedback baseline

- Status: local unified-harness port.
- Source reference: `external/BiTTA/BiTTA-main/`.
- Main preserved idea: binary queried feedback drives positive versus negative supervision.
- Main deviation: the local port uses the same source checkpoints and streaming protocol as the rest of this artifact, instead of the original training and evaluation environment.

## ATTA-style active baseline

- Status: local unified-harness port.
- Source reference: `external/ATTA/ATTA-main/`.
- Main preserved idea: actively query a small labeled subset and update online under the same budget.
- Main deviation: the artifact uses a simplified active-label update inside the CIFAR stream harness, not the full official SimATTA training stack.

## EATTA-style low-label baseline

- Status: local unified-harness port.
- Source reference: `external/EATTA/EATTA-main/`.
- Main preserved idea: use very sparse labels with an online active-update rule.
- Main deviation: the artifact uses a simplified local implementation aligned to the shared CIFAR protocol rather than the official long-term test-time adaptation runner.

## Full-label active reference

- Status: local reference baseline.
- Purpose: provide a strong supervised control inside the same query-budgeted stream protocol.
- Note: this is not claimed as an official ATTA or EATTA implementation.

## Matched-conservatism controls

- `self_gate_tent_zero_bit`: newly introduced in this artifact to test whether internal confidence changes can explain the gains without external bits.
- `throttle_tent_zero_bit`: newly introduced in this artifact to test whether a conservative update-throttling heuristic can match the 1-bit veto.

These controls are not ports of external baselines; they are explicit artifact contributions for reviewer-facing falsification.
