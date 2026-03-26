# Shared-Protocol Porting Disclosure

| Baseline | Official idea retained | Unified-harness changes | Why comparison remains informative |
|---|---|---|---|
| `BiTTA-style` | Queried binary feedback drives positive and negative adaptation decisions. | Implemented inside the shared CIFAR streaming harness with the same source checkpoints, windows, query-rate grid, and reporting pipeline; no separate official memory runner is invoked. | Tests whether binary feedback remains useful under the same backbone, stream, and budget protocol as the proposed gate. |
| `ATTA-style` | Budgeted queried labels are used during test-time adaptation. | Ported to the shared harness with the same stream API, query grid, and evaluation metrics rather than an external active-learning stack. | Provides a stronger queried-label reference under the same target streams and query accounting. |
| `EATTA-style` | Low-label updates are combined with reliable unlabeled samples. | Reliability heuristics are adapted to the local harness while keeping the same backbone family, windows, and query budgets. | Clarifies what richer low-label supervision can achieve under the same deployment protocol. |
| `COME-style` | Conservative label-free entropy/stability control is retained as a 0-bit baseline. | Implemented in the same batch-normalization adaptation harness and source-model family as Tent and the binary gate. | Gives a matched conservative 0-bit comparator under the same online adaptation pipeline. |

Note:
- These are shared-protocol ports, not claims of exact end-to-end reproduction of each official runner.
- The comparison target is behavior under a common protocol, not leaderboard reproduction.
