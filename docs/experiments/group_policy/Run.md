# DSPO game pilot

This branch selects DSPO. Assigned machine: AutoDL RTX 3080 Ti.

The exact configuration is in configs/group_policy/train.json. The common
algorithm and game-adaptation contract is in Plan.md. Initialization is the
shared SHA256-pinned random checkpoint. The pilot stops at 1,048,576 sampled
decisions or 7,200 cumulative seconds. Rejected samples count toward the budget.
