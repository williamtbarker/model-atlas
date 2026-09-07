# Development roadmap

## 1. Make the computation legible and visually compelling

Refine composition, spacing, typography, color, camera transitions, and operation
animations through direct browser review on an ordinary MacBook. Add a direct
source-to-residual view linking attention coefficients to signed output writes.
Keep every animation grounded in inspectable numbers.

Done when a viewer can follow one token's selected contribution through a block,
understand reinforcement and cancellation, and compare an intervention without
losing spatial orientation. Capture a real screen recording for the README.

## 2. Unify the whole DeepSeek architecture

Use the existing structural package to create a persistent hierarchy from model
overview through blocks, attention variants, hyper-connections, routers,
experts, and tensor coordinates. Refine camera-local structures under hard
geometry and label budgets. Replace scope changes with continuous navigation.

Done when every declared component is reachable while the overview remains
coherent and the active scene stays bounded. Measure memory and frame time on
the target laptop; do not infer performance from object counts alone.

## 3. Add evidence for large-model computation

Export small, explicitly selected recordings and interventions offline, starting
with a pretrained GPT-2 reference. Extend operation schemas for each actual
architecture before displaying its numerical traces. Stream trace chunks on
demand and distinguish recorded values, derived values, and unavailable values.

Done when numerical reconstructions agree with the source execution and the
browser can inspect selected real-model operations without downloading full
weights or running local inference.

“Every parameter is addressable” means a coordinate can be located. Reading its
trained scalar value additionally requires a value source. It does not require
one simultaneously rendered object per parameter.
