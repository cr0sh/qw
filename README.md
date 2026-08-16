# QuasarWave

QW is an inference runtime targeting MLX-only, single-user
deployments.

QW aims to be explicitly "focused", to achieve these goals below:

- Fixed model: Qwen3.8 27B(dense model) only. No generalization over
  different model structures.
- Fixed environment: MLX only. No generalization over CUDA, ROCm, ...
- Highest single stream TPS: The constraints above exists to enable fastest
  Qwen3.5 family implementation in the world.
