### Changed
- `moirai-gpu`: made the concrete WGPU execution stack optional behind the
  `wgpu-backend` feature while keeping `occupancy` launch planning available
  with `default-features = false`. Atlas GPU backends can consume Moirai
  launch planning without inheriting Moirai's concrete WGPU runtime version.

