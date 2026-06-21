use std::collections::HashMap;

/// Load balancer for distributed tasks
pub struct LoadBalancer {
    node_loads: HashMap<usize, f64>,
}

impl LoadBalancer {
    pub(super) fn new() -> Self {
        Self {
            node_loads: HashMap::new(),
        }
    }
}
