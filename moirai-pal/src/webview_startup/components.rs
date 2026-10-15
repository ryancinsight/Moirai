//! Closed real-observer compositions for causal native fixture comparisons.

use serde::Deserialize;

#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Components {
    #[default]
    Full,
    Controller,
    Startup,
}

impl Components {
    pub(crate) const fn controller(self) -> bool {
        matches!(self, Self::Full | Self::Controller)
    }

    pub(crate) const fn startup(self) -> bool {
        matches!(self, Self::Full | Self::Startup)
    }

    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Full => "full",
            Self::Controller => "controller",
            Self::Startup => "startup",
        }
    }
}
