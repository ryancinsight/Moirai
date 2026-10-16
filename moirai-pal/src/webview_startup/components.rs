//! Closed real-observer compositions for causal native fixture comparisons.

use serde::Deserialize;

#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Components {
    #[default]
    Full,
    Controller,
    Native,
    Events,
    Domain,
    Snapshot,
    Startup,
}

impl Components {
    pub(crate) const fn controller(self) -> bool {
        matches!(self, Self::Full | Self::Controller)
    }

    pub(crate) const fn native(self) -> bool {
        matches!(
            self,
            Self::Full
                | Self::Startup
                | Self::Native
                | Self::Events
                | Self::Domain
                | Self::Snapshot
        )
    }

    pub(crate) const fn cdp(self) -> bool {
        matches!(
            self,
            Self::Full | Self::Startup | Self::Events | Self::Domain | Self::Snapshot
        )
    }

    pub(crate) const fn enables_dom(self) -> bool {
        matches!(self, Self::Domain)
    }

    pub(crate) const fn admits_sample(self, completed: u64) -> bool {
        match self {
            Self::Full | Self::Startup => true,
            Self::Snapshot => completed == 0,
            Self::Controller | Self::Native | Self::Events | Self::Domain => false,
        }
    }

    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Full => "full",
            Self::Controller => "controller",
            Self::Native => "native",
            Self::Events => "events",
            Self::Domain => "domain",
            Self::Snapshot => "snapshot",
            Self::Startup => "startup",
        }
    }
}
