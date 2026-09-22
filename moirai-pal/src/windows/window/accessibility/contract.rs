//! Validated, host-neutral accessibility semantics for native windows.

use accesskit::{Action, Node, NodeId, Role, TreeId, TreeInfo, TreeUpdate};
use std::collections::{HashMap, HashSet};
use std::io;

/// Maximum number of nodes retained by one native accessibility tree.
pub const MAX_ACCESSIBILITY_NODES: usize = 4_096;
/// Maximum UTF-8 bytes retained by one accessibility string.
pub const MAX_ACCESSIBILITY_TEXT_BYTES: usize = 4_096;
/// Maximum queued accessibility actions awaiting the window thread.
pub const MAX_ACCESSIBILITY_ACTIONS: usize = 256;

/// Role exposed by the format-neutral native accessibility contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum AccessibilityRole {
    /// The application root.
    Application,
    /// A primary document region.
    Main,
    /// A navigation landmark containing links or commands.
    Navigation,
    /// A complementary region such as a sidebar.
    Complementary,
    /// A generic grouping container.
    Group,
    /// A command toolbar or title bar.
    Toolbar,
    /// A popup command menu.
    Menu,
    /// An actionable command inside a menu.
    MenuItem,
    /// An actionable button.
    Button,
    /// Static text.
    Text,
    /// An editable text control.
    TextInput,
    /// A binary choice control.
    CheckBox,
    /// A mutually exclusive choice control.
    Radio,
    /// A bounded numeric value control.
    Slider,
    /// A list or combobox selection control.
    ComboBox,
    /// A dialog surface.
    Dialog,
    /// A status or live-region message.
    Status,
    /// A table or grid surface.
    Table,
}

/// Action admitted by one native accessibility node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum AccessibilityAction {
    /// Activate the node's primary command.
    Activate,
    /// Move keyboard focus to the node.
    Focus,
    /// Replace the node's current value.
    SetValue,
    /// Toggle a binary choice.
    Toggle,
    /// Adjust a bounded numeric value.
    AdjustValue,
    /// Open a selection control.
    Open,
}

/// One validated node in a native accessibility tree.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AccessibilityNode {
    id: u64,
    role: AccessibilityRole,
    name: String,
    description: Option<String>,
    value: Option<String>,
    children: Vec<u64>,
    hidden: bool,
    disabled: bool,
    focusable: bool,
    expanded: Option<bool>,
    selected: Option<bool>,
    checked: Option<bool>,
    actions: Vec<AccessibilityAction>,
}

impl AccessibilityNode {
    /// Creates a node with a nonzero stable identity and bounded name.
    pub fn new(id: u64, role: AccessibilityRole, name: impl Into<String>) -> io::Result<Self> {
        let name = name.into();
        validate_id(id)?;
        validate_text(&name)?;
        Ok(Self {
            id,
            role,
            name,
            description: None,
            value: None,
            children: Vec::new(),
            hidden: false,
            disabled: false,
            focusable: false,
            expanded: None,
            selected: None,
            checked: None,
            actions: Vec::new(),
        })
    }

    /// Replaces the optional accessible description.
    pub fn set_description(&mut self, description: Option<String>) -> io::Result<()> {
        if let Some(value) = &description {
            validate_text(value)?;
        }
        self.description = description;
        Ok(())
    }

    /// Replaces the optional value exposed to assistive technology.
    pub fn set_value(&mut self, value: Option<String>) -> io::Result<()> {
        if let Some(value) = &value {
            validate_text(value)?;
        }
        self.value = value;
        Ok(())
    }

    /// Replaces the ordered child identities.
    pub fn set_children(&mut self, children: Vec<u64>) -> io::Result<()> {
        if children.len() > MAX_ACCESSIBILITY_NODES {
            return Err(limit_error("accessibility child count exceeds its bound"));
        }
        for child in &children {
            validate_id(*child)?;
        }
        self.children = children;
        Ok(())
    }

    /// Sets whether the node is hidden from assistive technology.
    pub const fn set_hidden(&mut self, hidden: bool) {
        self.hidden = hidden;
    }

    /// Sets whether the node rejects host actions.
    pub const fn set_disabled(&mut self, disabled: bool) {
        self.disabled = disabled;
    }

    /// Sets whether keyboard focus may land on the node.
    pub const fn set_focusable(&mut self, focusable: bool) {
        self.focusable = focusable;
    }

    /// Sets the optional expanded state.
    pub const fn set_expanded(&mut self, expanded: Option<bool>) {
        self.expanded = expanded;
    }

    /// Sets the optional selected state.
    pub const fn set_selected(&mut self, selected: Option<bool>) {
        self.selected = selected;
    }

    /// Sets the optional checked state.
    pub const fn set_checked(&mut self, checked: Option<bool>) {
        self.checked = checked;
    }

    /// Adds one action if it is not already present.
    pub fn add_action(&mut self, action: AccessibilityAction) {
        if !self.actions.contains(&action) {
            self.actions.push(action);
        }
    }
}

/// A validated, connected accessibility tree.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AccessibilityTree {
    root: u64,
    focus: u64,
    nodes: Vec<AccessibilityNode>,
}

impl AccessibilityTree {
    /// Validates a tree assembled from its complete node set.
    pub fn from_nodes(root: u64, focus: u64, nodes: Vec<AccessibilityNode>) -> io::Result<Self> {
        let tree = Self { root, focus, nodes };
        tree.validate()?;
        Ok(tree)
    }

    pub(super) fn validate(&self) -> io::Result<()> {
        if self.nodes.is_empty() || self.nodes.len() > MAX_ACCESSIBILITY_NODES {
            return Err(limit_error("accessibility node count is outside its bound"));
        }
        validate_id(self.root)?;
        validate_id(self.focus)?;
        let mut by_id = HashMap::with_capacity(self.nodes.len());
        for node in &self.nodes {
            validate_id(node.id)?;
            validate_text(&node.name)?;
            if let Some(description) = &node.description {
                validate_text(description)?;
            }
            if let Some(value) = &node.value {
                validate_text(value)?;
            }
            if (node.hidden || node.disabled) && (!node.actions.is_empty() || node.focusable) {
                return Err(invalid_error(
                    "hidden or disabled accessibility nodes cannot expose focus or actions",
                ));
            }
            if by_id.insert(node.id, node).is_some() {
                return Err(invalid_error(
                    "accessibility node identities must be unique",
                ));
            }
        }
        let root_node = by_id
            .get(&self.root)
            .ok_or_else(|| invalid_error("accessibility root is missing"))?;
        if self.focus != self.root {
            let focus_node = by_id
                .get(&self.focus)
                .ok_or_else(|| invalid_error("accessibility focus node is missing"))?;
            if focus_node.hidden || focus_node.disabled || !focus_node.focusable {
                return Err(invalid_error(
                    "accessibility focus must identify an enabled focusable node",
                ));
            }
        }

        let mut parents = HashSet::with_capacity(self.nodes.len());
        let mut visited = HashSet::with_capacity(self.nodes.len());
        let mut pending = vec![self.root];
        while let Some(id) = pending.pop() {
            if !visited.insert(id) {
                return Err(invalid_error("accessibility tree contains a cycle"));
            }
            let node = by_id
                .get(&id)
                .ok_or_else(|| invalid_error("accessibility child identity is missing"))?;
            for child in &node.children {
                if !parents.insert(*child) {
                    return Err(invalid_error(
                        "accessibility nodes must have one parent in the tree",
                    ));
                }
                pending.push(*child);
            }
        }
        if !parents.contains(&self.focus) && self.focus != self.root {
            return Err(invalid_error(
                "accessibility focus is outside the root tree",
            ));
        }
        if visited.len() != self.nodes.len() {
            return Err(invalid_error(
                "accessibility tree contains an unreachable node",
            ));
        }
        if parents.contains(&self.root) || root_node.hidden {
            return Err(invalid_error(
                "accessibility root cannot be hidden or parented",
            ));
        }
        Ok(())
    }

    pub(super) fn to_accesskit(&self) -> TreeUpdate {
        let nodes = self
            .nodes
            .iter()
            .map(|source| {
                let mut node = Node::new(accesskit_role(source.role));
                if !source.name.is_empty() {
                    node.set_label(source.name.clone());
                }
                if let Some(description) = &source.description {
                    node.set_description(description.clone());
                }
                if let Some(value) = &source.value {
                    node.set_value(value.clone());
                }
                if !source.children.is_empty() {
                    node.set_children(
                        source
                            .children
                            .iter()
                            .copied()
                            .map(NodeId::from)
                            .collect::<Vec<_>>(),
                    );
                }
                if source.hidden {
                    node.set_hidden();
                }
                if source.disabled {
                    node.set_disabled();
                }
                if source.focusable {
                    node.add_action(Action::Focus);
                }
                if let Some(expanded) = source.expanded {
                    node.set_expanded(expanded);
                }
                if let Some(selected) = source.selected {
                    node.set_selected(selected);
                }
                if let Some(checked) = source.checked {
                    node.set_toggled(checked.into());
                }
                for action in &source.actions {
                    node.add_action(accesskit_action(*action));
                }
                (NodeId::from(source.id), node)
            })
            .collect();
        TreeUpdate {
            nodes,
            tree: Some(TreeInfo::new(NodeId::from(self.root))),
            tree_id: TreeId::ROOT,
            focus: NodeId::from(self.focus),
        }
    }
}

fn accesskit_role(role: AccessibilityRole) -> Role {
    match role {
        AccessibilityRole::Application => Role::Application,
        AccessibilityRole::Main => Role::Main,
        AccessibilityRole::Navigation => Role::Navigation,
        AccessibilityRole::Complementary => Role::Complementary,
        AccessibilityRole::Group => Role::Group,
        AccessibilityRole::Toolbar => Role::Toolbar,
        AccessibilityRole::Menu => Role::Menu,
        AccessibilityRole::MenuItem => Role::MenuItem,
        AccessibilityRole::Button => Role::Button,
        AccessibilityRole::Text => Role::Label,
        AccessibilityRole::TextInput => Role::TextInput,
        AccessibilityRole::CheckBox => Role::CheckBox,
        AccessibilityRole::Radio => Role::RadioButton,
        AccessibilityRole::Slider => Role::Slider,
        AccessibilityRole::ComboBox => Role::ComboBox,
        AccessibilityRole::Dialog => Role::Dialog,
        AccessibilityRole::Status => Role::Status,
        AccessibilityRole::Table => Role::Table,
    }
}

pub(super) fn accesskit_action(action: AccessibilityAction) -> Action {
    match action {
        AccessibilityAction::Activate | AccessibilityAction::Toggle => Action::Click,
        AccessibilityAction::Focus => Action::Focus,
        AccessibilityAction::SetValue => Action::ReplaceSelectedText,
        AccessibilityAction::AdjustValue => Action::Increment,
        AccessibilityAction::Open => Action::Expand,
    }
}

fn validate_id(id: u64) -> io::Result<()> {
    if id == 0 {
        return Err(invalid_error(
            "accessibility node identities must be nonzero",
        ));
    }
    Ok(())
}

fn validate_text(value: &str) -> io::Result<()> {
    if value.len() > MAX_ACCESSIBILITY_TEXT_BYTES {
        return Err(limit_error("accessibility text exceeds its byte bound"));
    }
    Ok(())
}

fn invalid_error(message: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message)
}

fn limit_error(message: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message)
}
