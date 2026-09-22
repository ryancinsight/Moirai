//! Contract and action translation tests for the Windows accessibility seam.

use accesskit::{Action, ActionData, NodeId, TreeId};
use std::io;

use super::contract::accesskit_action;
use super::*;

fn node(id: u64, role: AccessibilityRole, name: &str) -> AccessibilityNode {
    AccessibilityNode::new(id, role, name).expect("test node is valid")
}

#[test]
fn validates_connected_tree_and_preserves_accesskit_semantics() {
    let mut root = node(1, AccessibilityRole::Application, "Metis");
    root.set_children(vec![2]).expect("child identity is valid");
    let mut button = node(2, AccessibilityRole::Button, "Calculate");
    button.set_focusable(true);
    button.add_action(AccessibilityAction::Activate);
    button.set_expanded(Some(false));
    button.set_selected(Some(false));
    button
        .set_description(Some(String::from("Run the calculation")))
        .expect("description is bounded");
    let tree = AccessibilityTree::from_nodes(1, 2, vec![root, button]).expect("tree is connected");
    let update = tree.to_accesskit();
    assert_eq!(update.focus, NodeId::from(2));
    assert_eq!(update.nodes.len(), 2);
    let (_, button) = update
        .nodes
        .iter()
        .find(|(id, _)| *id == NodeId::from(2))
        .expect("button is present");
    assert_eq!(button.label(), Some("Calculate"));
    assert_eq!(button.is_expanded(), Some(false));
    assert_eq!(button.is_selected(), Some(false));
    assert_eq!(
        accesskit_action(AccessibilityAction::Activate),
        Action::Click
    );
}

#[test]
fn maps_command_surface_roles_to_accesskit() {
    let roles = [
        (AccessibilityRole::Navigation, accesskit::Role::Navigation),
        (
            AccessibilityRole::Complementary,
            accesskit::Role::Complementary,
        ),
        (AccessibilityRole::Toolbar, accesskit::Role::Toolbar),
        (AccessibilityRole::Menu, accesskit::Role::Menu),
        (AccessibilityRole::MenuItem, accesskit::Role::MenuItem),
    ];
    for (source, expected) in roles {
        let node = node(1, source, "command surface");
        let tree = AccessibilityTree::from_nodes(1, 1, vec![node]).expect("tree is valid");
        let update = tree.to_accesskit();
        assert_eq!(update.nodes.first().expect("one node").1.role(), expected);
    }
}

#[test]
fn rejects_duplicate_unreachable_cyclic_and_hidden_action_nodes() {
    let duplicate = AccessibilityTree::from_nodes(
        1,
        1,
        vec![
            node(1, AccessibilityRole::Application, "root"),
            node(1, AccessibilityRole::Text, "duplicate"),
        ],
    );
    assert_eq!(
        duplicate.expect_err("duplicate identity must fail").kind(),
        io::ErrorKind::InvalidInput
    );

    let unreachable = AccessibilityTree::from_nodes(
        1,
        1,
        vec![
            node(1, AccessibilityRole::Application, "root"),
            node(2, AccessibilityRole::Text, "orphan"),
        ],
    );
    assert_eq!(
        unreachable.expect_err("unreachable node must fail").kind(),
        io::ErrorKind::InvalidInput
    );

    let mut cyclic_root = node(1, AccessibilityRole::Application, "root");
    cyclic_root
        .set_children(vec![2])
        .expect("child identity is valid");
    let mut cyclic_child = node(2, AccessibilityRole::Group, "child");
    cyclic_child
        .set_children(vec![1])
        .expect("child identity is valid");
    let cyclic = AccessibilityTree::from_nodes(1, 1, vec![cyclic_root, cyclic_child]);
    assert_eq!(
        cyclic.expect_err("cycle must fail").kind(),
        io::ErrorKind::InvalidInput
    );

    let mut hidden = node(2, AccessibilityRole::Button, "hidden");
    hidden.set_hidden(true);
    hidden.add_action(AccessibilityAction::Activate);
    let mut root = node(1, AccessibilityRole::Application, "root");
    root.set_children(vec![2]).expect("child identity is valid");
    let hidden_action = AccessibilityTree::from_nodes(1, 1, vec![root, hidden]);
    assert_eq!(
        hidden_action.expect_err("hidden actions must fail").kind(),
        io::ErrorKind::InvalidInput
    );
}

#[test]
fn rejects_unbounded_text_and_invalid_focus() {
    let oversized = "x".repeat(MAX_ACCESSIBILITY_TEXT_BYTES + 1);
    let result = AccessibilityNode::new(1, AccessibilityRole::Text, oversized);
    assert_eq!(
        result.expect_err("text bound must fail").kind(),
        io::ErrorKind::InvalidInput
    );

    let mut root = node(1, AccessibilityRole::Application, "root");
    root.set_children(vec![2]).expect("child identity is valid");
    let child = node(2, AccessibilityRole::Button, "button");
    let result = AccessibilityTree::from_nodes(1, 2, vec![root, child]);
    assert_eq!(
        result.expect_err("non-focusable focus must fail").kind(),
        io::ErrorKind::InvalidInput
    );
}

#[test]
fn maps_platform_actions_to_bounded_requests() {
    let request = action_request(accesskit::ActionRequest {
        action: Action::ReplaceSelectedText,
        target_tree: TreeId::ROOT,
        target_node: NodeId::from(9),
        data: Some(ActionData::Value(Box::<str>::from("value"))),
    })
    .expect("replace text is supported");
    assert_eq!(
        request,
        AccessibilityActionRequest {
            target_node: 9,
            action: AccessibilityAction::SetValue,
            value: Some(String::from("value")),
            delta: None,
        }
    );
    let set_value = action_request(accesskit::ActionRequest {
        action: Action::SetValue,
        target_tree: TreeId::ROOT,
        target_node: NodeId::from(9),
        data: Some(ActionData::Value(Box::<str>::from("replacement"))),
    })
    .expect("set value is supported");
    assert_eq!(set_value.action, AccessibilityAction::SetValue);
    assert_eq!(set_value.value.as_deref(), Some("replacement"));
    let increment = action_request(accesskit::ActionRequest {
        action: Action::Increment,
        target_tree: TreeId::ROOT,
        target_node: NodeId::from(9),
        data: None,
    })
    .expect("increment is supported");
    assert_eq!(increment.delta, Some(1));
    assert!(
        action_request(accesskit::ActionRequest {
            action: Action::ScrollDown,
            target_tree: TreeId::ROOT,
            target_node: NodeId::from(9),
            data: None,
        })
        .is_none()
    );
}
