use super::{
    InstanceName, InstanceRole, MAX_INSTANCE_MESSAGE_BYTES, MAX_INSTANCE_NAME_BYTES, claim,
    read_frame, write_frame,
};
use std::io::{Cursor, ErrorKind};

fn unique(label: &str) -> InstanceName {
    InstanceName::new(&format!("moirai-test-{label}-{}", std::process::id())).expect("name")
}

#[test]
fn names_are_validated() {
    assert!(InstanceName::new("org.example.viewer").is_ok());
    assert!(InstanceName::new(&"a".repeat(MAX_INSTANCE_NAME_BYTES)).is_ok());
    for rejected in [
        "",
        ".hidden",
        "-flag",
        "Upper",
        "with space",
        "slash/name",
        "dot..ok-but/no",
        &"a".repeat(MAX_INSTANCE_NAME_BYTES + 1),
    ] {
        assert!(InstanceName::new(rejected).is_err(), "{rejected:?}");
    }
}

#[test]
fn frames_round_trip_and_are_bounded() {
    let mut buffer = Vec::new();
    write_frame(&mut buffer, b"org.example://open").expect("write");
    assert_eq!(
        read_frame(&mut Cursor::new(buffer)).expect("read"),
        b"org.example://open"
    );
    let mut oversized = u32::try_from(MAX_INSTANCE_MESSAGE_BYTES + 1)
        .expect("bound")
        .to_le_bytes()
        .to_vec();
    oversized.extend([0; 8]);
    let error = read_frame(&mut Cursor::new(oversized)).expect_err("oversized");
    assert_eq!(error.kind(), ErrorKind::InvalidData);
    let error = read_frame(&mut Cursor::new(vec![3, 0, 0, 0, 1])).expect_err("truncated");
    assert_eq!(error.kind(), ErrorKind::UnexpectedEof);
}

#[test]
fn the_second_claim_forwards_to_the_first() {
    let name = unique("forward");
    let InstanceRole::Primary(mut primary) = claim(&name).expect("first claim") else {
        panic!("the first claim must be primary");
    };
    assert_eq!(primary.try_receive().expect("idle"), None);

    let InstanceRole::Secondary(secondary) = claim(&name).expect("second claim") else {
        panic!("the second claim must be secondary");
    };
    let oversized = vec![0; MAX_INSTANCE_MESSAGE_BYTES + 1];
    let InstanceRole::Secondary(rejected) = claim(&name).expect("third claim") else {
        panic!("the third claim must be secondary");
    };
    assert_eq!(
        rejected.send(&oversized).expect_err("oversized").kind(),
        ErrorKind::InvalidInput
    );
    secondary.send(b"org.example://study/7").expect("send");
    let received = (0..500)
        .find_map(|_| primary.try_receive().transpose())
        .expect("a forwarded message")
        .expect("received");
    assert_eq!(received, b"org.example://study/7");

    drop(primary);
    let InstanceRole::Primary(_) = claim(&name).expect("reclaim") else {
        panic!("a released name is claimed again");
    };
}
