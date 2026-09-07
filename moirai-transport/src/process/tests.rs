//! Native process lifecycle regressions.
use super::*;
use std::{
    io::{BufRead, BufReader, Read, Write},
    time::Duration,
};

fn fixture(mode: &str) -> ProcessSpec {
    ProcessSpec::new(std::env::current_exe().expect("test binary"))
        .args(["--exact", "process::tests::child_entry", "--nocapture"])
        .env_clear()
        .env("MOIRAI_PROCESS_FIXTURE", mode)
        .piped_stdio()
}

#[test]
fn child_entry() {
    // This test executable is also the real child-process fixture. Normal
    // harness execution has no fixture command; subprocess invocations do.
    let Ok(mode) = std::env::var("MOIRAI_PROCESS_FIXTURE") else {
        return;
    };
    match mode.as_str() {
        "echo" => {
            let mut input = String::new();
            std::io::stdin()
                .read_line(&mut input)
                .expect("fixture input");
            std::io::stdout()
                .write_all(format!("ECHO:{input}").as_bytes())
                .expect("fixture output");
        }
        "wait" => {
            std::io::stdout()
                .write_all(b"READY\n")
                .expect("ready output");
            let mut input = [0];
            std::io::stdin()
                .read_exact(&mut input)
                .expect("parent retains pipe until termination");
        }
        "descendant" => {
            #[expect(
                clippy::zombie_processes,
                reason = "Windows test intentionally exits the root; its owning job terminates the descendant"
            )]
            let child = std::process::Command::new(std::env::current_exe().expect("fixture path"))
                .args(["--exact", "process::tests::child_entry", "--nocapture"])
                .env("MOIRAI_PROCESS_FIXTURE", "wait")
                .stdin(std::process::Stdio::inherit())
                .stdout(std::process::Stdio::inherit())
                .spawn()
                .expect("descendant");
            std::io::stdout()
                .write_all(format!("DESCENDANT:{}\n", child.id()).as_bytes())
                .expect("descendant identity");
        }
        "exit" => std::process::exit(7),
        _ => panic!("Unknown child fixture mode"),
    }
}
fn line_until(reader: &mut BufReader<std::fs::File>, prefix: &str) -> String {
    loop {
        let mut line = String::new();
        assert_ne!(
            reader.read_line(&mut line).expect("child output"),
            0,
            "EOF before marker"
        );
        if line.starts_with(prefix) {
            return line;
        }
    }
}
#[test]
fn process_supervisor_waits_for_successful_child() {
    let mut process = ProcessSupervisor::new()
        .spawn(fixture("echo"), ProcessDropPolicy::TerminateOnDrop)
        .expect("spawn");
    let id = process.id();
    let mut writer = process.take_stdin().expect("pipe writer");
    let mut reader = BufReader::new(process.take_stdout().expect("pipe reader"));
    writer.write_all(b"unicode-\xce\xb1\n").expect("pipe write");
    assert_eq!(line_until(&mut reader, "ECHO:"), "ECHO:unicode-α\n");
    drop(writer);
    let status = process
        .wait_timeout(Duration::from_secs(5))
        .expect("wait")
        .expect("child exit");
    assert_eq!(status.id, id);
    assert_eq!(status.outcome, ProcessOutcome::Succeeded);
    assert_eq!(status.code, Some(0));
}
#[test]
fn process_supervisor_times_out_and_terminates_child() {
    let mut process = ProcessSupervisor::new()
        .spawn(fixture("wait"), ProcessDropPolicy::TerminateOnDrop)
        .expect("spawn");
    let mut reader = BufReader::new(process.take_stdout().expect("pipe reader"));
    assert_eq!(line_until(&mut reader, "READY"), "READY\n");
    assert_eq!(
        process
            .wait_timeout(Duration::ZERO)
            .expect("nonblocking observation"),
        None
    );
    let status = process
        .terminate_timeout(Duration::from_secs(5))
        .expect("termination");
    assert_eq!(status.id, process.id());
    assert_eq!(status.outcome, ProcessOutcome::Failed);
}
#[test]
fn unsuccessful_exit_is_preserved() {
    let mut process = ProcessSupervisor::new()
        .spawn(fixture("exit"), ProcessDropPolicy::TerminateOnDrop)
        .expect("spawn");
    let status = process
        .wait_timeout(Duration::from_secs(5))
        .expect("wait")
        .expect("exit");
    assert_eq!(status.code, Some(7));
    assert_eq!(status.outcome, ProcessOutcome::Failed);
    assert_eq!(status.exit_status().code(), Some(7));
}
#[cfg(windows)]
#[test]
fn process_creation_rejects_embedded_zero_and_overlong_arguments() {
    for argument in ["bad\0arg".to_owned(), "x".repeat(32_767)] {
        let error = ProcessSupervisor::new()
            .spawn(
                fixture("exit").arg(argument),
                ProcessDropPolicy::TerminateOnDrop,
            )
            .expect_err("invalid command line");
        assert_eq!(error, ProcessError::InvalidSpecification);
    }
}

#[cfg(windows)]
#[test]
fn job_termination_closes_descendant_pipes_after_root_exit() {
    let mut process = ProcessSupervisor::new()
        .spawn(
            fixture("descendant").tree_containment(),
            ProcessDropPolicy::TerminateOnDrop,
        )
        .expect("contained spawn");
    let mut reader = BufReader::new(process.take_stdout().expect("pipe reader"));
    let mut child_id = None;
    let mut ready = false;
    while child_id.is_none() || !ready {
        let mut line = String::new();
        assert_ne!(reader.read_line(&mut line).expect("descendant output"), 0);
        if let Some(identity) = line.trim().strip_prefix("DESCENDANT:") {
            child_id = Some(identity.parse::<u32>().expect("process identity"));
        }
        ready |= line == "READY\n";
    }
    assert_ne!(child_id.expect("descendant identity"), process.id().get());
    assert_eq!(
        process
            .wait_timeout(Duration::from_secs(5))
            .expect("root wait")
            .expect("root exit")
            .code,
        Some(0)
    );
    assert_eq!(
        process
            .terminate_timeout(Duration::from_secs(5))
            .expect("tree termination")
            .code,
        Some(0)
    );
    let mut remaining = String::new();
    reader
        .read_to_string(&mut remaining)
        .expect("all descendant pipe writers closed");
    assert_eq!(reader.read(&mut [0]).expect("confirmed pipe EOF"), 0);
}

#[cfg(windows)]
#[test]
fn dropping_job_closes_a_waiting_child_pipe() {
    let mut process = ProcessSupervisor::new()
        .spawn(
            fixture("wait").tree_containment(),
            ProcessDropPolicy::TerminateOnDrop,
        )
        .expect("contained spawn");
    let mut reader = BufReader::new(process.take_stdout().expect("pipe reader"));
    let writer = process.take_stdin().expect("keep input open independently");
    assert_eq!(line_until(&mut reader, "READY"), "READY\n");
    drop(process);
    let mut tail = Vec::new();
    assert_eq!(reader.read_to_end(&mut tail).expect("job closes writer"), 0);
    drop(writer);
}
