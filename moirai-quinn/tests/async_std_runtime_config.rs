#![cfg(feature = "runtime-async-std")]

use std::{
    future::poll_fn,
    process::{Command, Stdio},
    sync::Arc,
    task::{Context, Waker},
    time::{Duration, Instant},
};

use moirai_quinn::{AsyncStdRuntime, Runtime};

const CHILD_MODE: &str = "MOIRAI_ASYNC_STD_WITNESS_MODE";

fn run_child(mode: &str) {
    let mut child = Command::new(std::env::current_exe().unwrap());
    child
        .args(["--exact", "runtime_witness_child", "--nocapture"])
        .env(CHILD_MODE, mode)
        .env("ASYNC_STD_THREAD_COUNT", "1")
        .env("ASYNC_GLOBAL_EXECUTOR_THREADS", "1")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    if mode == "configured" {
        child.env("ASYNC_STD_THREAD_NAME", "moirai-async-std-worker");
    } else {
        child.env_remove("ASYNC_STD_THREAD_NAME");
    }
    let output = child.output().unwrap();
    assert!(
        output.status.success(),
        "{mode} child failed:\n{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn honors_async_std_executor_configuration() {
    run_child("configured");
}

#[test]
fn retains_async_std_default_worker_name() {
    run_child("default");
}

#[test]
fn preserves_an_already_initialized_executor() {
    run_child("preinitialized");
}

#[test]
fn runtime_witness_child() {
    let mode = std::env::var(CHILD_MODE).unwrap_or_default();
    if mode == "preinitialized" {
        async_global_executor::init_with_config(
            async_global_executor::GlobalExecutorConfig::default()
                .with_min_threads(1)
                .with_max_threads(1)
                .with_thread_name_fn(|| "preexisting-executor".to_string()),
        );
    }

    let runtime = Arc::new(AsyncStdRuntime);
    let mut timer = runtime.new_timer(Instant::now() + Duration::from_secs(60));
    assert!(timer
        .as_mut()
        .poll(&mut Context::from_waker(Waker::noop()))
        .is_pending());
    timer.as_mut().reset(Instant::now());

    let (gate, mut wait) = moirai_async::sync::oneshot::channel();
    let (tx, rx) = std::sync::mpsc::channel();
    runtime.spawn(Box::pin(async move {
        wait.recv().await.unwrap();
        poll_fn(|cx| timer.as_mut().poll(cx)).await;
        tx.send(
            std::thread::current()
                .name()
                .unwrap_or("unnamed")
                .to_string(),
        )
        .unwrap();
    }));
    drop(runtime);
    gate.send(()).unwrap();
    let worker = rx
        .recv_timeout(Duration::from_secs(5))
        .expect("detached task and reset timer must complete");
    let expected = match mode.as_str() {
        "configured" => Some("moirai-async-std-worker"),
        "default" => Some("async-std/runtime"),
        "preinitialized" => Some("preexisting-executor"),
        _ => None,
    };
    if let Some(expected) = expected {
        assert_eq!(worker, expected);
    }
}
