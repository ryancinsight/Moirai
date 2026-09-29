fn manifest_section<'a>(content: &'a str, header: &str) -> &'a str {
    let Some(start) = content.find(header) else {
        return "";
    };
    let body = &content[start + header.len()..];
    let end = body.find("\n[").unwrap_or(body.len());
    &body[..end]
}

fn manifest_section_declares_dependency(section: &str, dependency: &str) -> bool {
    let spaced = format!("{dependency} ");
    let assigned = format!("{dependency}=");
    section.lines().any(|line| {
        let trimmed = line.trim_start();
        !trimmed.starts_with('#')
            && (trimmed.starts_with(&spaced) || trimmed.starts_with(&assigned))
    })
}

/// Reports every `[[package]]` block in a `Cargo.lock` whose `name` is
/// `moirai` or starts with `moirai-` and which carries a `source` line: the
/// workspace path copy of such a package has no `source` field, so any
/// registry or git source on a Moirai package means a dependency resolved it
/// from outside the workspace.
fn moirai_lock_package_source_violations(lock: &str) -> Vec<String> {
    fn flush(name: Option<&str>, source: Option<&str>, violations: &mut Vec<String>) {
        let Some(name) = name else { return };
        let Some(source) = source else { return };
        if name == "moirai" || name.starts_with("moirai-") {
            violations.push(format!("{name} ({source})"));
        }
    }

    let mut violations = Vec::new();
    let mut name: Option<&str> = None;
    let mut source: Option<&str> = None;

    for line in lock.lines() {
        if line == "[[package]]" {
            flush(name.take(), source.take(), &mut violations);
            continue;
        }
        if let Some(value) = line
            .strip_prefix("name = \"")
            .and_then(|rest| rest.strip_suffix('"'))
        {
            name = Some(value);
            continue;
        }
        if let Some(value) = line
            .strip_prefix("source = \"")
            .and_then(|rest| rest.strip_suffix('"'))
        {
            source = Some(value);
        }
    }
    flush(name.take(), source.take(), &mut violations);

    violations
}

fn expected_ready_sum(count: usize) -> usize {
    count * (count + 1) / 2
}

fn cpu_work(seed: usize) -> u64 {
    let mut value = seed as u64;
    for index in 0..CPU_WORK {
        value = value.wrapping_add((index as u64).wrapping_mul(31));
    }
    value
}

fn expected_cpu_work_sum(work_items: usize) -> u64 {
    let work_items = work_items as u64;
    let per_item_offset = 31u64 * (CPU_WORK as u64) * ((CPU_WORK - 1) as u64) / 2;
    work_items * per_item_offset + work_items * (work_items - 1) / 2
}
