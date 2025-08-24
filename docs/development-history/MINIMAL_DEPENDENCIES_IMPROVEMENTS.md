# Minimal Dependencies Improvements

## Overview

This document summarizes the improvements made to minimize dependencies in the Moirai concurrency library, particularly focusing on the IPC module and overall dependency management.

## Key Changes

### 1. IPC Module Dependency Reduction

#### Before:
- Direct usage of `std::io::Error` throughout
- Dependency on `std::net::SocketAddr` for networking
- Use of `std::os::raw::c_void` for Windows compatibility
- Heavy reliance on std library types

#### After:
- Custom `IpcError` enum to avoid `std::io` dependency
- Removed `std::net` dependency by using string addresses
- Replaced `*mut c_void` with `usize` for Windows handles
- Minimal platform-specific imports only where necessary

### 2. Platform Abstraction Benefits

The platform abstraction layer (`platform.rs`) centralizes all platform-specific imports:
- Reduces duplicate imports across modules
- Makes WASM/no-std support easier
- Improves code maintainability
- Follows DRY principle

### 3. Dependency Analysis

#### Required Dependencies:
- **libc** (Unix only): Essential for shared memory operations
  - Used for: `shm_open`, `mmap`, `munmap`, `ftruncate`
  - Platform-specific: Only included on Unix targets
  - No alternative: Direct system calls required for IPC

#### Removed Dependencies:
- **std::io**: Replaced with custom `IpcError` type
- **std::net**: Replaced with string-based addressing
- **std::os::raw** (partial): Minimized usage

### 4. Error Handling Improvements

Created minimal `IpcError` enum:
```rust
pub enum IpcError {
    SystemError(i32),    // OS error codes
    InvalidArgument,     // Bad input
    NotImplemented,      // Placeholder features
    NotFound,           // Resource missing
    PermissionDenied,   // Access denied
}
```

Benefits:
- No dependency on std::io::Error
- Lightweight and focused
- Easy to extend
- Clear error categories

### 5. Code Quality Improvements

Applied design principles:
- **KISS**: Simple error types instead of complex std::io chains
- **YAGNI**: Removed unused RDMA structs
- **DRY**: Centralized platform imports
- **SRP**: Each module has focused dependencies

### 6. Future Considerations

For further dependency reduction:
1. Consider implementing IPC without libc (using inline assembly)
2. Create no-std alternative implementations
3. Make IPC module optional feature
4. Implement pure-Rust shared memory (complex but possible)

## Summary

Successfully minimized dependencies while maintaining functionality:
- IPC module now has minimal external dependencies
- Only essential platform-specific code remains
- Custom error types avoid heavy std dependencies
- Code is cleaner and more maintainable
- WASM/no-std support is easier to implement

The `libc` dependency remains as it's essential for Unix IPC functionality, but all other dependencies have been minimized or eliminated where possible.