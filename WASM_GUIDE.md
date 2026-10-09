# Moirai WebAssembly Guide

## Overview

Moirai provides first-class WebAssembly support, enabling high-performance concurrent computing directly in web browsers. By leveraging Web Workers and SharedArrayBuffer, Moirai brings true parallelism to JavaScript applications.

## Benefits of Moirai in WASM

### 1. **True Parallelism in the Browser**
- Utilizes Web Workers for parallel execution
- SharedArrayBuffer for zero-copy communication between workers
- Atomics for efficient synchronization

### 2. **Unified API**
- Same Moirai API works in native and WASM environments
- Seamless integration with JavaScript async/await
- Type-safe bindings through wasm-bindgen

### 3. **Performance**
- Near-native performance for compute-intensive tasks
- Efficient memory usage with zero-copy operations
- Automatic work distribution across available cores

### 4. **Use Cases**
- **Data Processing**: Process large datasets directly in the browser
- **Scientific Computing**: Run simulations and numerical computations
- **Image/Video Processing**: Real-time media manipulation
- **Game Engines**: Physics simulations and AI
- **Cryptography**: Parallel cryptographic operations
- **Machine Learning**: Client-side ML inference

## Building for WASM

### Prerequisites

1. Install Rust and wasm-pack:
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh
```

2. Add the wasm32 target:
```bash
rustup target add wasm32-unknown-unknown
```

### Building

Use the provided build script:
```bash
./build-wasm.sh
```

Or build manually:
```bash
cd moirai-core
cargo build --target wasm32-unknown-unknown --no-default-features --features wasm
```

### Creating a WASM Package

```toml
# Cargo.toml
[package]
name = "my-moirai-app"
version = "0.1.0"
edition = "2021"

[lib]
crate-type = ["cdylib"]

[dependencies]
moirai-core = { version = "0.1", default-features = false, features = ["wasm"] }
wasm-bindgen = "0.2"
```

```rust
// src/lib.rs
use wasm_bindgen::prelude::*;
use moirai_core::prelude::*;

#[wasm_bindgen]
pub struct MoiraiApp {
    executor: WasmExecutor,
}

#[wasm_bindgen]
impl MoiraiApp {
    #[wasm_bindgen(constructor)]
    pub fn new(num_workers: usize) -> Result<MoiraiApp, JsValue> {
        let executor = WasmExecutor::new(num_workers)?;
        Ok(MoiraiApp { executor })
    }
    
    #[wasm_bindgen]
    pub async fn parallel_process(&self, data: Vec<f64>) -> Vec<f64> {
        // Your parallel processing logic here
        data.into_iter()
            .map(|x| x * 2.0)
            .collect()
    }
}
```

## JavaScript Integration

### Basic Setup

```html
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Moirai WASM App</title>
    <meta http-equiv="Cross-Origin-Embedder-Policy" content="require-corp">
    <meta http-equiv="Cross-Origin-Opener-Policy" content="same-origin">
</head>
<body>
    <script type="module">
        import init, { MoiraiApp } from './pkg/my_moirai_app.js';
        
        async function run() {
            // Initialize WASM module
            await init();
            
            // Create Moirai instance with 4 workers
            const app = new MoiraiApp(4);
            
            // Process data in parallel
            const data = Array.from({length: 1000000}, (_, i) => i);
            const result = await app.parallel_process(data);
            
            console.log('Processed:', result.length, 'items');
        }
        
        run();
    </script>
</body>
</html>
```

### Advanced Example: Parallel Map-Reduce

```javascript
import init, { MoiraiApp } from './pkg/moirai_wasm.js';

class DataProcessor {
    constructor(numWorkers = navigator.hardwareConcurrency || 4) {
        this.appPromise = this.initialize(numWorkers);
    }
    
    async initialize(numWorkers) {
        await init();
        return new MoiraiApp(numWorkers);
    }
    
    async mapReduce(data, mapFn, reduceFn, initial) {
        const app = await this.appPromise;
        
        // Serialize functions for Web Workers
        const mapStr = mapFn.toString();
        const reduceStr = reduceFn.toString();
        
        return app.map_reduce(data, mapStr, reduceStr, initial);
    }
    
    async parallelSort(data) {
        const app = await this.appPromise;
        return app.parallel_sort(data);
    }
    
    async parallelFilter(data, predicate) {
        const app = await this.appPromise;
        const predicateStr = predicate.toString();
        return app.parallel_filter(data, predicateStr);
    }
}

// Usage
const processor = new DataProcessor();

// Parallel sum of squares
const numbers = Array.from({length: 1000000}, (_, i) => i);
const sumOfSquares = await processor.mapReduce(
    numbers,
    x => x * x,
    (a, b) => a + b,
    0
);

// Parallel filtering
const filtered = await processor.parallelFilter(
    numbers,
    x => x % 2 === 0
);
```

## Performance Considerations

### 1. **SharedArrayBuffer Requirements**
- Requires secure context (HTTPS)
- Needs specific CORS headers:
  ```
  Cross-Origin-Embedder-Policy: require-corp
  Cross-Origin-Opener-Policy: same-origin
  ```

### 2. **Memory Management**
- WASM has a linear memory model
- Use `--max-memory` flag to set memory limits
- Monitor memory usage with `performance.memory`

### 3. **Optimization Tips**
- Batch operations to reduce overhead
- Use typed arrays for numeric data
- Minimize data serialization between JS and WASM
- Profile with Chrome DevTools

### 4. **Bundle Size Optimization**
```toml
# Cargo.toml
[profile.release]
opt-level = "z"     # Optimize for size
lto = true          # Link-time optimization
codegen-units = 1   # Single codegen unit
strip = true        # Strip symbols
```

## Debugging

### Enable Debug Symbols
```toml
[profile.wasm-dev]
inherits = "release"
debug = true
```

### Console Logging
```rust
use web_sys::console;

console::log_1(&"Debug message".into());
console::error_1(&format!("Error: {}", error).into());
```

### Source Maps
```bash
wasm-pack build --dev
```

## Browser Compatibility

| Feature | Chrome | Firefox | Safari | Edge |
|---------|--------|---------|--------|------|
| WebAssembly | ✅ 57+ | ✅ 52+ | ✅ 11+ | ✅ 16+ |
| Web Workers | ✅ | ✅ | ✅ | ✅ |
| SharedArrayBuffer | ✅ 68+ | ✅ 79+ | ✅ 15.2+ | ✅ 79+ |
| Atomics | ✅ 68+ | ✅ 78+ | ✅ 15.2+ | ✅ 79+ |

## Security Considerations

1. **Site Isolation**: Required for SharedArrayBuffer
2. **Content Security Policy**: May need adjustment for Workers
3. **CORS**: Proper headers required for cross-origin resources
4. **Spectre Mitigations**: SharedArrayBuffer has additional security requirements

## Troubleshooting

### Common Issues

1. **SharedArrayBuffer not defined**
   - Ensure CORS headers are set correctly
   - Check browser compatibility
   - Verify HTTPS is being used

2. **Worker creation fails**
   - Check Content Security Policy
   - Ensure worker scripts are accessible
   - Verify correct MIME types

3. **Memory errors**
   - Increase WASM memory limits
   - Check for memory leaks
   - Use memory profiling tools

### Performance Profiling

```javascript
// Measure parallel operation performance
console.time('parallel_operation');
const result = await app.process(largeDataset);
console.timeEnd('parallel_operation');

// Memory usage
if (performance.memory) {
    console.log('Memory:', {
        used: performance.memory.usedJSHeapSize / 1048576,
        total: performance.memory.totalJSHeapSize / 1048576
    });
}
```

## Future Enhancements

1. **WASM Threads**: Direct thread support without Web Workers
2. **SIMD in WASM**: Vectorized operations
3. **Exception Handling**: Better error propagation
4. **Interface Types**: Improved JS/WASM interop
5. **Module Linking**: Better code organization

## Example Applications

### 1. Image Processing
```javascript
const processor = new MoiraiApp(4);

async function applyFilter(imageData, kernel) {
    const pixels = new Float32Array(imageData.data);
    const filtered = await processor.convolve(pixels, kernel);
    return new ImageData(new Uint8ClampedArray(filtered), imageData.width);
}
```

### 2. Data Analytics
```javascript
async function analyzeDataset(data) {
    const app = new MoiraiApp(navigator.hardwareConcurrency);
    
    const [mean, stddev, median] = await Promise.all([
        app.calculate_mean(data),
        app.calculate_stddev(data),
        app.calculate_median(data)
    ]);
    
    return { mean, stddev, median };
}
```

### 3. Real-time Simulation
```javascript
class ParticleSimulation {
    constructor(numParticles) {
        this.app = new MoiraiApp(4);
        this.particles = new Float32Array(numParticles * 3); // x, y, z
    }
    
    async update(deltaTime) {
        this.particles = await this.app.update_particles(
            this.particles,
            deltaTime
        );
    }
}
```

## Conclusion

Moirai's WASM support brings powerful parallel computing capabilities to web applications. By leveraging Web Workers and SharedArrayBuffer, developers can build high-performance applications that run entirely in the browser, opening new possibilities for client-side computing.