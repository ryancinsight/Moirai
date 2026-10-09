#!/bin/bash

# Build script for Moirai WASM

set -e

echo "Building Moirai for WebAssembly..."

# Install wasm-pack if not already installed
if ! command -v wasm-pack &> /dev/null; then
    echo "Installing wasm-pack..."
    curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh
fi

# Install wasm32 target if not already installed
rustup target add wasm32-unknown-unknown

# Build moirai-core for WASM
echo "Building moirai-core..."
cd moirai-core
cargo build --target wasm32-unknown-unknown --no-default-features --features wasm
cd ..

# Build moirai-iter for WASM
echo "Building moirai-iter..."
cd moirai-iter
cargo build --target wasm32-unknown-unknown --no-default-features --features no-std
cd ..

# Create WASM package with wasm-pack
echo "Creating WASM package..."
mkdir -p wasm-pkg
cd wasm-pkg

# Create a wrapper crate for wasm-bindgen
cat > Cargo.toml << EOF
[package]
name = "moirai-wasm"
version = "0.1.0"
edition = "2021"

[lib]
crate-type = ["cdylib"]

[dependencies]
moirai-core = { path = "../moirai-core", default-features = false, features = ["wasm"] }
moirai-iter = { path = "../moirai-iter", default-features = false, features = ["no-std"] }
wasm-bindgen = "0.2"
wasm-bindgen-futures = "0.4"
js-sys = "0.3"
web-sys = { version = "0.3", features = ["console", "Worker", "SharedArrayBuffer"] }

[profile.release]
opt-level = "z"
lto = true
EOF

# Create wrapper lib.rs
cat > src/lib.rs << 'EOF'
use wasm_bindgen::prelude::*;
use moirai_core::prelude::*;

#[wasm_bindgen]
pub struct MoiraiWasm {
    executor: Option<WasmExecutor>,
}

#[wasm_bindgen]
impl MoiraiWasm {
    #[wasm_bindgen(constructor)]
    pub fn new(num_workers: usize) -> Result<MoiraiWasm, JsValue> {
        console_error_panic_hook::set_once();
        
        let executor = WasmExecutor::new(num_workers)?;
        
        Ok(MoiraiWasm {
            executor: Some(executor),
        })
    }
    
    #[wasm_bindgen]
    pub fn parallel_map(&self, data: Vec<f64>, multiplier: f64) -> Vec<f64> {
        // Example parallel map operation
        data.into_iter()
            .map(|x| x * multiplier)
            .collect()
    }
    
    #[wasm_bindgen]
    pub fn parallel_reduce(&self, data: Vec<f64>) -> f64 {
        // Example parallel reduce operation
        data.into_iter().sum()
    }
}

#[wasm_bindgen(start)]
pub fn main() {
    console_error_panic_hook::set_once();
    web_sys::console::log_1(&"Moirai WASM initialized".into());
}
EOF

mkdir -p src

# Build with wasm-pack
wasm-pack build --target web --out-dir pkg

echo "WASM build complete! Output in wasm-pkg/pkg/"

# Create example HTML
cat > pkg/index.html << 'EOF'
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Moirai WASM Example</title>
</head>
<body>
    <h1>Moirai WASM Example</h1>
    <button id="runTest">Run Parallel Test</button>
    <div id="results"></div>
    
    <script type="module">
        import init, { MoiraiWasm } from './moirai_wasm.js';
        
        async function run() {
            await init();
            
            // Create Moirai instance with 4 workers
            const moirai = new MoiraiWasm(4);
            
            document.getElementById('runTest').addEventListener('click', () => {
                const data = Array.from({length: 1000000}, (_, i) => i);
                
                console.time('parallel_map');
                const mapped = moirai.parallel_map(data, 2.0);
                console.timeEnd('parallel_map');
                
                console.time('parallel_reduce');
                const sum = moirai.parallel_reduce(mapped);
                console.timeEnd('parallel_reduce');
                
                document.getElementById('results').innerHTML = `
                    <p>Mapped ${data.length} items</p>
                    <p>Sum: ${sum}</p>
                `;
            });
        }
        
        run();
    </script>
</body>
</html>
EOF

echo "Example HTML created at wasm-pkg/pkg/index.html"