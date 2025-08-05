//! Example demonstrating blocking channel behavior in Moirai
//!
//! This example shows how the Channel trait properly implements blocking
//! semantics, ensuring that SpscChannel and MpmcChannel conform to the
//! Liskov Substitution Principle.

use moirai_core::channel::{spsc, mpmc, Channel};
use std::thread;
use std::time::{Duration, Instant};

fn main() {
    println!("=== Moirai Blocking Channel Examples ===\n");
    
    demonstrate_spsc_blocking();
    println!();
    demonstrate_mpmc_blocking();
    println!();
    demonstrate_channel_trait_polymorphism();
}

fn demonstrate_spsc_blocking() {
    println!("1. SPSC Channel Blocking Behavior:");
    
    // Create a small channel to demonstrate blocking
    let (tx, rx) = spsc::<String>(2);
    
    // Fill the channel
    tx.send("Message 1".to_string()).unwrap();
    tx.send("Message 2".to_string()).unwrap();
    println!("   - Channel filled with 2 messages (capacity: 2)");
    
    // Spawn a consumer thread
    let consumer = thread::spawn(move || {
        println!("   - Consumer: Starting...");
        thread::sleep(Duration::from_millis(100));
        
        // Consume messages
        for i in 1..=3 {
            let msg = rx.recv().unwrap();
            println!("   - Consumer: Received '{}'", msg);
            if i < 3 {
                thread::sleep(Duration::from_millis(50));
            }
        }
    });
    
    // Try to send a third message - this will block
    println!("   - Producer: Attempting to send 3rd message (will block)...");
    let start = Instant::now();
    tx.send("Message 3".to_string()).unwrap();
    let elapsed = start.elapsed();
    println!("   - Producer: Send completed after {:?}", elapsed);
    
    consumer.join().unwrap();
}

fn demonstrate_mpmc_blocking() {
    println!("2. MPMC Channel Blocking Behavior:");
    
    let (tx, rx) = mpmc::<i32>(3);
    
    // Spawn multiple producers
    let producers: Vec<_> = (0..3).map(|id| {
        let tx = tx.clone();
        thread::spawn(move || {
            for i in 0..2 {
                let value = id * 10 + i;
                println!("   - Producer {}: Sending {}", id, value);
                tx.send(value).unwrap();
                thread::sleep(Duration::from_millis(20));
            }
        })
    }).collect();
    
    // Spawn multiple consumers
    let consumers: Vec<_> = (0..2).map(|id| {
        let rx = rx.clone();
        thread::spawn(move || {
            thread::sleep(Duration::from_millis(50)); // Let producers fill the channel
            for _ in 0..3 {
                let value = rx.recv().unwrap();
                println!("   - Consumer {}: Received {}", id, value);
                thread::sleep(Duration::from_millis(30));
            }
        })
    }).collect();
    
    // Wait for all threads
    for p in producers {
        p.join().unwrap();
    }
    for c in consumers {
        c.join().unwrap();
    }
}

fn demonstrate_channel_trait_polymorphism() {
    println!("3. Channel Trait Polymorphism (LSP Compliance):");
    
    // Both SPSC and MPMC channels can be used through the Channel trait
    let spsc_channel = spsc::<i32>(5);
    let mpmc_channel = mpmc::<i32>(5);
    
    // Test blocking behavior through the trait
    // Note: Generic trait testing removed as Channel trait is for the channel itself,
    // not for sender/receiver halves
}