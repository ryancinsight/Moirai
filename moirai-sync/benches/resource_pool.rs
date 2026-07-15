use criterion::{black_box, criterion_group, criterion_main, Criterion};
use moirai_sync::{ShardedResourcePool, SizeBounded};

struct Resource {
    size: u64,
}

impl SizeBounded for Resource {
    fn size(&self) -> u64 {
        self.size
    }
}

fn recycle_take(c: &mut Criterion) {
    let pool = ShardedResourcePool::new(1024, 1 << 20);

    c.bench_function("resource_pool/recycle_take", |b| {
        b.iter(|| {
            pool.recycle(Resource { size: 64 });
            black_box(
                pool.take_at_least(64)
                    .expect("recycled resource must be available"),
            );
        });
    });
}

criterion_group!(benches, recycle_take);
criterion_main!(benches);
