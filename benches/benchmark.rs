use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use metrohash::MetroHashSet;

use rand::{
    distributions::{Distribution, Uniform},
    Rng, SeedableRng,
};

use fast_map::{FastMap, U32KeysBlock};

type MapSIMD = FastMap<U32KeysBlock, u32>;

fn bench_lookup(c: &mut Criterion) {
    let mut group = c.benchmark_group("FastMap lookup");

    for size in (5..=22).map(|shift| 1 << shift) {
        let interval = Uniform::new(0, 4 * size as u32);
        let mut rng = rand_pcg::Pcg32::seed_from_u64(3);

        let indices: Vec<_> = {
            let mut added = MetroHashSet::default();
            while added.len() < size {
                added.insert(interval.sample(&mut rng));
            }

            added.into_iter().collect()
        };

        let mut indices_sorted = indices.clone();
        indices_sorted.sort();

        {
            let mut map = MapSIMD::with_capacity(size);
            indices.iter().for_each(|&i| map.try_insert(i, i).unwrap());

            group.bench_with_input(BenchmarkId::new("FastMap SIMD", size), &size, |b, _| {
                b.iter(|| {
                    indices_sorted.iter().copied().for_each(|k| {
                        black_box(unsafe { *map.get_existing(k) });
                    })
                });
            });
        }
    }
}

criterion_group!(benches, bench_lookup);
criterion_main!(benches);
