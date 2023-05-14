use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use metrohash::{MetroBuildHasher, MetroHashMap};
use rand::{Rng, SeedableRng};

use fast_map::{FastMap, U32KeysBlock, U32Single};

type MapSIMD = FastMap<U32KeysBlock, u32>;

// fn bench_creation(c: &mut Criterion) {
//     let mut group = c.benchmark_group("MetroHashMap vs FastMap creation");
//
//     for size in (5..=22).map(|shift| 1 << shift) {
//         let mut rng = rand_pcg::Pcg32::seed_from_u64(3);
//         group.bench_with_input(BenchmarkId::new("MetroHashMap", size), &size, |b, _| {
//             b.iter(|| {
//                 let map: MetroHashMap<u32, u32> = (0..size)
//                     .map(|_| {
//                         let v = rng.gen_range(0..u32::MAX);
//                         (v, v)
//                     })
//                     .collect();
//                 black_box(map);
//             })
//         });
//
//         let mut rng = rand_pcg::Pcg32::seed_from_u64(3);
//         group.bench_with_input(BenchmarkId::new("FastMap SIMD", size), &size, |b, _| {
//             b.iter(|| {
//                 let mut map = MapSIMD::with_capacity(size);
//                 (0..size)
//                     .map(|_| rng.gen_range(0..u32::MAX))
//                     .for_each(|key| {
//                         map.try_insert(key, key).unwrap();
//                     });
//                 black_box(map);
//             })
//         });
//
//         let mut rng = rand_pcg::Pcg32::seed_from_u64(3);
//         group.bench_with_input(BenchmarkId::new("FastMap Simple", size), &size, |b, _| {
//             b.iter(|| {
//                 let mut map = MapSimple::with_capacity(size);
//                 (0..size)
//                     .map(|_| rng.gen_range(0..u32::MAX))
//                     .for_each(|key| {
//                         map.try_insert(key, key).unwrap();
//                     });
//                 black_box(map);
//             })
//         });
//     }
// }

fn bench_lookup(c: &mut Criterion) {
    let mut group = c.benchmark_group("MetroHashMap vs FastMap lookup");

    for size in (5..=22).map(|shift| 1 << shift) {
        {
            let mut rng = rand_pcg::Pcg32::seed_from_u64(3);
            let mut map = MetroHashMap::with_capacity_and_hasher(size, MetroBuildHasher::default());
            while map.len() < size {
                let v = rng.gen_range(0..u32::MAX);
                if map.get(&v).is_none() {
                    map.insert(v, v);
                }
            }

            group.bench_with_input(BenchmarkId::new("MetroHashMap", size), &size, |b, _| {
                b.iter(|| black_box(map.get(&rng.gen_range(0..u32::MAX))));
            });
        }

        {
            let mut rng = rand_pcg::Pcg32::seed_from_u64(3);
            let mut map = MapSIMD::with_capacity(size);
            while map.len() < size {
                let v = rng.gen_range(0..u32::MAX);
                if map.get(v).is_none() {
                    map.try_insert(v, v).unwrap();
                }
            }

            group.bench_with_input(BenchmarkId::new("FastMap SIMD", size), &size, |b, _| {
                b.iter(|| black_box(map.get(rng.gen_range(0..u32::MAX))));
            });
        }
    }
}

criterion_group!(benches, bench_lookup);
criterion_main!(benches);
