use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use metrohash::{MetroBuildHasher, MetroHashMap, MetroHashSet};
use rand::{
    distributions::{Distribution, Uniform},
    Rng, SeedableRng,
};

use fast_map::{FastMap, U32KeysBlock};

type MapSIMD = FastMap<U32KeysBlock, u32>;

fn bench_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("MetroHashMap vs FastMap creation");

    for size in (5..=22).map(|shift| 1 << shift) {
        let mut rng = rand_pcg::Pcg32::seed_from_u64(3);
        let interval = Uniform::new(0, size);
        let mut found_map =
            MetroHashMap::with_capacity_and_hasher(size, MetroBuildHasher::default());

        let mut index: u32 = 0;
        while found_map.len() < size {
            let value = interval.sample(&mut rng);
            if found_map.get(&value).is_none() {
                found_map.insert(value, index);
                index += 1;
            }
        }
        assert_eq!(index as usize, size);

        group.bench_with_input(BenchmarkId::new("MetroHashMap", size), &size, |b, _| {
            b.iter(|| {
                let map: MetroHashMap<u32, u32> = (0..size)
                    .map(|_| {
                        let v = rng.gen_range(0..u32::MAX);
                        (v, v)
                    })
                    .collect();
                black_box(map);
            })
        });

        let mut rng = rand_pcg::Pcg32::seed_from_u64(3);
        group.bench_with_input(BenchmarkId::new("FastMap SIMD", size), &size, |b, _| {
            b.iter(|| {
                let mut map = MapSIMD::with_capacity(size);
                (0..size)
                    .map(|_| rng.gen_range(0..u32::MAX))
                    .for_each(|key| {
                        map.try_insert(key, key).unwrap();
                    });
                black_box(map);
            })
        });
    }
}

fn bench_lookup(c: &mut Criterion) {
    let mut group = c.benchmark_group("MetroHashMap vs FastMap lookup");

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
            let map: MetroHashMap<_, _> = indices.iter().map(|&i| (i, i)).collect();
            group.bench_with_input(BenchmarkId::new("MetroHashMap", size), &size, |b, _| {
                b.iter(|| {
                    indices_sorted.iter().for_each(|k| {
                        black_box(unsafe { *map.get(k).unwrap_unchecked() });
                    })
                });
            });
        }

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
