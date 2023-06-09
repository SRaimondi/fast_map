use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use metrohash::{MetroHashMap, MetroHashSet};

use rand::{
    distributions::{Distribution, Uniform},
    SeedableRng,
};

use fast_map::FastMapU32;

fn bench_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("MetroHashMap - FastMap creation");

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

        {
            group.bench_with_input(BenchmarkId::new("MetroHashMap", size), &size, |b, _| {
                b.iter(|| {
                    let map: MetroHashMap<_, _> = indices.iter().map(|&i| (i, i)).collect();
                    black_box(map);
                });
            });
        }

        {
            group.bench_with_input(BenchmarkId::new("FastMap", size), &size, |b, _| {
                b.iter(|| {
                    let mut map = FastMapU32::with_capacity(size);
                    indices
                        .iter()
                        .for_each(|&i| unsafe { map.insert_direct(i, i) });
                    black_box(map);
                });
            });
        }
    }
}

fn bench_lookup_existing(c: &mut Criterion) {
    let mut group = c.benchmark_group("MetroHashMap - FastMap lookup existing key");

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
                    indices_sorted.iter().copied().for_each(|k| {
                        black_box(*map.get(&k).unwrap());
                    })
                });
            });
        }

        {
            let mut map = FastMapU32::with_capacity(size);
            indices.iter().for_each(|&i| map.try_insert(i, i).unwrap());

            group.bench_with_input(BenchmarkId::new("FastMap", size), &size, |b, _| {
                b.iter(|| {
                    indices_sorted.iter().copied().for_each(|k| {
                        black_box(unsafe { *map.get_existing(k) });
                    })
                });
            });
        }
    }
}

fn bench_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("MetroHashMap - FastMap search key");

    for size in (5..=22).map(|shift| 1 << shift) {
        let interval = Uniform::new(0, 4 * size as u32);
        let mut rng = rand::thread_rng();

        let indices: Vec<_> = {
            let mut added = MetroHashSet::default();
            while added.len() < size {
                added.insert(interval.sample(&mut rng));
            }

            added.into_iter().collect()
        };

        {
            let mut rng = rand_pcg::Pcg32::seed_from_u64(101);
            let map: MetroHashMap<_, _> = indices.iter().map(|&i| (i, i)).collect();

            group.bench_with_input(BenchmarkId::new("MetroHashMap", size), &size, |b, _| {
                b.iter(|| black_box(map.get(&interval.sample(&mut rng))));
            });
        }

        {
            let mut rng = rand_pcg::Pcg32::seed_from_u64(101);
            let map = FastMapU32::from_exact_iter(indices.iter().map(|&i| (i, i)));

            group.bench_with_input(BenchmarkId::new("FastMap", size), &size, |b, _| {
                b.iter(|| black_box(map.get(interval.sample(&mut rng))));
            });
        }
    }
}

criterion_group!(
    benches,
    bench_creation,
    //bench_lookup_existing,
    bench_search
);
criterion_main!(benches);
