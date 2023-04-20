// Copyright 2023 Ulvetanna, Inc.

use ff::Field;
use rayon::prelude::*;

/// Pre-condition: length of q must be <32
pub fn expand_query<F: Field>(q: &[F]) -> Vec<F> {
	assert!(q.len() < 32, "precondition: q must have length <32");
	let size = 2usize.pow(q.len() as u32);

	let mut result = Vec::with_capacity(size);
	result.push(F::one());
	for v in q.iter() {
		let mid = result.len();
		result.resize(mid << 1, F::zero());
		let (left, right) = result.split_at_mut(mid);
		left.par_iter_mut()
			.zip(right.par_iter_mut())
			.for_each(|(left_i, right_i)| {
				let prod = *left_i * *v;
				*left_i -= prod;
				*right_i = prod;
			});
	}
	result
}

#[cfg(test)]
mod tests {
	use super::*;
	use rand::{SeedableRng, rngs::StdRng};
	use std::iter::repeat_with;
	use lcpc_test_fields::ft63::*;

	fn expand_query_naive<F: Field>(q: &[F]) -> Vec<F> {
		assert!(q.len() < 32, "precondition: q must have length <32");
		let size = 2usize.pow(q.len() as u32);

		(0..size)
			.map(|i| eval_basis_unchecked(q, i))
			.collect()
	}

	fn eval_basis_unchecked<F: Field>(q: &[F], i: usize) -> F {
		q.iter()
			.enumerate()
			.map(|(j, &v)| {
				if i & (1 << j) == 0 {
					F::one() - v
				} else {
					v
				}
			})
			.fold(F::one(), |prod, term| prod * term)
	}

	#[test]
	fn test_expand_query_impls_consistent() {
		let mut rng = StdRng::seed_from_u64(0);
		let q = repeat_with(|| Ft63::random(&mut rng))
			.take(8)
			.collect::<Vec<_>>();
		let result1 = expand_query(&q);
		let result2 = expand_query_naive(&q);
		assert_eq!(result1, result2);
	}
}
