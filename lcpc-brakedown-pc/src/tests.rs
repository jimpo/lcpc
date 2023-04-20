// Copyright 2021 Riad S. Wahby <rsw@cs.stanford.edu>
//
// This file is part of lcpc-brakedown-pc, which is part of lcpc.
//
// Licensed under the Apache License, Version 2.0 (see
// LICENSE or https://www.apache.org/licenses/LICENSE-2.0).
// This file may not be copied, modified, or distributed
// except according to those terms.

use super::{BrakedownCommit, SdigEncoding, SdigEncodingS};

use blake3::Hasher as Blake3;
use ff::Field;
use lcpc_2d::{LcCommit, LcEncoding};
use lcpc_test_fields::{ft255::*, ft191::*, random_coeffs};
use merlin::Transcript;
use ndarray::{linalg::Dot, Array};
use rand::{thread_rng, Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use sprs::{CsMat, TriMat};
use std::iter::repeat_with;

const fn log2(v: usize) -> usize {
    (63 - (v.next_power_of_two() as u64).leading_zeros()) as usize
}

#[test]
fn col_opens() {
    use super::codespec::{SdigCode1, SdigCode2, SdigCode3, SdigCode4, SdigCode5, SdigCode6};

    println!("1: {}", SdigEncodingS::<Ft255, SdigCode1>::_n_col_opens());
    println!("2: {}", SdigEncodingS::<Ft255, SdigCode2>::_n_col_opens());
    println!("3: {}", SdigEncodingS::<Ft255, SdigCode3>::_n_col_opens());
    println!("4: {}", SdigEncodingS::<Ft255, SdigCode4>::_n_col_opens());
    println!("5: {}", SdigEncodingS::<Ft255, SdigCode5>::_n_col_opens());
    println!("6: {}", SdigEncodingS::<Ft255, SdigCode6>::_n_col_opens());
}

#[test]
fn sprs_playground() {
    let mut rng = thread_rng();
    let n_rows = 65537;
    let n_cols = 32749;

    let m: CsMat<_> = {
        let mut tmp = TriMat::new((n_rows, n_cols));
        for i in 0..n_rows {
            let col1 = rng.gen_range(0..n_cols);
            let col2 = {
                let mut tmp = rng.gen_range(0..n_cols);
                while tmp == col1 {
                    tmp = rng.gen_range(0..n_cols);
                }
                tmp
            };
            tmp.add_triplet(i, col1, Ft191::random(&mut rng));
            tmp.add_triplet(i, col2, Ft191::random(&mut rng));
        }
        // to_csr appears to be considerably faster than to_csc
        // (note that because of the transpose, we end up with csc in the end)
        tmp.to_csr().transpose_into()
    };

    let v = {
        let mut tmp = Vec::with_capacity(n_rows);
        for _ in 0..n_rows {
            tmp.push(Ft191::random(&mut rng));
        }
        Array::from(tmp)
    };

    let mut t = Ft191::zero();
    for i in 0..10 {
        let mv = m.dot(&v);
        t += mv[i % n_cols];
    }
    println!("{:?}", t);
}

#[test]
fn test_matgen_encode() {
    let mut rng = thread_rng();
    use super::codespec::SdigCode3 as TestCode;
    use super::encode::{codeword_length, encode};
    use super::matgen::generate;

    let n = 256usize + (rng.gen::<usize>() % 4096);
    let (precodes, postcodes) = generate::<Ft191, TestCode>(n, 0u64);

    let xi_len = codeword_length(&precodes, &postcodes);
    let mut xi = Vec::with_capacity(xi_len);
    for _ in 0..xi_len {
        xi.push(Ft191::random(&mut rng));
    }
    encode(&mut xi, &precodes, &postcodes);
}

const N_ITERS: usize = 10;
#[test]
#[ignore]
fn prove_verify_size_bench() {
    use super::codespec::SdigCode3 as TestCode;
    use ff::PrimeField;
    use std::time::Instant;

    for lgl in (13..=29).step_by(2) {
        // commit to random poly of specified size
        let coeffs = random_coeffs(lgl);
        let enc = SdigEncodingS::<Ft255, TestCode>::new(coeffs.len(), 0);
        let comm = LcCommit::<Blake3, _>::commit(&coeffs, &enc).unwrap();
        let root = comm.get_root();

        // evaluate the random polynomial we just generated at a random point x
        let x = repeat_with(|| Ft255::random(&mut rand::thread_rng()))
            .take(log2(coeffs.len()))
            .collect::<Vec<_>>();

        let mut xxx = 0u8;
        let now = Instant::now();
        for i in 0..N_ITERS {
            let mut tr = Transcript::new(b"test transcript");
            tr.append_message(b"polycommit", root.as_ref());
            tr.append_message(b"ncols", &(enc.get_n_col_opens() as u64).to_be_bytes()[..]);
            let pf = comm.prove(&x, &enc, &mut tr).unwrap();
            let encoded: Vec<u8> = bincode::serialize(&pf).unwrap();
            xxx ^= encoded[i];
        }
        let pf_dur = now.elapsed().as_nanos() / N_ITERS as u128;

        let mut tr = Transcript::new(b"test transcript");
        tr.append_message(b"polycommit", root.as_ref());
        tr.append_message(b"ncols", &(enc.get_n_col_opens() as u64).to_be_bytes()[..]);
        let pf = comm.prove(&x, &enc, &mut tr).unwrap();
        let encoded: Vec<u8> = bincode::serialize(&pf).unwrap();
        let len = encoded.len();

        let now = Instant::now();
        for i in 0..N_ITERS {
            let mut tr = Transcript::new(b"test transcript");
            tr.append_message(b"polycommit", root.as_ref());
            tr.append_message(b"ncols", &(enc.get_n_col_opens() as u64).to_be_bytes()[..]);
            xxx ^= pf
                .verify(
                    root.as_ref(),
                    &x,
                    &enc,
                    &mut tr,
                )
                .unwrap()
                .to_repr()
                .as_ref()[i];
        }
        let vf_dur = now.elapsed().as_nanos() / N_ITERS as u128;

        println!("{}: {} {} {} {}", lgl, pf_dur, vf_dur, len, xxx);
    }
}

#[test]
#[ignore]
fn rough_bench() {
    use super::codespec::SdigCode3 as TestCode;
    use std::time::Instant;

    for lgl in (13..=29).step_by(2) {
        // commit to random poly of specified size
        let coeffs = random_coeffs(lgl);
        let enc = SdigEncodingS::<Ft255, TestCode>::new(coeffs.len(), 0);
        let mut xxx = 0u8;

        let now = Instant::now();
        for i in 0..N_ITERS {
            let comm = LcCommit::<Blake3, _>::commit(&coeffs, &enc).unwrap();
            let root = comm.get_root();
            xxx ^= root.as_ref()[i];
        }
        let dur = now.elapsed().as_nanos() / N_ITERS as u128;
        println!("{}: {} {:?}", lgl, dur, xxx);
    }
}

#[test]
fn end_to_end_one_proof_ml() {
    let mut rng = rand::thread_rng();

    // commit to a random polynomial at a random rate
    let lgl = 5 + rng.gen::<usize>() % 16;
    let coeffs = random_coeffs(lgl);
    let enc = SdigEncoding::new_ml(lgl, 0);
    let comm = BrakedownCommit::<Blake3, _>::commit(&coeffs, &enc).unwrap();
    // this is the polynomial commitment
    let root = comm.get_root();
    assert_eq!(1 << lgl, comm.get_n_rows() * comm.get_n_per_row());

    // evaluate the random polynomial we just generated at a random point x
    let x = repeat_with(|| Ft191::random(&mut rand::thread_rng()))
        .take(log2(coeffs.len()))
        .collect::<Vec<_>>();

    // compute an evaluation proof
    let mut tr1 = Transcript::new(b"test transcript");
    tr1.append_message(b"polycommit", root.as_ref());
    tr1.append_message(b"ncols", &(enc.get_n_col_opens() as u64).to_be_bytes()[..]);
    let pf = comm.prove(&x, &enc, &mut tr1).unwrap();

    // verify it and finish evaluation
    let mut tr2 = Transcript::new(b"test transcript");
    tr2.append_message(b"polycommit", root.as_ref());
    tr2.append_message(b"ncols", &(enc.get_n_col_opens() as u64).to_be_bytes()[..]);
    let enc2 = SdigEncoding::new_from_dims(pf.get_n_per_row(), pf.get_n_cols(), 0);
    pf.verify(
        root.as_ref(),
        &x,
        &enc2,
        &mut tr2,
    )
    .unwrap();
}

#[test]
fn end_to_end_two_proofs_ml() {
    let mut rng = rand::thread_rng();

    // commit to a random polynomial at a random rate
    let lgl = 5 + rng.gen::<usize>() % 16;
    let coeffs = random_coeffs(lgl);
    let enc = SdigEncoding::new_ml(lgl, 1);
    let comm = BrakedownCommit::<Blake3, _>::commit(&coeffs, &enc).unwrap();
    // this is the polynomial commitment
    let root = comm.get_root();

    // evaluate the random polynomial we just generated at a random point x
    let x = repeat_with(|| Ft191::random(&mut rand::thread_rng()))
        .take(log2(coeffs.len()))
        .collect::<Vec<_>>();

    // compute an evaluation proof
    let mut tr1 = Transcript::new(b"test transcript");
    tr1.append_message(b"polycommit", root.as_ref());
    tr1.append_message(b"ncols", &(enc.get_n_col_opens() as u64).to_be_bytes()[..]);
    let pf = comm.prove(&x, &enc, &mut tr1).unwrap();

    let challenge_after_first_proof_prover = {
        let mut key: <ChaCha20Rng as SeedableRng>::Seed = Default::default();
        tr1.challenge_bytes(b"ligero-pc//challenge", &mut key);
        let mut deg_test_rng = ChaCha20Rng::from_seed(key);
        Ft191::random(&mut deg_test_rng)
    };

    // produce a second proof with the same transcript
    tr1.append_message(b"polycommit", root.as_ref());
    tr1.append_message(b"ncols", &(enc.get_n_col_opens() as u64).to_be_bytes()[..]);
    let pf2 = comm.prove(&x, &enc, &mut tr1).unwrap();

    // verify it and finish evaluation
    let mut tr2 = Transcript::new(b"test transcript");
    tr2.append_message(b"polycommit", root.as_ref());
    tr2.append_message(b"ncols", &(enc.get_n_col_opens() as u64).to_be_bytes()[..]);
    let enc2 = SdigEncoding::new_from_dims(pf.get_n_per_row(), pf.get_n_cols(), 1);
    let res = pf
        .verify(
            root.as_ref(),
            &x,
            &enc2,
            &mut tr2,
        )
        .unwrap();

    let challenge_after_first_proof_verifier = {
        let mut key: <ChaCha20Rng as SeedableRng>::Seed = Default::default();
        tr2.challenge_bytes(b"ligero-pc//challenge", &mut key);
        let mut deg_test_rng = ChaCha20Rng::from_seed(key);
        Ft191::random(&mut deg_test_rng)
    };
    assert_eq!(
        challenge_after_first_proof_prover,
        challenge_after_first_proof_verifier
    );

    // second proof verification with the same transcript
    tr2.append_message(b"polycommit", root.as_ref());
    tr2.append_message(b"ncols", &(enc.get_n_col_opens() as u64).to_be_bytes()[..]);
    let enc3 = SdigEncoding::new_from_dims(pf2.get_n_per_row(), pf2.get_n_cols(), 1);
    let res2 = pf2
        .verify(
            root.as_ref(),
            &x,
            &enc3,
            &mut tr2,
        )
        .unwrap();

    assert_eq!(res, res2);
}
