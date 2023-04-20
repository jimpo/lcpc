// Copyright 2021 Riad S. Wahby <rsw@cs.stanford.edu>
//
// This file is part of lcpc-brakedown-pc, which is part of lcpc.
//
// Licensed under the Apache License, Version 2.0 (see
// LICENSE or https://www.apache.org/licenses/LICENSE-2.0).
// This file may not be copied, modified, or distributed
// except according to those terms.

use super::{BrakedownCommit, SdigEncoding};

use blake3::{Hasher as Blake3, traits::digest::{Digest, FixedOutputReset}};
use ff::{Field, PrimeField};
use lcpc_2d::{FieldHash, LcEncoding};
use merlin::Transcript;
use num_traits::Num;
use sprs::MulAcc;
use test::{black_box, Bencher};
use lcpc_test_fields::{def_bench, ft127::*, ft191::*, ft255::*, random_coeffs};
use std::iter::repeat_with;

#[bench]
fn matgen_bench(b: &mut Bencher) {
    use super::codespec::SdigCode3 as TestCode;
    use super::matgen::generate;

    b.iter(|| {
        generate::<Ft127, TestCode>(1048576, 0u64);
    })
}

fn commit_bench<D, Ft>(b: &mut Bencher, log_len: usize)
where
    D: Digest + FixedOutputReset,
    Ft: Field + FieldHash + MulAcc + Num + PrimeField,
{
    let coeffs = random_coeffs(log_len);
    let enc = SdigEncoding::new_ml(log_len, 0);

    b.iter(|| {
        black_box(BrakedownCommit::<D, Ft>::commit(&coeffs, &enc).unwrap());
    });
}

fn prove_bench<D, Ft>(b: &mut Bencher, log_len: usize)
where
    D: Digest + FixedOutputReset,
    Ft: Field + FieldHash + MulAcc + Num + PrimeField,
{
    let coeffs = random_coeffs(log_len);
    let enc = SdigEncoding::new_ml(log_len, 0);
    let comm = BrakedownCommit::<D, Ft>::commit(&coeffs, &enc).unwrap();

    // random point to eval at
    let x: Vec<Ft> = repeat_with(|| Ft::random(&mut rand::thread_rng()))
        .take(log_len)
        .collect();

    b.iter(|| {
        let mut tr = Transcript::new(b"bench transcript");
        tr.append_message(b"polycommit", comm.get_root().as_ref());
        tr.append_message(b"rate", &0.25f64.to_be_bytes()[..]);
        tr.append_message(b"ncols", &(enc.get_n_col_opens() as u64).to_be_bytes()[..]);
        tr.append_message(
            b"ndegs",
            &(enc.get_n_degree_tests() as u64).to_be_bytes()[..],
        );
        black_box(comm.prove(&x, &enc, &mut tr).unwrap());
    });
}

fn verify_bench<D, Ft>(b: &mut Bencher, log_len: usize)
where
    D: Digest + FixedOutputReset,
    Ft: Field + FieldHash + MulAcc + Num + PrimeField,
{
    let coeffs = random_coeffs(log_len);
    let enc = SdigEncoding::new_ml(log_len, 0);
    let comm = BrakedownCommit::<D, Ft>::commit(&coeffs, &enc).unwrap();

    // random point to eval at
    let x: Vec<Ft> = repeat_with(|| Ft::random(&mut rand::thread_rng()))
        .take(log_len)
        .collect();

    let mut tr = Transcript::new(b"bench transcript");
    tr.append_message(b"polycommit", comm.get_root().as_ref());
    tr.append_message(b"rate", &0.25f64.to_be_bytes()[..]);
    tr.append_message(b"ncols", &(enc.get_n_col_opens() as u64).to_be_bytes()[..]);
    tr.append_message(
        b"ndegs",
        &(enc.get_n_degree_tests() as u64).to_be_bytes()[..],
    );
    let pf = comm.prove(&x, &enc, &mut tr).unwrap();
    let root = comm.get_root();

    b.iter(|| {
        let mut tr = Transcript::new(b"bench transcript");
        tr.append_message(b"polycommit", comm.get_root().as_ref());
        tr.append_message(b"rate", &0.25f64.to_be_bytes()[..]);
        tr.append_message(b"ncols", &(enc.get_n_col_opens() as u64).to_be_bytes()[..]);
        tr.append_message(
            b"ndegs",
            &(enc.get_n_degree_tests() as u64).to_be_bytes()[..],
        );
        black_box(
            pf.verify(
                root.as_ref(),
                &x,
                &enc,
                &mut tr,
            )
            .unwrap(),
        );
    });
}

def_bench!(commit, Ft127, Blake3, 16);
def_bench!(commit, Ft127, Blake3, 20);
def_bench!(commit, Ft127, Blake3, 24);

def_bench!(prove, Ft127, Blake3, 16);
def_bench!(prove, Ft127, Blake3, 20);
def_bench!(prove, Ft127, Blake3, 24);

def_bench!(verify, Ft127, Blake3, 16);
def_bench!(verify, Ft127, Blake3, 20);
def_bench!(verify, Ft127, Blake3, 24);

def_bench!(commit, Ft191, Blake3, 16);
def_bench!(commit, Ft191, Blake3, 20);
def_bench!(commit, Ft191, Blake3, 24);

def_bench!(prove, Ft191, Blake3, 16);
def_bench!(prove, Ft191, Blake3, 20);
def_bench!(prove, Ft191, Blake3, 24);

def_bench!(verify, Ft191, Blake3, 16);
def_bench!(verify, Ft191, Blake3, 20);
def_bench!(verify, Ft191, Blake3, 24);

def_bench!(commit, Ft255, Blake3, 16);
def_bench!(commit, Ft255, Blake3, 20);
def_bench!(commit, Ft255, Blake3, 24);

def_bench!(prove, Ft255, Blake3, 16);
def_bench!(prove, Ft255, Blake3, 20);
def_bench!(prove, Ft255, Blake3, 24);

def_bench!(verify, Ft255, Blake3, 16);
def_bench!(verify, Ft255, Blake3, 20);
def_bench!(verify, Ft255, Blake3, 24);
