use crate::parallel::{IntoParallelIterator, ParallelIterator};

#[derive(Debug, PartialEq, Eq)]
struct WholeStreamCount(usize);

impl std::iter::Sum<u8> for WholeStreamCount {
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = u8>,
    {
        Self(iter.count())
    }
}

impl std::iter::Product<u8> for WholeStreamCount {
    fn product<I>(iter: I) -> Self
    where
        I: Iterator<Item = u8>,
    {
        Self(iter.count() + 1)
    }
}

fn standard_sum_without_output_reassociation<I>(iter: I) -> WholeStreamCount
where
    I: ParallelIterator<Item = u8>,
{
    iter.sum()
}

fn standard_product_without_output_reassociation<I>(iter: I) -> WholeStreamCount
where
    I: ParallelIterator<Item = u8>,
{
    iter.product()
}

#[test]
fn standard_terminals_do_not_require_output_reassociation() {
    let data = vec![1_u8, 2, 3, 4];

    assert_eq!(
        standard_sum_without_output_reassociation(data.clone().into_par_iter()),
        WholeStreamCount(4)
    );
    assert_eq!(
        standard_product_without_output_reassociation(data.into_par_iter()),
        WholeStreamCount(5)
    );
}

#[derive(Debug, PartialEq, Eq)]
struct BatchSensitiveSum(usize);

impl std::iter::Sum<u8> for BatchSensitiveSum {
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = u8>,
    {
        let count = iter.count();
        Self(count * count)
    }
}

impl std::iter::Sum<Self> for BatchSensitiveSum {
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        Self(iter.map(|partial| partial.0).sum())
    }
}

#[derive(Debug, PartialEq, Eq)]
struct BatchSensitiveProduct(usize);

impl std::iter::Product<u8> for BatchSensitiveProduct {
    fn product<I>(iter: I) -> Self
    where
        I: Iterator<Item = u8>,
    {
        Self(iter.count() + 1)
    }
}

impl std::iter::Product<Self> for BatchSensitiveProduct {
    fn product<I>(iter: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        Self(iter.fold(1, |accumulator, partial| {
            accumulator.wrapping_mul(partial.0)
        }))
    }
}

#[test]
fn batch_sensitive_traits_distinguish_standard_and_reassociated_terminals() {
    const LEN: usize = 8_192;
    let data = vec![1_u8; LEN];

    assert_eq!(
        data.clone().into_par_iter().sum::<BatchSensitiveSum>(),
        BatchSensitiveSum(LEN * LEN)
    );
    assert_eq!(
        data.clone()
            .into_par_iter()
            .sum_reassociated::<BatchSensitiveSum>(),
        BatchSensitiveSum(LEN)
    );
    assert_eq!(
        data.clone()
            .into_par_iter()
            .product::<BatchSensitiveProduct>(),
        BatchSensitiveProduct(LEN + 1)
    );
    assert_eq!(
        data.into_par_iter()
            .product_reassociated::<BatchSensitiveProduct>(),
        BatchSensitiveProduct(0)
    );
}

#[test]
fn reassociated_arithmetic_preserves_primitive_identities_and_values() {
    let data = vec![1_u64, 2, 3, 4, 5];
    assert_eq!(data.clone().into_par_iter().sum_reassociated::<u64>(), 15);
    assert_eq!(data.into_par_iter().product_reassociated::<u64>(), 120);
    assert_eq!(
        Vec::<u64>::new().into_par_iter().sum_reassociated::<u64>(),
        0
    );
    assert_eq!(
        Vec::<u64>::new()
            .into_par_iter()
            .product_reassociated::<u64>(),
        1
    );
}
