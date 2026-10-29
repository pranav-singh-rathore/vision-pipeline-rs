//! Minimum-cost rectangular assignment (Hungarian algorithm with potentials, O(n^2 m)).
//!
//! Returns (row, col) pairs covering every row when rows <= cols, or every column
//! otherwise, which is what `scipy.optimize.linear_sum_assignment` returns.

pub fn linear_sum_assignment(cost: &[f64], rows: usize, cols: usize) -> Vec<(usize, usize)> {
    if rows == 0 || cols == 0 {
        return Vec::new();
    }
    if rows <= cols {
        solve(|r, c| cost[r * cols + c], rows, cols)
    } else {
        let mut pairs: Vec<(usize, usize)> = solve(|r, c| cost[c * cols + r], cols, rows)
            .into_iter()
            .map(|(c, r)| (r, c))
            .collect();
        pairs.sort_unstable();
        pairs
    }
}

/// Requires n <= m. Indices are 1-based internally, with 0 as the virtual column.
fn solve(a: impl Fn(usize, usize) -> f64, n: usize, m: usize) -> Vec<(usize, usize)> {
    let mut u = vec![0.0; n + 1];
    let mut v = vec![0.0; m + 1];
    let mut p = vec![0usize; m + 1]; // p[j] = row assigned to column j
    let mut way = vec![0usize; m + 1];
    let mut minv = vec![0.0; m + 1];
    let mut used = vec![false; m + 1];

    for i in 1..=n {
        p[0] = i;
        let mut j0 = 0;
        minv.fill(f64::INFINITY);
        used.fill(false);
        loop {
            used[j0] = true;
            let i0 = p[j0];
            let mut delta = f64::INFINITY;
            let mut j1 = 0;
            for j in 1..=m {
                if !used[j] {
                    let cur = a(i0 - 1, j - 1) - u[i0] - v[j];
                    if cur < minv[j] {
                        minv[j] = cur;
                        way[j] = j0;
                    }
                    if minv[j] < delta {
                        delta = minv[j];
                        j1 = j;
                    }
                }
            }
            for j in 0..=m {
                if used[j] {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
            if p[j0] == 0 {
                break;
            }
        }
        loop {
            let j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
            if j0 == 0 {
                break;
            }
        }
    }

    let mut pairs: Vec<(usize, usize)> = (1..=m)
        .filter(|&j| p[j] != 0)
        .map(|j| (p[j] - 1, j - 1))
        .collect();
    pairs.sort_unstable();
    pairs
}

#[cfg(test)]
mod tests {
    use super::*;

    fn total(cost: &[f64], cols: usize, pairs: &[(usize, usize)]) -> f64 {
        pairs.iter().map(|&(r, c)| cost[r * cols + c]).sum()
    }

    #[test]
    fn square() {
        let cost = [4.0, 1.0, 3.0, 2.0, 0.0, 5.0, 3.0, 2.0, 2.0];
        let pairs = linear_sum_assignment(&cost, 3, 3);
        assert_eq!(pairs, vec![(0, 1), (1, 0), (2, 2)]);
        assert_eq!(total(&cost, 3, &pairs), 5.0);
    }

    #[test]
    fn more_rows_than_cols() {
        // 3 rows, 2 cols: best is row 0 -> col 1, row 2 -> col 0.
        let cost = [5.0, 1.0, 9.0, 9.0, 2.0, 7.0];
        let pairs = linear_sum_assignment(&cost, 3, 2);
        assert_eq!(pairs, vec![(0, 1), (2, 0)]);
    }

    #[test]
    fn more_cols_than_rows() {
        let cost = [3.0, 1.0, 2.0];
        assert_eq!(linear_sum_assignment(&cost, 1, 3), vec![(0, 1)]);
    }

    #[test]
    fn empty() {
        assert!(linear_sum_assignment(&[], 0, 4).is_empty());
    }
}
