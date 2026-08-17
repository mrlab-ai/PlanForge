use std::cmp::Ordering;
use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::hash::Hash;
use std::time::Duration;

/// The CPU time this process has used so far, user and system.
///
/// The invariant synthesis works to a time budget, and against a wall clock the
/// budget decides how many invariants are found by how busy the machine was --
/// which makes the SAS encoding, and with it every number measured from it,
/// depend on the load. Mainline Fast Downward reads `time.process_time()` here
/// for the same reason.
pub fn process_cpu_time() -> Duration {
    let mut spent = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: `clock_gettime` writes through the pointer we hand it and nowhere
    // else, and the struct is fully initialized.
    let result = unsafe { libc::clock_gettime(libc::CLOCK_PROCESS_CPUTIME_ID, &mut spent) };
    assert_eq!(
        result,
        0,
        "the process CPU clock is not readable: {}",
        std::io::Error::last_os_error()
    );
    Duration::new(
        spent.tv_sec.try_into().expect("CPU time is not negative"),
        spent.tv_nsec.try_into().expect("nanoseconds fit in a u32"),
    )
}

/// A set that remembers the order things were inserted in.
///
/// The translator repeatedly needs both: membership has to be cheap, because
/// the alternative in several places was a linear `Vec::contains` inside a
/// loop over the same vector, and the order has to be reproducible, because it
/// reaches the SAS task. A `HashSet` gives the first and not the second.
#[derive(Debug, Clone)]
pub struct OrderedSet<T> {
    positions: HashMap<T, usize>,
}

impl<T: Eq + Hash> Default for OrderedSet<T> {
    fn default() -> Self {
        OrderedSet {
            positions: HashMap::new(),
        }
    }
}

impl<T: Eq + Hash> OrderedSet<T> {
    /// Inserts `value` and reports whether it was new. An element keeps the
    /// position it first got.
    pub fn insert(&mut self, value: T) -> bool {
        let next = self.positions.len();
        match self.positions.entry(value) {
            Entry::Occupied(_) => false,
            Entry::Vacant(slot) => {
                slot.insert(next);
                true
            }
        }
    }

    /// The elements, in insertion order.
    pub fn into_vec(self) -> Vec<T> {
        let mut values: Vec<(T, usize)> = self.positions.into_iter().collect();
        values.sort_unstable_by_key(|&(_, position)| position);
        values.into_iter().map(|(value, _)| value).collect()
    }
}

impl<T: Eq + Hash> FromIterator<T> for OrderedSet<T> {
    fn from_iter<I: IntoIterator<Item = T>>(values: I) -> Self {
        let mut set = OrderedSet::default();
        for value in values {
            set.insert(value);
        }
        set
    }
}

/// Compares two names the way `format!("{:?}", name)` compares them.
///
/// `Debug` for `str` wraps the name in quotes, so the closing quote terminates
/// it: `derived!` sorts *before* `derived` because `!` is below `"`. Several
/// translator orderings were written as comparisons of `Debug` output and the
/// SAS variable order depends on them, so they are reproduced here rather than
/// formatted afresh at every comparison.
pub fn cmp_quoted(left: &str, right: &str) -> Ordering {
    debug_assert!(
        !left.contains(['"', '\\']) && !right.contains(['"', '\\']),
        "name needs Debug escaping, so the quote is no longer a terminator: {left:?} {right:?}"
    );
    left.bytes()
        .chain(std::iter::once(b'"'))
        .cmp(right.bytes().chain(std::iter::once(b'"')))
}

/// Compares two name lists the way `Debug` compares them: element by element,
/// and a list that still has elements sorts before one that ended, because `,`
/// is below `]`.
pub fn cmp_quoted_slice(left: &[String], right: &[String]) -> Ordering {
    for (left, right) in left.iter().zip(right) {
        match cmp_quoted(left, right) {
            Ordering::Equal => {}
            order => return order,
        }
    }
    right.len().cmp(&left.len())
}

/// Visits the Cartesian product of `lists` without materializing it.
///
/// The empty product is visited once with an empty selection. Returning
/// `false` from `visit` stops enumeration immediately; the return value says
/// whether every combination was visited.
pub fn for_each_product<'a, T>(
    lists: &'a [Vec<T>],
    visit: &mut impl FnMut(&[&'a T]) -> bool,
) -> bool {
    for_each_product_inner(lists, None, visit).expect("a product without a deadline cannot expire")
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CpuTimeDeadlineExceeded;

/// The deadline-aware form of [`for_each_product`].
pub fn for_each_product_before<'a, T>(
    lists: &'a [Vec<T>],
    deadline: Duration,
    visit: &mut impl FnMut(&[&'a T]) -> bool,
) -> Result<bool, CpuTimeDeadlineExceeded> {
    for_each_product_inner(lists, Some(deadline), visit)
}

fn for_each_product_inner<'a, T>(
    lists: &'a [Vec<T>],
    deadline: Option<Duration>,
    visit: &mut impl FnMut(&[&'a T]) -> bool,
) -> Result<bool, CpuTimeDeadlineExceeded> {
    fn recurse<'a, T>(
        lists: &'a [Vec<T>],
        depth: usize,
        selection: &mut Vec<&'a T>,
        deadline: Option<Duration>,
        visit: &mut impl FnMut(&[&'a T]) -> bool,
    ) -> Result<bool, CpuTimeDeadlineExceeded> {
        if deadline.is_some_and(|limit| process_cpu_time() >= limit) {
            return Err(CpuTimeDeadlineExceeded);
        }
        if depth == lists.len() {
            return Ok(visit(selection));
        }
        for item in &lists[depth] {
            selection.push(item);
            let keep_going = recurse(lists, depth + 1, selection, deadline, visit)?;
            selection.pop();
            if !keep_going {
                return Ok(false);
            }
        }
        Ok(true)
    }

    recurse(
        lists,
        0,
        &mut Vec::with_capacity(lists.len()),
        deadline,
        visit,
    )
}

#[cfg(test)]
mod product_tests {
    use std::time::Duration;

    use super::{for_each_product, for_each_product_before};

    #[test]
    fn empty_product_has_one_empty_combination() {
        let lists: Vec<Vec<usize>> = vec![];
        let mut combinations = Vec::new();
        assert!(for_each_product(&lists, &mut |selection| {
            combinations.push(selection.iter().map(|&&value| value).collect::<Vec<_>>());
            true
        }));
        assert_eq!(combinations, [Vec::<usize>::new()]);
    }

    #[test]
    fn product_visitor_stops_without_enumerating_the_tail() {
        let lists = vec![vec![1, 2, 3], vec![10, 20, 30]];
        let mut combinations = Vec::new();
        assert!(!for_each_product(&lists, &mut |selection| {
            combinations.push((selection[0], selection[1]));
            combinations.len() < 2
        }));
        assert_eq!(combinations, [(&1, &10), (&1, &20)]);
    }

    #[test]
    fn expired_deadline_interrupts_before_the_first_combination() {
        let lists = vec![vec![1, 2], vec![10, 20]];
        let mut visited = 0;
        assert!(
            for_each_product_before(&lists, Duration::ZERO, &mut |_| {
                visited += 1;
                true
            })
            .is_err()
        );
        assert_eq!(visited, 0);
    }
}
