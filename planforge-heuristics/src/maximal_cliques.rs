use std::collections::BTreeSet;

pub(crate) fn maximal_cliques(
    vertex_count: usize,
    compatible: impl Fn(usize, usize) -> bool,
) -> Vec<Vec<usize>> {
    let mut graph = vec![Vec::new(); vertex_count];
    for left in 0..vertex_count {
        for right in (left + 1)..vertex_count {
            if compatible(left, right) {
                graph[left].push(right);
                graph[right].push(left);
            }
        }
    }

    let mut result = Vec::new();
    bron_kerbosch(
        &graph,
        &mut Vec::new(),
        (0..vertex_count).collect(),
        Vec::new(),
        &mut result,
    );
    result
}

fn bron_kerbosch(
    graph: &[Vec<usize>],
    current: &mut Vec<usize>,
    candidates: Vec<usize>,
    excluded: Vec<usize>,
    maximal_cliques: &mut Vec<Vec<usize>>,
) {
    if candidates.is_empty() && excluded.is_empty() {
        let mut clique = current.clone();
        clique.sort_unstable();
        maximal_cliques.push(clique);
        return;
    }

    let pivot = candidates
        .iter()
        .chain(excluded.iter())
        .copied()
        .max_by_key(|&vertex| graph[vertex].len());
    let pivot_neighbors: BTreeSet<_> = pivot
        .map(|vertex| graph[vertex].iter().copied().collect())
        .unwrap_or_default();
    let mut remaining_candidates = candidates.clone();
    let mut local_excluded = excluded;
    let vertices: Vec<_> = candidates
        .iter()
        .copied()
        .filter(|candidate| !pivot_neighbors.contains(candidate))
        .collect();

    for vertex in vertices {
        current.push(vertex);
        let neighbors: BTreeSet<_> = graph[vertex].iter().copied().collect();
        let next_candidates = remaining_candidates
            .iter()
            .copied()
            .filter(|candidate| neighbors.contains(candidate))
            .collect();
        let next_excluded = local_excluded
            .iter()
            .copied()
            .filter(|candidate| neighbors.contains(candidate))
            .collect();
        bron_kerbosch(
            graph,
            current,
            next_candidates,
            next_excluded,
            maximal_cliques,
        );
        current.pop();
        remaining_candidates.retain(|candidate| *candidate != vertex);
        local_excluded.push(vertex);
    }
}
