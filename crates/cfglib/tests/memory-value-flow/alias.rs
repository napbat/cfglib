//! Alias oracles: structured index paths, byte ranges, and the guarantee
//! that the analysis closes whatever relation it is given.

use cfglib::{
    BlockId, ConservativeMemoryAlias, ExactMemoryAlias, MemoryAlias, MemoryDefinition,
    MemoryEventSite, MemorySsa, ProgramPoint, index_paths_may_overlap,
};

use crate::machine::{Barrier, Inst, chain};

/// One component of an index path: a constant, or a value known only at run
/// time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Index {
    Constant(u8),
    Dynamic,
}

/// The projection of a component to its compile-time value, when it has one.
fn fixed() -> impl Fn(&Index) -> Option<u8> + Copy {
    |index| match index {
        Index::Constant(value) => Some(*value),
        Index::Dynamic => None,
    }
}

/// Every path of up to three components over two constants and a dynamic
/// index.
fn all_paths() -> Vec<Vec<Index>> {
    let alphabet = [Index::Constant(0), Index::Constant(1), Index::Dynamic];
    let mut paths = vec![Vec::new()];
    let mut frontier = paths.clone();
    for _ in 0..3 {
        frontier = frontier
            .iter()
            .flat_map(|path: &Vec<Index>| {
                alphabet.iter().map(move |index| {
                    let mut longer = path.clone();
                    longer.push(*index);
                    longer
                })
            })
            .collect();
        paths.extend(frontier.clone());
    }
    paths
}

#[test]
fn index_paths_overlap_unless_two_known_components_disagree_or_the_arities_match_and_differ() {
    let paths = all_paths();
    assert_eq!(paths.len(), 1 + 3 + 9 + 27);
    for left in &paths {
        for right in &paths {
            let expected = if left.len() == right.len() {
                !left.iter().zip(right).any(
                    |(a, b)| matches!((a, b), (Index::Constant(x), Index::Constant(y)) if x != y),
                )
            } else {
                true
            };
            assert_eq!(
                index_paths_may_overlap(left, right, fixed()),
                expected,
                "{left:?} vs {right:?}"
            );
            assert_eq!(
                index_paths_may_overlap(right, left, fixed()),
                index_paths_may_overlap(left, right, fixed()),
                "symmetric"
            );
        }
        assert!(
            index_paths_may_overlap(left, left, fixed()),
            "a path overlaps itself"
        );
    }
}

#[test]
fn the_projection_is_only_asked_about_components_of_equal_arity_paths() {
    let mut asked = 0;
    let overlap = index_paths_may_overlap(&[1, 2], &[1, 2, 3], |component: &i32| {
        asked += 1;
        Some(*component)
    });
    assert!(overlap, "unequal arities overlap conservatively");
    assert_eq!(asked, 0, "no projection is needed to decide it");
    assert!(!index_paths_may_overlap(
        &[1, 2],
        &[1, 3],
        |component: &i32| Some(*component)
    ));
}

/// An object, a path of indices into it.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct Element {
    object: &'static str,
    path: Vec<Index>,
}

struct Indexed;

impl MemoryAlias<Element> for Indexed {
    fn may_alias(&self, left: &Element, right: &Element) -> bool {
        left.object == right.object && index_paths_may_overlap(&left.path, &right.path, fixed())
    }
}

fn element(object: &'static str, path: &[Index]) -> Element {
    Element {
        object,
        path: path.to_vec(),
    }
}

#[test]
fn an_indexed_oracle_separates_constant_indices_unless_a_dynamic_one_bridges_them() {
    let (zero, one, any) = (
        element("a", &[Index::Constant(0)]),
        element("a", &[Index::Constant(1)]),
        element("a", &[Index::Dynamic]),
    );
    // a[0] = v1; v2 = a[1]
    let separate = chain(vec![vec![
        Inst::store(zero.clone(), 1),
        Inst::load(one.clone(), 2),
    ]]);
    let memory: MemorySsa<Element, u8, Barrier> = MemorySsa::compute(&separate, &Indexed);
    assert_eq!(memory.classes().len(), 2);
    assert!(!memory.may_alias(&zero, &one));
    assert_eq!(
        memory.reaching_definitions(MemoryEventSite::new(point(0, 1), 0)),
        [&MemoryDefinition::LiveIn {
            class: memory.class_of(&one).unwrap()
        }]
    );

    // A dynamic access a[?] overlaps both, and the analysis closes the
    // relation: a[0] and a[1] end up in one class although they are disjoint.
    let bridged = chain(vec![vec![
        Inst::store(zero.clone(), 1),
        Inst::load(one.clone(), 2),
        Inst::load(any.clone(), 3),
    ]]);
    let memory: MemorySsa<Element, u8, Barrier> = MemorySsa::compute(&bridged, &Indexed);
    assert_eq!(memory.classes().len(), 1);
    assert!(memory.may_alias(&zero, &one));
    assert_eq!(
        memory.reaching_definitions(MemoryEventSite::new(point(0, 1), 0)),
        [&MemoryDefinition::Event {
            site: MemoryEventSite::new(point(0, 0), 0)
        }]
    );
}

fn point(block: usize, inst_idx: usize) -> ProgramPoint {
    ProgramPoint {
        block: BlockId::from_index(block),
        inst_idx,
    }
}

/// A byte range of an object: a custom oracle that is not an equality.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct Bytes {
    object: u8,
    start: u32,
    length: u32,
}

struct Overlapping;

impl MemoryAlias<Bytes> for Overlapping {
    fn may_alias(&self, left: &Bytes, right: &Bytes) -> bool {
        left.object == right.object
            && left.start < right.start + right.length
            && right.start < left.start + left.length
    }
}

fn bytes(object: u8, start: u32, length: u32) -> Bytes {
    Bytes {
        object,
        start,
        length,
    }
}

#[test]
fn a_range_oracle_that_is_not_transitive_is_closed_into_classes() {
    let (first, second, third, apart, other_object) = (
        bytes(0, 0, 4),
        bytes(0, 2, 4),
        bytes(0, 5, 3),
        bytes(0, 10, 2),
        bytes(1, 0, 4),
    );
    assert!(Overlapping.may_alias(&first, &second));
    assert!(Overlapping.may_alias(&second, &third));
    assert!(
        !Overlapping.may_alias(&first, &third),
        "the oracle itself is not transitive"
    );

    let cfg = chain(vec![vec![
        Inst::store(first, 1),
        Inst::store(second, 2),
        Inst::store(third, 3),
        Inst::store(apart, 4),
        Inst::store(other_object, 5),
    ]]);
    let memory: MemorySsa<Bytes, u8, Barrier> = MemorySsa::compute(&cfg, &Overlapping);
    assert!(
        memory.may_alias(&first, &third),
        "closed through the middle range"
    );
    assert!(!memory.may_alias(&first, &apart));
    assert!(!memory.may_alias(&first, &other_object));
    assert_eq!(memory.classes().len(), 3);
    for (position, class) in memory.classes().iter().enumerate() {
        assert_eq!(
            class.id().index(),
            position,
            "class identities are dense and ordered"
        );
        assert!(class.locations().windows(2).all(|pair| pair[0] < pair[1]));
    }
    let merged = memory.class(memory.class_of(&first).unwrap()).unwrap();
    assert_eq!(merged.locations(), [first, second, third]);
    assert!(
        !memory.may_alias(&first, &bytes(7, 0, 1)),
        "an unseen location aliases nothing"
    );
}

#[test]
fn the_two_stock_oracles_are_one_class_per_location_and_one_class_overall() {
    let cfg = chain(vec![vec![
        Inst::store(0_u8, 1),
        Inst::store(1, 2),
        Inst::store(2, 3),
    ]]);
    let exact: MemorySsa<u8, u8, Barrier> = MemorySsa::compute(&cfg, &ExactMemoryAlias);
    assert_eq!(exact.classes().len(), 3);
    assert!(exact.may_alias(&0, &0) && !exact.may_alias(&0, &1));

    let conservative: MemorySsa<u8, u8, Barrier> =
        MemorySsa::compute(&cfg, &ConservativeMemoryAlias);
    assert_eq!(conservative.classes().len(), 1);
    assert_eq!(conservative.classes()[0].locations(), [0, 1, 2]);
    assert!(conservative.may_alias(&0, &2));
}

#[test]
fn an_asymmetric_oracle_is_symmetrized() {
    // The oracle only answers true in one direction; either direction merges.
    let one_way = |left: &u8, right: &u8| *left == 0 && *right == 1;
    let cfg = chain(vec![vec![Inst::store(1_u8, 1), Inst::store(0, 2)]]);
    let memory: MemorySsa<u8, u8, Barrier> = MemorySsa::compute(&cfg, &one_way);
    assert!(memory.may_alias(&0, &1));
    assert!(memory.may_alias(&1, &0));
    assert_eq!(memory.classes().len(), 1);
}
