use super::*;

fn create_test_node(id: i32, g_value: f64) -> SearchNode {
    // Use simple pool offset for test states.
    let state = ConcreteState::new(id as usize);
    let evaluation = EvaluationResult::new_with_id(state.get_id(), g_value, false);
    SearchNode::root(state, evaluation)
}

fn create_test_operator() -> Operator {
    // Create a test operator using the public constructor.
    Operator::new("test_op".to_string(), vec![], vec![], vec![], 3)
}

#[test]
fn test_search_node_basic() {
    let node = create_test_node(1, 10.0);

    assert_eq!(node.g_value(), 10.0);
    assert_eq!(node.depth(), 0);
    assert!(!node.is_dead_end());
    assert!(node.path().is_empty());
}

#[test]
fn test_search_node_successor() {
    let parent = Rc::new(create_test_node(1, 5.0));
    let child_state = create_test_node(2, 8.0).state;
    let child_eval = EvaluationResult::new_with_id(child_state.get_id(), 8.0, false);

    let operator = create_test_operator();

    let child = parent.successor(child_state, operator.clone(), child_eval);

    assert_eq!(child.g_value(), 8.0);
    assert_eq!(child.depth(), 1);
    assert_eq!(child.operator, Some(operator));
    assert!(child.parent.is_some());
}

#[test]
fn test_fifo_open_list() {
    let mut open_list = FifoOpenList::new();

    assert!(open_list.is_empty());
    assert_eq!(open_list.len(), 0);
    assert!(open_list.pop().is_none());

    let node1 = create_test_node(1, 10.0);
    let node2 = create_test_node(2, 20.0);

    open_list.insert(node1);
    open_list.insert(node2);

    assert!(!open_list.is_empty());
    assert_eq!(open_list.len(), 2);

    // FIFO: first inserted should be first out
    let popped1 = open_list.pop().unwrap();
    assert_eq!(popped1.g_value(), 10.0);

    let popped2 = open_list.pop().unwrap();
    assert_eq!(popped2.g_value(), 20.0);

    assert!(open_list.is_empty());
}

#[test]
fn test_lifo_open_list() {
    let mut open_list = LifoOpenList::new();

    let node1 = create_test_node(1, 10.0);
    let node2 = create_test_node(2, 20.0);

    open_list.insert(node1);
    open_list.insert(node2);

    // LIFO: last inserted should be first out
    let popped1 = open_list.pop().unwrap();
    assert_eq!(popped1.g_value(), 20.0);

    let popped2 = open_list.pop().unwrap();
    assert_eq!(popped2.g_value(), 10.0);

    assert!(open_list.is_empty());
}

#[test]
fn test_open_list_peek() {
    let mut fifo = FifoOpenList::new();
    let node = create_test_node(1, 42.0);

    assert!(fifo.peek().is_none());

    fifo.insert(node);

    let peeked = fifo.peek().unwrap();
    assert_eq!(peeked.g_value(), 42.0);
    assert_eq!(fifo.len(), 1); // Peek shouldn't remove

    let popped = fifo.pop().unwrap();
    assert_eq!(popped.g_value(), 42.0);
    assert!(fifo.peek().is_none());
}

#[test]
fn test_open_list_clear() {
    let mut open_list = FifoOpenList::new();

    open_list.insert(create_test_node(1, 10.0));
    open_list.insert(create_test_node(2, 20.0));

    assert_eq!(open_list.len(), 2);

    open_list.clear();

    assert!(open_list.is_empty());
    assert_eq!(open_list.len(), 0);
}

#[test]
fn test_required_evaluators() {
    let fifo = FifoOpenList::new();
    let lifo = LifoOpenList::new();

    assert!(fifo.required_evaluators().is_empty());
    assert!(lifo.required_evaluators().is_empty());
}
