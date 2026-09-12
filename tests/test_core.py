import ast
import inspect
import sys
import typing

import itertools
import pytest

from pure_eval import Evaluator, CannotEval, EvaluationLimit
from pure_eval.core import is_expression_interesting, group_expressions


def check_eval(source, *expected_values, total=True):
    frame = inspect.currentframe().f_back
    evaluator = Evaluator.from_frame(frame)
    root = ast.parse(source)
    values = []
    for node, value in evaluator.find_expressions(root):
        expr = ast.Expression(body=node)
        ast.copy_location(expr, node)
        code = compile(expr, "<expr>", "eval")
        expected = eval(code, frame.f_globals, frame.f_locals)
        assert value == expected
        values.append(value)
        if total:
            assert value in expected_values

    for expected in expected_values:
        assert expected in values


def test_eval_names():
    x = 3
    check_eval(
        "(x, check_eval, len), nonexistent",
        x, check_eval, len, (x, check_eval, len)
    )


def test_eval_literals():
    check_eval(
        "(1, 'a', [{}])",
        (1, 'a', [{}]),
        1, 'a', [{}],
        {},
    )


def test_eval_attrs():
    class Foo:
        bar = 9

        @property
        def prop(self):
            return 0

        def method(self):
            pass

    foo = Foo()
    foo.spam = 44

    check_eval(
        "foo.bar + foo.spam + Foo.bar",
        foo.bar, foo.spam, Foo.bar,
        foo.bar + foo.spam,
        foo.bar + foo.spam + Foo.bar,
        foo, Foo
    )

    check_eval(
        "Foo.spam + Foo.prop + foo.prop + foo.method() + Foo.method",
        foo, Foo, Foo.method, foo.method
    )

    check_eval("typing.List", typing, typing.List)


def test_eval_dict():
    d = {1: 2}

    # All is well, d[1] is evaluated
    check_eval(
        "d[1]",
        d[1], d, 1
    )

    class BadHash:
        def __hash__(self):
            return 0

    d[BadHash()] = 3

    # d[1] is not evaluated because d contains a bad key
    check_eval(
        "d[1]",
        d, 1
    )

    d = {1: 2}
    b = BadHash()

    # d[b] is not evaluated because b is a bad key
    check_eval(
        "d[b]",
        d, b
    )

    def make_d():
        return {1: 2}

    str(make_d())

    # Cannot eval make_d()[1] because the left part cannot eval
    check_eval(
        "make_d()[1]",
        make_d, 1
    )

    # Cannot eval d[:1] because slices aren't hashable
    check_eval(
        "d[:1]",
        d, 1
    )

    d = {(1, 3): 2}
    b = BadHash()

    # d[(1, b)] is not evaluated because b is a bad key
    check_eval(
        "d[(1, b)], d[(1, 3)]",
        # (1, b) is a bad key, but it's a valid tuple element
        d, b, d[(1, 3)], (1, 3), 1, 3, (1, b)
    )

    e = 3
    check_eval(
        "{(1, e): 2}, {(1, b): 1}",  # b is a bad key
        b, 1, (1, e), 2, e, {(1, e): 2}, (1, b)
    )

    check_eval("{{}: {}}", {})


def test_eval_set():
    a = 1
    b = {2, 3}  # unhashable itself
    check_eval(
        "{a}, b, {a, b, 4}, {b}",  # d is a bad key
        a, {a}, b, 4
    )


def test_eval_sequence_subscript():
    lst = [12, 34, 56]
    i = 1
    check_eval(
        "lst[i] + lst[:i][0] + lst[i:][i] + lst[::2][False]",
        lst[i], lst[:i][0], lst[i:][i], lst[::2],
        lst[i] + lst[:i][0],
        lst[i] + lst[:i][0] + lst[i:][i],
        lst[i] + lst[:i][0] + lst[i:][i] + lst[::2][False],
        lst, i, lst[:i], 0, lst[i:], 2,
    )

    check_eval(
        "('abc', 'def')[1][2]",
        ('abc', 'def')[1][2],
        total=False
    )

    check_eval(
        "[lst][0][2]",
        lst, [lst], [lst][0], [lst][0][2], 2, 0
    )

    check_eval(
        "(lst, )[0][2]",
        lst, (lst, ), (lst, )[0], (lst, )[0][2], 2, 0
    )


def test_eval_unary_op():
    a = 123
    check_eval(
        "a, -a, +a, ~a",
        a, -a, +a, ~a,
        (a, -a, +a, ~a),
    )
    check_eval(
        "not a",
        a, not a,
    )
    b = ""
    check_eval(
        "not b, -b",
        b, not b,
    )


def test_eval_binary_op():
    a = 123
    b = 456
    check_eval(
        "a + b - a * b - (a ** b) // (b % a)",
         a + b - a * b - (a ** b) // (b % a),
         a + b, a * b, (a ** b), (b % a),
         a + b - a * b, (a ** b) // (b % a),
         a, b,
    )
    check_eval(
        "a / b",
         a / b, a, b,
    )
    check_eval(
        "a & b",
         a & b, a, b,
    )
    check_eval(
        "a | b",
         a | b, a, b,
    )
    check_eval(
        "a ^ b",
         a ^ b, a, b,
    )
    check_eval(
        "a << 2",
         a << 2, a, 2
    )
    check_eval(
        "a >> 2",
         a >> 2, a, 2
    )
    check_eval(
        "'a %s c' % b",
         'a %s c' % b,
         'a %s c', b,
    )
    check_eval(
        "'a %s c' % check_eval, a @ b, a + []",
         'a %s c', check_eval, a, b, [],
    )


def check_interesting(source):
    frame = inspect.currentframe().f_back
    evaluator = Evaluator.from_frame(frame)
    root = ast.parse(source)
    node = root.body[0].value
    cannot = value = None
    try:
        value = evaluator[node]
    except CannotEval as e:
        cannot = e

    expr = ast.Expression(body=node)
    ast.copy_location(expr, node)
    code = compile(expr, "<expr>", "eval")
    try:
        expected = eval(code, frame.f_globals, frame.f_locals)
    except Exception:
        if cannot:
            return None
        else:
            raise
    else:
        if cannot:
            raise cannot
        else:
            assert value == expected

    return is_expression_interesting(node, value)


def test_is_expression_interesting():
    class Foo:
        def method(self):
            pass

        alias = method

    foo = Foo()
    x = [check_interesting]
    foo.x = x
    assert check_interesting('x')
    assert not check_interesting('help')
    assert not check_interesting('check_interesting')
    assert not check_interesting('[1]')
    assert check_interesting('[1, 2][0]')
    assert check_interesting('foo')
    assert not check_interesting('Foo')
    assert check_interesting('foo.x')
    assert not check_interesting('foo.method')
    assert check_interesting('foo.alias')
    assert not check_interesting('Foo.method')
    assert check_interesting('Foo.alias')
    assert check_interesting('x[0]')
    assert not check_interesting('typing.List')
    assert check_interesting('[typing.List][0]')


def test_boolop():
    for a, b, c in [
        [0, 123, 456],
        [0, [0], [[0]]],
        [set(), {1}, {1, (1,)}],
    ]:
        str((a, b, c))
        for length in [2, 3, 4]:
            for vals in itertools.product(["1/0", "a", "b", "c"], repeat=length):
                for op in [
                    "not in",
                    "is not",
                    *"+ - / // * & ^ % @ | >> or and < <= > >= == != in is".split(),
                ]:
                    op = " %s " % op
                    source = op.join(vals)
                    check_interesting(source)


def test_is():
    for a, b, c in [
        [check_interesting, CannotEval(), CannotEval],
    ]:
        str((a, b, c))
        for length in [2, 3, 4]:
            for vals in itertools.product(["1/0", "a", "b", "c"], repeat=length):
                for op in ["is", "is not"]:
                    op = " %s " % op
                    source = op.join(vals)
                    check_interesting(source)


def test_calls():
    # No keywords allowed
    with pytest.raises(CannotEval):
        check_interesting("str(b'', encoding='utf8')")

    # This function not allowed
    with pytest.raises(CannotEval):
        check_interesting("print(3)")

    assert check_interesting("slice(3)")
    assert check_interesting("slice(3, 5)")
    assert check_interesting("slice(3, 5, 1)")
    assert check_interesting("int()")
    assert check_interesting("int('5')")
    assert check_interesting("int('55', 12)")
    assert check_interesting("range(3)")
    assert check_interesting("range(3, 5)")
    assert check_interesting("range(3, 5, 1)")
    assert check_interesting("round(3.14159)")
    assert check_interesting("round(3.14159, 2)")
    assert check_interesting("complex()")
    assert check_interesting("complex(5, 2)")
    assert check_interesting("list()")
    assert check_interesting("tuple()")
    assert check_interesting("dict()")
    assert check_interesting("bytes()")
    assert check_interesting("frozenset()")
    assert check_interesting("bytearray()")
    assert check_interesting("abs(3)")
    assert check_interesting("hex(3)")
    assert check_interesting("bin(3)")
    assert check_interesting("oct(3)")
    assert check_interesting("bool(3)")
    assert check_interesting("chr(3)")
    assert check_interesting("ord('3')")
    assert check_interesting("len([CannotEval, len])")
    assert check_interesting("list([CannotEval, len])")
    assert check_interesting("tuple([CannotEval, len])")
    assert check_interesting("str(b'123', 'utf8')")
    assert check_interesting("bytes('123', 'utf8')")
    assert check_interesting("bytearray('123', 'utf8')")
    assert check_interesting("divmod(123, 4)")
    assert check_interesting("pow(123, 4)")
    assert check_interesting("id(id)")
    assert check_interesting("type(id)")
    assert check_interesting("all([1, 2])")
    assert check_interesting("any([1, 2])")
    assert check_interesting("sum([1, 2])")
    assert check_interesting("sum([len])") is None
    assert check_interesting("sorted([[1, 2], [3, 4]])")
    assert check_interesting("min([[1, 2], [3, 4]])")
    assert check_interesting("max([[1, 2], [3, 4]])")
    assert check_interesting("hash(((1, 2), (3, 4)))")
    assert check_interesting("set(((1, 2), (3, 4)))")
    assert check_interesting("dict(((1, 2), (3, 4)))")
    assert check_interesting("frozenset(((1, 2), (3, 4)))")
    assert check_interesting("ascii(((1, 2), (3, 4)))")
    assert check_interesting("str(((1, 2), (3, 4)))")
    assert check_interesting("repr(((1, 2), (3, 4)))")


def test_unsupported():
    with pytest.raises(CannotEval):
        check_interesting("[x for x in []]")

    with pytest.raises(CannotEval):
        check_interesting("{**{}}")

    with pytest.raises(CannotEval):
        check_interesting("[*[]]")

    with pytest.raises(CannotEval):
        check_interesting("int(*[1])")


def test_group_expressions():
    x = (1, 2)
    evaluator = Evaluator({'x': x})
    tree = ast.parse('x[0] + x[x[0]]').body[0].value
    expressions = evaluator.find_expressions(tree)
    grouped = set(
        (frozenset(nodes), value)
        for nodes, value in
        group_expressions(expressions)
    )
    expected = {
        (frozenset([tree.left, subscript_item(tree.right)]),
         x[0]),
        (frozenset([tree.left.value, subscript_item(tree.right).value, tree.right.value]),
         x),
        (frozenset([subscript_item(tree.left), subscript_item(subscript_item(tree.right))]),
         0),
        (frozenset([tree.right]),
         x[x[0]]),
        (frozenset([tree]),
         x[0] + x[x[0]]),
    }
    assert grouped == expected

    grouped = set(
        (frozenset(nodes), value)
        for nodes, value in
        evaluator.interesting_expressions_grouped(tree)
    )
    expected = set(
        (nodes, value)
        for nodes, value in expected
        if value != 0
    )
    assert grouped == expected


def subscript_item(node):
    if sys.version_info < (3, 9):
        return node.slice.value
    else:
        return node.slice


def test_evaluator_wrong_getitem():
    evaluator = Evaluator({})
    with pytest.raises(TypeError, match="node should be an ast.expr, not 'str'"):
        # noinspection PyTypeChecker
        str(evaluator["foo"])


@pytest.mark.parametrize("expr", ["lst[:,:]", "lst[9]"])
def test_cannot_subscript(expr):
    with pytest.raises(Exception):
        eval(expr)

    evaluator = Evaluator({'lst': [1]})
    tree = ast.parse(expr)
    node = tree.body[0].value
    assert isinstance(node, ast.Subscript)
    with pytest.raises(CannotEval):
        str(evaluator[node])


def parse_expr(source):
    return ast.parse(source, mode="eval").body


def test_evaluation_limit_inherits_cannot_eval():
    assert issubclass(EvaluationLimit, CannotEval)
    assert isinstance(EvaluationLimit(), CannotEval)
    assert str(EvaluationLimit()) == "EvaluationLimit"


def test_evaluate_limited_counts_every_node():
    # a + b: BinOp root plus two Name nodes = 3 nodes
    evaluator = Evaluator({'a': 10, 'b': 20})
    tree = parse_expr('a + b')
    assert evaluator.evaluate_limited(tree, max_nodes=3) == 30

    # A successful limited call caches the root in the shared cache,
    # so a subsequent call needs no budget at all.
    assert evaluator.evaluate_limited(tree, max_nodes=0) == 30
    assert evaluator[tree] == 30

    evaluator = Evaluator({'a': 10, 'b': 20})
    with pytest.raises(EvaluationLimit):
        evaluator.evaluate_limited(tree, max_nodes=2)

    # Nothing was cached by the failed limited call.
    assert evaluator._cache == {}


def test_evaluate_limited_root_is_billed():
    evaluator = Evaluator({'a': 1})
    tree = parse_expr('a')
    with pytest.raises(EvaluationLimit):
        evaluator.evaluate_limited(tree, max_nodes=0)
    assert evaluator._cache == {}
    assert evaluator.evaluate_limited(tree, max_nodes=1) == 1


@pytest.mark.parametrize("source, count", [
    ("123", 1),
    ("-1", 2),  # UnaryOp root plus the Constant operand
    ("[1, 2, 3]", 4),
    ("(1, 2)", 3),
    ("{1, 2}", 3),
    ("{'a': 1}", 3),
    ("[[1, 2]]", 4),
    ("[1, [2, 3]]", 5),
])
def test_evaluate_limited_literal_containers(source, count):
    tree = parse_expr(source)

    evaluator = Evaluator({})
    assert evaluator.evaluate_limited(tree, max_nodes=count) == ast.literal_eval(tree)
    # Only the root is cached, the literal shortcut caches no children.
    assert set(evaluator._cache) == {tree}

    # One unit short: the literal shortcut cannot bypass the budget.
    evaluator = Evaluator({})
    with pytest.raises(EvaluationLimit):
        evaluator.evaluate_limited(tree, max_nodes=count - 1)
    assert evaluator._cache == {}


def test_evaluate_limited_nested_evaluation_shares_budget():
    # a + b + c: outer BinOp, inner BinOp and three Name nodes = 5
    evaluator = Evaluator({'a': 1, 'b': 2, 'c': 3})
    tree = parse_expr('a + b + c')
    assert evaluator.evaluate_limited(tree, max_nodes=5) == 6
    evaluator = Evaluator({'a': 1, 'b': 2, 'c': 3})
    with pytest.raises(EvaluationLimit):
        evaluator.evaluate_limited(tree, max_nodes=4)
    assert evaluator._cache == {}


def test_evaluate_limited_subscript_shares_cache():
    lst = [10, 20, 30]
    evaluator = Evaluator({'lst': lst})
    tree = parse_expr('lst[1]')
    # Subscript root, Name 'lst' and the index Constant = 3 expr nodes
    assert evaluator.evaluate_limited(tree, max_nodes=3) == 20
    # Root and children now live in the ordinary shared cache.
    assert evaluator[tree] == 20

    evaluator = Evaluator({'lst': lst})
    with pytest.raises(EvaluationLimit):
        evaluator.evaluate_limited(tree, max_nodes=2)
    assert evaluator._cache == {}
    # The ordinary unlimited entry works as a retry after rollback.
    assert evaluator[tree] == 20


def test_evaluate_limited_short_circuited_branches_free():
    # The right operand of a short-circuiting bool op is never visited.
    evaluator = Evaluator({})
    assert evaluator.evaluate_limited(parse_expr('0 and nope'), max_nodes=2) == 0

    evaluator = Evaluator({})
    assert evaluator.evaluate_limited(parse_expr('1 or nope'), max_nodes=2) == 1

    evaluator = Evaluator({})
    assert evaluator.evaluate_limited(
        parse_expr('0 and 2 and nope'), max_nodes=2
    ) == 0

    evaluator = Evaluator({})
    assert evaluator.evaluate_limited(
        parse_expr('1 or nope or nope'), max_nodes=2
    ) == 1

    # When the branch IS evaluated it costs budget (then CannotEval).
    evaluator = Evaluator({})
    with pytest.raises(EvaluationLimit):
        evaluator.evaluate_limited(parse_expr('1 and nope'), max_nodes=2)
    evaluator = Evaluator({})
    with pytest.raises(CannotEval):
        evaluator.evaluate_limited(parse_expr('1 and nope'), max_nodes=3)


def test_evaluate_limited_same_ast_object_billed_once():
    # One Name object used as both operands.
    name_x = ast.Name('x', ast.Load())
    tree = ast.BinOp(name_x, ast.Add(), name_x)
    evaluator = Evaluator({'x': 7})
    assert evaluator.evaluate_limited(tree, max_nodes=2) == 14

    evaluator = Evaluator({'x': 7})
    with pytest.raises(EvaluationLimit):
        evaluator.evaluate_limited(tree, max_nodes=1)

    # One Constant object repeated inside a literal container.
    const = ast.Constant(5)
    tree = ast.List([const, const], ast.Load())
    evaluator = Evaluator({})
    assert evaluator.evaluate_limited(tree, max_nodes=2) == [5, 5]

    evaluator = Evaluator({})
    with pytest.raises(EvaluationLimit):
        evaluator.evaluate_limited(tree, max_nodes=1)


def test_evaluate_limited_cache_hits_are_free():
    evaluator = Evaluator({'a': 10, 'b': 20})
    tree = parse_expr('a + b')
    # Pre-cache one operand: only the root and the other operand remain.
    evaluator[tree.left]
    assert evaluator.evaluate_limited(tree, max_nodes=2) == 30

    evaluator = Evaluator({'a': 10, 'b': 20})
    evaluator[tree.left]
    with pytest.raises(EvaluationLimit):
        evaluator.evaluate_limited(tree, max_nodes=1)

    # A node already cached inside a literal subtree is a cache hit too.
    const = ast.Constant(9)
    evaluator = Evaluator({})
    assert evaluator[const] == 9
    tree = ast.List([const, const], ast.Load())
    assert evaluator.evaluate_limited(tree, max_nodes=1) == [9, 9]


def test_evaluate_limited_rollback_preserves_prior_cache():
    # A distinct int object: usable in the arithmetic below while still
    # letting us assert object identity through the cache.
    sentinel = 10 ** 9
    evaluator = Evaluator({'a': sentinel, 'b': 2, 'c': 3})
    tree = parse_expr('a + b + c')
    # Pre-cache the Name 'a' before the limited call.
    name_a = tree.left.left
    assert evaluator[name_a] is sentinel

    with pytest.raises(EvaluationLimit):
        # Outer BinOp, inner BinOp and Name b are new; Name c would be 5th.
        evaluator.evaluate_limited(tree, max_nodes=3)

    # The prior cache survives untouched, including object identity.
    assert set(evaluator._cache) == {name_a}
    assert evaluator._cache[name_a] is sentinel

    # Retrying with a larger budget works.
    assert evaluator.evaluate_limited(tree, max_nodes=4) == sentinel + 5


def test_evaluate_limited_independent_calls_recount():
    evaluator = Evaluator({'a': 10, 'b': 20})
    tree = parse_expr('a + b')
    for _ in range(3):
        with pytest.raises(EvaluationLimit):
            evaluator.evaluate_limited(tree, max_nodes=2)
        assert evaluator._cache == {}
    assert evaluator.evaluate_limited(tree, max_nodes=3) == 30
    # After a success the root is cached, so a zero-budget call is a hit.
    assert evaluator.evaluate_limited(tree, max_nodes=0) == 30


def test_evaluate_limited_cannot_eval_caching_unchanged():
    evaluator = Evaluator({})
    tree = parse_expr('missing')
    with pytest.raises(CannotEval):
        evaluator.evaluate_limited(tree, max_nodes=10)
    # A normal CannotEval is cached as a failure as before.
    assert evaluator._cache[tree] is CannotEval
    # The cached failure is a cache hit and needs no budget.
    with pytest.raises(CannotEval):
        evaluator.evaluate_limited(tree, max_nodes=0)

    # Partial successes before a plain CannotEval stay cached.
    evaluator = Evaluator({'a': 1})
    tree = parse_expr('[a, missing]')
    with pytest.raises(CannotEval):
        evaluator.evaluate_limited(tree, max_nodes=10)
    assert evaluator._cache[tree.elts[0]] == 1
    assert evaluator._cache[tree] is CannotEval


@pytest.mark.parametrize("bad_max_nodes", [
    -1, True, False, 1.0, "1", None, (), [], object()
])
def test_evaluate_limited_invalid_max_nodes(bad_max_nodes):
    evaluator = Evaluator({'a': 1})
    tree = parse_expr('a')
    with pytest.raises(ValueError):
        evaluator.evaluate_limited(tree, max_nodes=bad_max_nodes)
    # The rejection happens before any evaluation: cache untouched.
    assert evaluator._cache == {}
    # A valid call afterwards works normally.
    assert evaluator.evaluate_limited(tree, max_nodes=1) == 1


def test_evaluate_limited_wrong_node_type():
    evaluator = Evaluator({})
    with pytest.raises(TypeError, match="node should be an ast.expr"):
        # noinspection PyTypeChecker
        evaluator.evaluate_limited("a", max_nodes=5)
    assert evaluator._cache == {}
