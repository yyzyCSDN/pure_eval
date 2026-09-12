import ast
import inspect
import sys
import typing

import itertools
import pytest

from pure_eval import Evaluator, CannotEval
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


def parsed(source):
    return ast.parse(source).body[0].value


def test_invalidate_names_replacement_propagates_to_parents():
    x = 1
    y = [2]
    names = {'x': x, 'y': y}
    evaluator = Evaluator(names)
    node = parsed('(x, y)')

    result = evaluator[node]
    assert result == (1, y)
    cached_x = evaluator[node.elts[0]]
    cached_y = evaluator[node.elts[1]]
    assert cached_x == 1
    assert cached_y is y

    names['x'] = 10
    evaluator.invalidate_names(['x'])

    assert evaluator[node] == (10, y)
    # The unaffected y node is served from the old cache with object identity.
    assert evaluator[node.elts[1]] is y
    # The rebuilt container still contains that very same object.
    assert evaluator[node][1] is y
    # The x node was invalidated and re-evaluated.
    assert evaluator[node.elts[0]] == 10
    assert evaluator[node.elts[0]] is not cached_x


def test_invalidate_names_unaffected_cache_keeps_identity():
    sentinel = object()
    names = {'x': 1, 'obj': sentinel}
    evaluator = Evaluator(names)
    node = parsed('obj')
    literal = parsed('123')

    assert evaluator[node] is sentinel
    assert evaluator[literal] == 123

    evaluator.invalidate_names(['x'])
    assert evaluator[node] is sentinel
    assert evaluator[literal] == 123
    assert node in evaluator._cache


def test_invalidate_names_add_binding_retries_cannot_eval():
    names = {}
    evaluator = Evaluator(names)
    node = parsed('x + 1')

    with pytest.raises(CannotEval):
        evaluator[node]
    with pytest.raises(CannotEval):
        evaluator[node.left]
    assert node in evaluator._cache
    assert node.left in evaluator._cache

    # Adding the previously missing name invalidates both the Name node and
    # the parent expression whose cached result was CannotEval.
    names['x'] = 41
    evaluator.invalidate_names(['x'])
    assert node not in evaluator._cache
    assert node.left not in evaluator._cache

    assert evaluator[node.left] == 41
    assert evaluator[node] == 42


def test_invalidate_names_delete_binding():
    names = {'x': 5}
    evaluator = Evaluator(names)
    node = parsed('x')
    assert evaluator[node] == 5

    del names['x']
    evaluator.invalidate_names(['x'])
    with pytest.raises(CannotEval):
        evaluator[node]

    # Adding it back lets the cached failure be evaluated again.
    names['x'] = 6
    evaluator.invalidate_names(['x'])
    assert evaluator[node] == 6


def test_invalidate_names_subscript():
    names = {'d': {1: 2}, 'k': 1}
    evaluator = Evaluator(names)
    node = parsed('d[k]')

    assert evaluator[node] == 2

    names['d'] = {1: 3}
    evaluator.invalidate_names(['d'])
    assert evaluator[node] == 3

    # Changing k to a missing key makes evaluation fail.
    names['k'] = 5
    evaluator.invalidate_names(['k'])
    with pytest.raises(CannotEval):
        evaluator[node]

    # And a valid k again makes it succeed.
    names['k'] = 1
    evaluator.invalidate_names(['k'])
    assert evaluator[node] == 3


def test_invalidate_names_attribute_binding():
    class Box:
        pass

    box = Box()
    box.attr = 1
    other = Box()
    other.attr = 5

    names = {'box': box}
    evaluator = Evaluator(names)
    node = parsed('box.attr')

    assert evaluator[node] == 1

    # Internal mutation of a bound object is deliberately NOT tracked.
    box.attr = 2
    assert evaluator[node] == 1

    # Replacing the binding and explicitly invalidating does re-evaluate.
    names['box'] = other
    evaluator.invalidate_names(['box'])
    assert evaluator[node] == 5


def test_invalidate_names_boolop_short_circuit():
    names = {'flag': 1, 'a': 100}
    evaluator = Evaluator(names)
    node = parsed('flag or a')

    assert evaluator[node] == 1
    # The unread right operand was never evaluated.
    assert node.values[1] not in evaluator._cache

    # Invalidating an unread name leaves the cached result untouched.
    names['a'] = 101
    evaluator.invalidate_names(['a'])
    assert evaluator[node] == 1
    assert node in evaluator._cache

    # Taking the other branch records fresh dependencies.
    names['flag'] = 0
    evaluator.invalidate_names(['flag'])
    assert evaluator[node] == 101

    # The new branch association must have replaced the old one: invalidating
    # 'a' now invalidates the whole Or expression.
    names['a'] = 111
    evaluator.invalidate_names(['a'])
    assert evaluator[node] == 111


def test_invalidate_names_boolop_and_short_circuit():
    names = {'flag': 0, 'a': 100}
    evaluator = Evaluator(names)
    node = parsed('flag and a')

    assert evaluator[node] == 0
    assert node.values[1] not in evaluator._cache

    names['a'] = 101
    evaluator.invalidate_names(['a'])
    assert evaluator[node] == 0

    names['flag'] = 1
    evaluator.invalidate_names(['flag'])
    assert evaluator[node] == 101

    names['a'] = 111
    evaluator.invalidate_names(['a'])
    assert evaluator[node] == 111


def test_invalidate_names_chained_compare_short_circuit():
    names = {'a': 1, 'b': 0, 'c': 999}
    evaluator = Evaluator(names)
    node = parsed('a < b < c')

    result = evaluator[node]
    assert result is False
    # The comparator after the failing comparison was never read.
    assert node.comparators[1] not in evaluator._cache

    # Invalidating the unread name 'c' keeps the exact cached result object.
    names['c'] = -5
    evaluator.invalidate_names(['c'])
    assert evaluator[node] is result

    # Making the first comparison pass reads c on the next evaluation:
    # 1 < 2 is True, then 2 < -5 is False.
    names['b'] = 2
    evaluator.invalidate_names(['b'])
    assert evaluator[node] is False

    # c is now a genuine dependency (it was not before), so invalidating it
    # re-evaluates the whole chained comparison.
    names['c'] = 3
    evaluator.invalidate_names(['c'])
    assert evaluator[node] is True


def test_invalidate_names_with_find_expressions():
    names = {'x': 1, 'y': 2}
    evaluator = Evaluator(names)
    root = parsed('x + y')

    assert dict(evaluator.find_expressions(root))  # populated the cache
    names['x'] = 10
    evaluator.invalidate_names(['x'])

    values = {node: value for node, value in evaluator.find_expressions(root)}
    assert values[root] == 12
    assert values[root.left] == 10
    assert values[root.right] == 2


def test_invalidate_names_duplicates_and_iterables():
    names = {'x': 1, 'y': 2}
    evaluator = Evaluator(names)
    node = parsed('x')
    assert evaluator[node] == 1

    # Duplicate names are harmless and only invalidate once.
    evaluator.invalidate_names(['x', 'x', 'x'])
    assert node not in evaluator._cache
    assert evaluator[node] == 1

    # Any finite iterable (not just lists) works, including empty ones.
    evaluator.invalidate_names(('x',))
    evaluator.invalidate_names({'x'})
    evaluator.invalidate_names(iter(['x']))
    evaluator.invalidate_names(iter([]))
    evaluator.invalidate_names(_ for _ in ())
    assert evaluator[node] == 1


def test_invalidate_names_rejects_str_and_bytes():
    evaluator = Evaluator({})
    with pytest.raises(TypeError):
        evaluator.invalidate_names('x')
    with pytest.raises(TypeError):
        evaluator.invalidate_names(b'x')


def test_invalidate_names_rejects_non_str_elements():
    evaluator = Evaluator({})
    for bad_names in [
        [1],
        [b'x'],
        [None],
        ['x', 1],
        ['x', None],
    ]:
        with pytest.raises(TypeError):
            evaluator.invalidate_names(bad_names)


def test_invalidate_names_non_iterable_propagates_typeerror():
    evaluator = Evaluator({})
    for bad in [None, 42, 3.14, True]:
        with pytest.raises(TypeError):
            evaluator.invalidate_names(bad)


def test_invalidate_names_iteration_error_propagates_unchanged():
    class Boom(Exception):
        pass

    def exploding():
        yield 'x'
        raise Boom()

    names = {'x': 1}
    evaluator = Evaluator(names)
    node = parsed('x')
    assert evaluator[node] == 1

    with pytest.raises(Boom):
        evaluator.invalidate_names(exploding())

    # The error during iteration must not have changed the cache.
    assert node in evaluator._cache
    assert evaluator[node] == 1


def test_invalidate_names_validation_failure_is_atomic():
    names = {'x': 1, 'y': 2}
    evaluator = Evaluator(names)
    node_x = parsed('x')
    node_y = parsed('y')
    assert evaluator[node_x] == 1
    assert evaluator[node_y] == 2

    # The valid 'x' comes before the invalid element, but validation failure
    # must not invalidate anything.
    with pytest.raises(TypeError):
        evaluator.invalidate_names(['x', 3])

    assert node_x in evaluator._cache
    assert node_y in evaluator._cache
    assert evaluator[node_x] == 1
    assert evaluator[node_y] == 2
