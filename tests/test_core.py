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


def parse_expr(source):
    return ast.parse(source, mode="eval").body


def test_explain_success():
    x = 1
    evaluator = Evaluator.from_frame(inspect.currentframe())
    node = parse_expr("x + 1")
    assert evaluator.explain(node) == {
        "ok": True,
        "value": 2,
        "reason": None,
        "node": None,
        "path": None,
    }
    # The original entry point is unchanged
    assert evaluator[node] == 2
    # Falsy/None values are proper cached successes, not missing entries
    for source, value in [("0", 0), ("False", False), ("None", None), ("''", "")]:
        assert evaluator.explain(parse_expr(source))["value"] == value


@pytest.mark.parametrize("source, reason", [
    ("missing", "missing_name"),
    ("[i for i in ()]", "unsupported_syntax"),
    ("lambda: 0", "unsupported_syntax"),
    ("int(x=1)", "unsupported_syntax"),
    ("int(*[1])", "unsupported_syntax"),
    ("{**{}}", "unsupported_syntax"),
    ("1 / 0", "operation_error"),
    ("[][0]", "operation_error"),
    ("{}[1]", "operation_error"),
    ("int('x')", "operation_error"),
])
def test_explain_reasons(source, reason):
    evaluator = Evaluator.from_frame(inspect.currentframe())
    node = parse_expr(source)
    result = evaluator.explain(node)
    assert result["ok"] is False
    assert result["value"] is None
    assert result["reason"] == reason
    assert isinstance(result["node"], ast.AST)
    assert isinstance(result["path"], tuple)
    assert set(result) == {"ok", "value", "reason", "node", "path"}
    # The original entry point still raises CannotEval for the same node
    with pytest.raises(CannotEval):
        evaluator[node]


def test_explain_unsafe_operation():
    class Foo:
        @property
        def prop(self):
            raise AssertionError("user property must not be invoked")

    foo = Foo()
    evaluator = Evaluator.from_frame(inspect.currentframe())

    node = parse_expr("foo.prop")
    result = evaluator.explain(node)
    assert result["reason"] == "unsafe_operation"
    assert result["node"] is node
    assert result["path"] == ()
    # Explaining again uses the cache and still does not call the property
    assert evaluator.explain(node)["reason"] == "unsafe_operation"

    node = parse_expr("print(1)")
    result = evaluator.explain(node)
    assert result["reason"] == "unsafe_operation"
    assert result["node"] is node.func
    assert result["path"] == ("func",)


def test_explain_missing_attribute_is_operation_error():
    class Foo:
        pass

    foo = Foo()
    evaluator = Evaluator.from_frame(inspect.currentframe())
    node = parse_expr("foo.absent")
    result = evaluator.explain(node)
    assert result["reason"] == "operation_error"
    assert result["node"] is node
    assert result["path"] == ()


def test_explain_paths():
    evaluator = Evaluator({"lst": [1], "print": print})

    node = parse_expr("missing")
    assert evaluator.explain(node)["path"] == ()

    node = parse_expr("False or missing")
    result = evaluator.explain(node)
    assert result["path"] == ("values", 1)
    assert result["node"] is node.values[1]

    node = parse_expr("1 + missing")
    result = evaluator.explain(node)
    assert result["path"] == ("right",)
    assert result["node"] is node.right

    node = parse_expr("missing + 1")
    assert evaluator.explain(node)["path"] == ("left",)

    node = parse_expr("-missing")
    assert evaluator.explain(node)["path"] == ("operand",)

    node = parse_expr("1 < missing")
    result = evaluator.explain(node)
    assert result["path"] == ("comparators", 0)
    assert result["node"] is node.comparators[0]

    node = parse_expr("missing(1)")
    result = evaluator.explain(node)
    assert result["path"] == ("func",)
    assert result["node"] is node.func

    node = parse_expr("print(missing)")
    result = evaluator.explain(node)
    assert result["path"] == ("args", 0)
    assert result["node"] is node.args[0]

    node = parse_expr("(missing, 1)")
    result = evaluator.explain(node)
    assert result["path"] == ("elts", 0)
    assert result["node"] is node.elts[0]

    node = parse_expr("lst[:missing]")
    result = evaluator.explain(node)
    assert result["path"] == ("slice", "upper")
    assert result["node"] is node.slice.upper

    node = parse_expr("(missing, 1)[0]")
    result = evaluator.explain(node)
    assert result["path"] == ("value", "elts", 0)
    assert result["node"] is node.value.elts[0]


def test_explain_short_circuit():
    # No names at all, yet the unexecuted branches are never inspected
    evaluator = Evaluator({})

    result = evaluator.explain(parse_expr("1 or missing"))
    assert result["ok"] and result["value"] == 1

    result = evaluator.explain(parse_expr("0 and missing"))
    assert result["ok"] and result["value"] == 0

    node = parse_expr("1 == 2 == missing")
    result = evaluator.explain(node)
    assert result["ok"] and result["value"] is False

    node = parse_expr("1 == 1 == missing")
    result = evaluator.explain(node)
    assert result["reason"] == "missing_name"
    assert result["path"] == ("comparators", 1)


def test_explain_failure_cached_by_getitem():
    evaluator = Evaluator({})
    node = parse_expr("missing")
    with pytest.raises(CannotEval):
        evaluator[node]
    result = evaluator.explain(node)
    assert result["reason"] == "missing_name"
    assert result["node"] is node
    assert result["path"] == ()


def test_explain_cached_failure_is_not_reevaluated(monkeypatch):
    import pure_eval.core as core

    calls = []

    def counting_truediv(a, b):
        calls.append((a, b))
        return a / b

    monkeypatch.setattr(core.operator, "truediv", counting_truediv)
    evaluator = Evaluator({})
    node = parse_expr("1 / 0")
    first = evaluator.explain(node)
    second = evaluator.explain(node)
    assert first["reason"] == second["reason"] == "operation_error"
    assert first["node"] is second["node"] is node
    assert len(calls) == 1
    with pytest.raises(CannotEval):
        evaluator[node]
    assert len(calls) == 1


def test_explain_path_rebuilt_per_request_root():
    evaluator = Evaluator({})
    missing = ast.Name(id="missing", ctx=ast.Load())
    bool_root = ast.BoolOp(ast.Or(), [ast.Constant(0), missing])
    bin_root = ast.BinOp(ast.Constant(1), ast.Add(), missing)

    assert evaluator.explain(bool_root)["path"] == ("values", 1)
    assert evaluator.explain(bin_root)["path"] == ("right",)
    assert evaluator.explain(missing)["path"] == ()

    # A child failing first (via __getitem__) still yields the path from a
    # different request root without re-evaluating the child
    evaluator = Evaluator({"d": {}})
    inner = parse_expr("d['x']")
    with pytest.raises(CannotEval):
        evaluator[inner]
    outer = ast.BinOp(inner, ast.Add(), ast.Constant(1))
    result = evaluator.explain(outer)
    assert result["reason"] == "operation_error"
    assert result["node"] is inner
    assert result["path"] == ("left",)


def test_explain_wrong_type():
    evaluator = Evaluator({})
    with pytest.raises(TypeError, match="ast.expr"):
        # noinspection PyTypeChecker
        evaluator.explain("foo")
