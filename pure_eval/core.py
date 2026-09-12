import ast
import builtins
import operator
from collections import ChainMap, OrderedDict, deque
from contextlib import suppress
from types import FrameType
from typing import Any, Tuple, Iterable, List, Mapping, Dict, Union, Set

from pure_eval.my_getattr_static import getattr_static
from pure_eval.utils import (
    CannotEval,
    has_ast_name,
    copy_ast_without_context,
    is_standard_types,
    of_standard_types,
    is_any,
    of_type,
    ensure_dict,
    MISSING_NAME,
    UNSUPPORTED_SYNTAX,
    UNSAFE_OPERATION,
    OPERATION_ERROR,
)


class _Failure(Exception):
    """
    Internal signal that an expression could not be evaluated.
    It carries the reason (one of the four reason constants), the AST node
    which actually blocked evaluation and the path of AST field names and
    list indices from the currently handled node down to that node.
    """

    def __init__(self, reason: str, node: ast.AST, *path: Any):
        self.reason = reason
        self.node = node
        self.path = tuple(path)
        super().__init__(reason)


class _CachedFailure:
    """
    A failed evaluation stored in the cache: the reason, the blocking node
    and the intrinsic path from the cache key node to that blocking node.
    """

    __slots__ = ("reason", "node", "path")

    def __init__(self, reason: str, node: ast.AST, path: Tuple[Any, ...]):
        self.reason = reason
        self.node = node
        self.path = path


_unknown = object()


class Evaluator:
    def __init__(self, names: Mapping[str, Any]):
        """
        Construct a new evaluator with the given variable names.
        This is a low level API, typically you will use `Evaluator.from_frame(frame)`.

        :param names: a mapping from variable names to their values.
        """

        self.names = names
        self._cache = {}  # type: Dict[ast.expr, Any]

    @classmethod
    def from_frame(cls, frame: FrameType) -> 'Evaluator':
        """
        Construct an Evaluator that can look up variables from the given frame.

        :param frame: a frame object, e.g. from a traceback or `inspect.currentframe().f_back`.
        """

        return cls(ChainMap(
            ensure_dict(frame.f_locals),
            ensure_dict(frame.f_globals),
            ensure_dict(frame.f_builtins),
        ))

    def __getitem__(self, node: ast.expr) -> Any:
        """
        Find the value of the given node.
        If it cannot be evaluated safely, this raises `CannotEval`.
        The result is cached either way.

        :param node: an AST expression to evaluate
        :return: the value of the node
        """

        if not isinstance(node, ast.expr):
            raise TypeError("node should be an ast.expr, not {!r}".format(type(node).__name__))

        try:
            return self._eval(node)
        except _Failure:
            raise CannotEval

    def explain(self, node: ast.expr) -> Dict[str, Any]:
        """
        Explain whether the given node can be evaluated, like `__getitem__`,
        but returning a diagnostic dictionary instead of raising `CannotEval`:

            {"ok": bool, "value": Any, "reason": str, "node": ast.expr, "path": tuple}

        On success `ok` is True, `value` is the value of the node and
        `reason`, `node` and `path` are None.

        On failure `ok` is False, `value` is None, `node` is the AST node
        which actually blocked evaluation and `path` is a tuple of AST field
        names and list indices from this node (the request root) to that
        blocking node, e.g. ('values', 1) for the second value of a boolean
        operation, or () when the root node itself blocks evaluation.

        `reason` is one of:
            - missing_name: an ast.Name was not found in the names mapping.
            - unsupported_syntax: an AST node type or syntax form which this
              evaluator does not handle, e.g. comprehensions or calls with
              keyword or star arguments.
            - unsafe_operation: supported syntax but a value, call target or
              attribute access would cross the standard-types/static-access
              safety boundary.
            - operation_error: the operands passed the safety checks but a
              built-in operation raised (e.g. division by zero, out of range
              index, missing key), or a static attribute does not exist.

        The result is cached either way. Hitting a cached failure never
        re-evaluates the expression; the path is still rebuilt from this
        request root following the original evaluation order.

        :param node: an AST expression to evaluate
        :return: a dictionary describing the evaluation result
        """

        if not isinstance(node, ast.expr):
            raise TypeError("node should be an ast.expr, not {!r}".format(type(node).__name__))

        try:
            value = self._eval(node)
        except _Failure as failure:
            return {
                "ok": False,
                "value": None,
                "reason": failure.reason,
                "node": failure.node,
                "path": failure.path,
            }
        return {
            "ok": True,
            "value": value,
            "reason": None,
            "node": None,
            "path": None,
        }

    def _eval(self, node: ast.expr) -> Any:
        """
        Cached evaluation, raising `_Failure` with the reason, the blocking
        node and the path relative to `node` when evaluation is impossible.
        """

        cached = self._cache.get(node, _unknown)
        if cached is not _unknown:
            if isinstance(cached, _CachedFailure):
                raise _Failure(cached.reason, cached.node, *cached.path)
            return cached

        try:
            result = self._handle(node)
        except _Failure as failure:
            self._cache[node] = _CachedFailure(failure.reason, failure.node, failure.path)
            raise
        self._cache[node] = result
        return result

    def _child(self, node: ast.expr, *path: Any) -> Any:
        """
        Evaluate a child node, prepending the given field/index path segments
        to a failure so that paths are rebuilt per request without reusing
        paths cached for a different parent expression.
        """

        try:
            return self._eval(node)
        except _Failure as failure:
            if path:
                failure.path = (*path, *failure.path)
            raise

    def _handle(self, node: ast.expr) -> Any:
        """
        This is where the evaluation happens.
        Users should use `__getitem__`, i.e. `evaluator[node]`,
        or `explain`, as it provides caching.

        :param node: an AST expression to evaluate
        :return: the value of the node
        """

        with suppress(Exception):
            return ast.literal_eval(node)

        if isinstance(node, ast.Name):
            try:
                return self.names[node.id]
            except KeyError:
                raise _Failure(MISSING_NAME, node)
        elif isinstance(node, ast.Attribute):
            value = self._child(node.value, "value")
            try:
                return getattr_static(value, node.attr)
            except CannotEval as e:
                raise _Failure(e.reason or UNSAFE_OPERATION, node)
        elif isinstance(node, ast.Subscript):
            return self._handle_subscript(node)
        elif isinstance(node, (ast.List, ast.Tuple, ast.Set, ast.Dict)):
            return self._handle_container(node)
        elif isinstance(node, ast.UnaryOp):
            return self._handle_unary(node)
        elif isinstance(node, ast.BinOp):
            return self._handle_binop(node)
        elif isinstance(node, ast.BoolOp):
            return self._handle_boolop(node)
        elif isinstance(node, ast.Compare):
            return self._handle_compare(node)
        elif isinstance(node, ast.Call):
            return self._handle_call(node)
        raise _Failure(UNSUPPORTED_SYNTAX, node)

    def _handle_call(self, node):
        if node.keywords:
            raise _Failure(UNSUPPORTED_SYNTAX, node)
        func = self._child(node.func, "func")
        args = [
            self._child(arg, "args", i)
            for i, arg in enumerate(node.args)
        ]

        def check_arg(arg, i):
            try:
                return of_standard_types(arg, check_dict_values=False, deep=False)
            except CannotEval:
                raise _Failure(UNSAFE_OPERATION, node.args[i], "args", i)

        if (
            is_any(
                func,
                slice,
                int,
                range,
                round,
                complex,
                list,
                tuple,
                abs,
                hex,
                bin,
                oct,
                bool,
                ord,
                float,
                len,
                chr,
            )
            or len(args) == 0
            and is_any(func, set, dict, str, frozenset, bytes, bytearray, object)
            or len(args) >= 2
            and is_any(func, str, divmod, bytes, bytearray, pow)
        ):
            args = [check_arg(arg, i) for i, arg in enumerate(args)]
            try:
                return func(*args)
            except Exception:
                raise _Failure(OPERATION_ERROR, node)

        if len(args) == 1:
            arg = args[0]
            arg_node = node.args[0]
            if is_any(func, id, type):
                try:
                    return func(arg)
                except Exception:
                    raise _Failure(OPERATION_ERROR, node)
            if is_any(func, all, any, sum):
                try:
                    of_type(arg, tuple, frozenset, list, set, dict, OrderedDict, deque)
                    for x in arg:
                        of_standard_types(x, check_dict_values=False, deep=False)
                except CannotEval:
                    raise _Failure(UNSAFE_OPERATION, arg_node, "args", 0)
                try:
                    return func(arg)
                except Exception:
                    raise _Failure(OPERATION_ERROR, node)

            if is_any(
                func, sorted, min, max, hash, set, dict, ascii, str, repr, frozenset
            ):
                try:
                    of_standard_types(arg, check_dict_values=True, deep=True)
                except CannotEval:
                    raise _Failure(UNSAFE_OPERATION, arg_node, "args", 0)
                try:
                    return func(arg)
                except Exception:
                    raise _Failure(OPERATION_ERROR, node)
        raise _Failure(UNSAFE_OPERATION, node.func, "func")

    def _handle_compare(self, node):
        left = self._child(node.left, "left")
        left_node = node.left
        left_path = ("left",)
        result = True

        for i, (op, right_node) in enumerate(zip(node.ops, node.comparators)):
            right = self._child(right_node, "comparators", i)

            op_type = type(op)
            op_func = {
                ast.Eq: operator.eq,
                ast.NotEq: operator.ne,
                ast.Lt: operator.lt,
                ast.LtE: operator.le,
                ast.Gt: operator.gt,
                ast.GtE: operator.ge,
                ast.Is: operator.is_,
                ast.IsNot: operator.is_not,
                ast.In: (lambda a, b: a in b),
                ast.NotIn: (lambda a, b: a not in b),
            }[op_type]

            if op_type not in (ast.Is, ast.IsNot):
                try:
                    of_standard_types(left, check_dict_values=False, deep=True)
                except CannotEval:
                    raise _Failure(UNSAFE_OPERATION, left_node, *left_path)
                try:
                    of_standard_types(right, check_dict_values=False, deep=True)
                except CannotEval:
                    raise _Failure(UNSAFE_OPERATION, right_node, "comparators", i)

            try:
                result = op_func(left, right)
            except Exception:
                raise _Failure(OPERATION_ERROR, node)
            if not result:
                return result
            left = right
            left_node = right_node
            left_path = ("comparators", i)

        return result

    def _handle_boolop(self, node):
        try:
            left = of_standard_types(
                self._child(node.values[0], "values", 0),
                check_dict_values=False,
                deep=False,
            )
        except CannotEval:
            raise _Failure(UNSAFE_OPERATION, node.values[0], "values", 0)

        for i, right_node in enumerate(node.values[1:], start=1):
            # We need short circuiting so that the whole operation can be evaluated
            # even if the right operand can't
            def evaluate_right():
                try:
                    return of_standard_types(
                        self._child(right_node, "values", i),
                        check_dict_values=False,
                        deep=False,
                    )
                except CannotEval:
                    raise _Failure(UNSAFE_OPERATION, right_node, "values", i)

            if isinstance(node.op, ast.Or):
                left = left or evaluate_right()
            else:
                assert isinstance(node.op, ast.And)
                left = left and evaluate_right()
        return left

    def _handle_binop(self, node):
        op_type = type(node.op)
        op = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.FloorDiv: operator.floordiv,
            ast.Mod: operator.mod,
            ast.Pow: operator.pow,
            ast.LShift: operator.lshift,
            ast.RShift: operator.rshift,
            ast.BitOr: operator.or_,
            ast.BitXor: operator.xor,
            ast.BitAnd: operator.and_,
        }.get(op_type)
        if not op:
            raise _Failure(UNSUPPORTED_SYNTAX, node)
        left_value = self._child(node.left, "left")
        hash_type = is_any(type(left_value), set, frozenset, dict, OrderedDict)
        try:
            left = of_standard_types(left_value, check_dict_values=False, deep=hash_type)
        except CannotEval:
            raise _Failure(UNSAFE_OPERATION, node.left, "left")
        formatting = type(left_value) in (str, bytes) and op_type == ast.Mod

        try:
            right = of_standard_types(
                self._child(node.right, "right"),
                check_dict_values=formatting,
                deep=formatting or hash_type,
            )
        except CannotEval:
            raise _Failure(UNSAFE_OPERATION, node.right, "right")
        try:
            return op(left, right)
        except Exception:
            raise _Failure(OPERATION_ERROR, node)

    def _handle_unary(self, node: ast.UnaryOp):
        try:
            value = of_standard_types(
                self._child(node.operand, "operand"),
                check_dict_values=False,
                deep=False,
            )
        except CannotEval:
            raise _Failure(UNSAFE_OPERATION, node.operand, "operand")
        op_type = type(node.op)
        op = {
            ast.USub: operator.neg,
            ast.UAdd: operator.pos,
            ast.Not: operator.not_,
            ast.Invert: operator.invert,
        }.get(op_type)
        if op is None:
            raise _Failure(UNSUPPORTED_SYNTAX, node)
        try:
            return op(value)
        except Exception:
            raise _Failure(OPERATION_ERROR, node)

    def _handle_subscript(self, node):
        value = self._child(node.value, "value")
        try:
            of_standard_types(
                value, check_dict_values=False, deep=is_any(type(value), dict, OrderedDict)
            )
        except CannotEval:
            raise _Failure(UNSAFE_OPERATION, node.value, "value")

        slice_node = node.slice
        if isinstance(slice_node, ast.Slice):
            index_node = slice_node
            index_path = ("slice",)
            parts = []
            for part_name in ("lower", "upper", "step"):
                part = getattr(slice_node, part_name)
                if part is None:
                    parts.append(None)
                else:
                    parts.append(self._child(part, "slice", part_name))
            index = slice(*parts)
        elif isinstance(slice_node, ast.ExtSlice):
            raise _Failure(UNSUPPORTED_SYNTAX, node)
        else:
            if isinstance(slice_node, ast.Index):
                index_node = slice_node.value
                index_path = ("slice", "value")
            else:
                index_node = slice_node
                index_path = ("slice",)
            index = self._child(index_node, *index_path)
        try:
            of_standard_types(index, check_dict_values=False, deep=True)
        except CannotEval:
            raise _Failure(UNSAFE_OPERATION, index_node, *index_path)

        try:
            return value[index]
        except Exception:
            raise _Failure(OPERATION_ERROR, node)

    def _handle_container(
            self,
            node: Union[ast.List, ast.Tuple, ast.Set, ast.Dict]
    ) -> Union[List, Tuple, Set, Dict]:
        """Handle container nodes, including List, Set, Tuple and Dict"""
        if isinstance(node, ast.Dict):
            if None in node.keys:  # ** unpacking inside {}, not yet supported
                raise _Failure(UNSUPPORTED_SYNTAX, node)
            elts = [
                self._child(key, "keys", i)
                for i, key in enumerate(node.keys)
            ]
        else:
            elts = [
                self._child(elt, "elts", i)
                for i, elt in enumerate(node.elts)
            ]
        if isinstance(node, ast.List):
            return elts
        if isinstance(node, ast.Tuple):
            return tuple(elts)

        # Set and Dict
        for i, elt in enumerate(elts):
            if not is_standard_types(elt, check_dict_values=False, deep=True):
                if isinstance(node, ast.Dict):
                    raise _Failure(UNSAFE_OPERATION, node.keys[i], "keys", i)
                else:
                    raise _Failure(UNSAFE_OPERATION, node.elts[i], "elts", i)

        if isinstance(node, ast.Set):
            try:
                return set(elts)
            except TypeError:
                raise _Failure(OPERATION_ERROR, node)

        assert isinstance(node, ast.Dict)

        pairs = [
            (elt, self._child(val, "values", i))
            for i, (elt, val) in enumerate(zip(elts, node.values))
        ]
        try:
            return dict(pairs)
        except TypeError:
            raise _Failure(OPERATION_ERROR, node)

    def find_expressions(self, root: ast.AST) -> Iterable[Tuple[ast.expr, Any]]:
        """
        Find all expressions in the given tree that can be safely evaluated.
        This is a low level API, typically you will use `interesting_expressions_grouped`.

        :param root: any AST node
        :return: generator of pairs (tuples) of expression nodes and their corresponding values.
        """

        for node in ast.walk(root):
            if not isinstance(node, ast.expr):
                continue

            try:
                value = self[node]
            except CannotEval:
                continue

            yield node, value

    def interesting_expressions_grouped(self, root: ast.AST) -> List[Tuple[List[ast.expr], Any]]:
        """
        Find all interesting expressions in the given tree that can be safely evaluated,
        grouping equivalent nodes together.

        For more control and details, see:
         - Evaluator.find_expressions
         - is_expression_interesting
         - group_expressions

        :param root: any AST node
        :return: A list of pairs (tuples) containing:
                    - A list of equivalent AST expressions
                    - The value of the first expression node
                       (which should be the same for all nodes, unless threads are involved)
        """

        return group_expressions(
            pair
            for pair in self.find_expressions(root)
            if is_expression_interesting(*pair)
        )


def is_expression_interesting(node: ast.expr, value: Any) -> bool:
    """
    Determines if an expression is potentially interesting, at least in my opinion.
    Returns False for the following expressions whose value is generally obvious:
        - Literals (e.g. 123, 'abc', [1, 2, 3], {'a': (), 'b': ([1, 2], [3])})
        - Variables or attributes whose name is equal to the value's __name__.
            For example, a function `def foo(): ...` is not interesting when referred to
            as `foo` as it usually would, but `bar` can be interesting if `bar is foo`.
            Similarly the method `self.foo` is not interesting.
        - Builtins (e.g. `len`) referred to by their usual name.

    This is a low level API, typically you will use `interesting_expressions_grouped`.

    :param node: an AST expression
    :param value: the value of the node
    :return: a boolean: True if the expression is interesting, False otherwise
    """

    with suppress(ValueError):
        ast.literal_eval(node)
        return False

    # TODO exclude inner modules, e.g. numpy.random.__name__ == 'numpy.random' != 'random'
    # TODO exclude common module abbreviations, e.g. numpy as np, pandas as pd
    if has_ast_name(value, node):
        return False

    if (
            isinstance(node, ast.Name)
            and getattr(builtins, node.id, object()) is value
    ):
        return False

    return True


def group_expressions(expressions: Iterable[Tuple[ast.expr, Any]]) -> List[Tuple[List[ast.expr], Any]]:
    """
    Organise expression nodes and their values such that equivalent nodes are together.
    Two nodes are considered equivalent if they have the same structure,
    ignoring context (Load, Store, or Delete) and location (lineno, col_offset).
    For example, this will group together the same variable name mentioned multiple times in an expression.

    This will not check the values of the nodes. Equivalent nodes should have the same values,
    unless threads are involved.

    This is a low level API, typically you will use `interesting_expressions_grouped`.

    :param expressions: pairs of AST expressions and their values, as obtained from
                          `Evaluator.find_expressions`, or `(node, evaluator[node])`.
    :return: A list of pairs (tuples) containing:
                - A list of equivalent AST expressions
                - The value of the first expression node
                   (which should be the same for all nodes, unless threads are involved)
    """

    result = {}
    for node, value in expressions:
        dump = ast.dump(copy_ast_without_context(node))
        result.setdefault(dump, ([], value))[0].append(node)
    return list(result.values())
