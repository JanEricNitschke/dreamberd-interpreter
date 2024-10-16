import re
import sys
from argparse import ArgumentParser
from pathlib import Path
from time import sleep

from dreamberd.base import InterpretationError, NonFormattedError, Token, TokenType
from dreamberd.builtin import KEYWORDS, DreamberdValue, Name, Variable
from dreamberd.interpreter import (
    AsyncStatements,
    WhenStatementWatchers,
    interpret_code_statements,
    interpret_code_statements_main_wrapper,
    load_global_dreamberd_variables,
    load_globals,
    load_public_global_variables,
)
from dreamberd.processor.lexer import tokenize
from dreamberd.processor.syntax_tree import generate_syntax_tree

__all__ = ["run_repl", "run_file"]

__REPL_FILENAME = "__repl__"
sys.setrecursionlimit(100000)


def __get_next_repl_input(closed_scope_layers: int = 0) -> tuple[str, list[Token]]:
    print("   " * closed_scope_layers, "\033[33m>\033[39m ", end="", sep="")
    code = input()
    tokens = tokenize(__REPL_FILENAME, code)
    start_closed_scope_layers = closed_scope_layers
    for t in tokens:
        if t.type == TokenType.L_CURLY:
            closed_scope_layers += 1
        elif t.type == TokenType.R_CURLY:
            closed_scope_layers -= 1
    if closed_scope_layers < 0:
        msg = "Too many closed braces with not enough open braces."
        raise InterpretationError(msg)
    kickback_layers, i = 0, 0
    trimmed_code = code.strip()
    while i < len(trimmed_code) and trimmed_code[i] == "}":
        kickback_layers += 1
        i += 1
    if kickback_layers:
        print(
            "\033[1A\r",
            "   " * (start_closed_scope_layers - kickback_layers),
            "\033[33m>\033[39m ",
            code,
            "   " * (start_closed_scope_layers),
            sep="",
            flush=True,
        )
    if closed_scope_layers:
        new_code, new_tokens = __get_next_repl_input(closed_scope_layers)
        code += new_code
        tokens += new_tokens
    match non_whitespace_tokens := [
        t for t in tokens if t.type != TokenType.WHITESPACE
    ]:
        case [Token(type=TokenType.NAME)]:
            tokens = [
                Token(TokenType.NAME, "print", -1, -1),
                Token(TokenType.WHITESPACE, " ", -1, -1),
                non_whitespace_tokens[0],
                Token(TokenType.BANG, "!", -1, -1),
            ]
        case _:
            # Not sure why this is a match tbh.
            pass
    return code, tokens


def run_repl() -> None:
    namespaces: list[dict[str, Variable | Name]] = [KEYWORDS.copy()]  # pyright: ignore[reportAssignmentType]
    load_globals(__REPL_FILENAME, "", {}, set(), [], {})
    load_global_dreamberd_variables(namespaces)
    load_public_global_variables(namespaces)
    async_statements: AsyncStatements = []
    when_statement_watchers: WhenStatementWatchers = [{}]
    while True:
        try:
            code, tokens = __get_next_repl_input()
            statements = generate_syntax_tree(__REPL_FILENAME, tokens, code)
            interpret_code_statements(
                statements, namespaces, async_statements, when_statement_watchers
            )
        except EOFError:
            print()
            sys.exit()
        except KeyboardInterrupt:
            print()
        except InterpretationError as e:
            print(e)
        except NonFormattedError as e:
            print("\033[31m\n", e, "\n\033[39m", sep="")


def run_file(main_filename: str) -> None:  # idk what else to call this
    with Path(main_filename).open(encoding="utf-8") as f:
        code_lines = f.readlines()

    # split up into separate 'files' by finding which lines start with multiple equal signs
    files: list[tuple[str | None, str]] = []
    if any(matches := [re.match(r"=====.*", line) for line in code_lines]):
        for i, match in reversed([*enumerate(matches)]):
            if match is None:
                continue
            name = match.group().strip("=").strip() or None
            files.insert(0, (name, "".join(code_lines[i + 1 :])))
            del code_lines[i:]
        files.insert(0, (None, "".join(code_lines[0:])))
    else:
        files = [(None, "".join(code_lines))]

    # execute code for each file
    importable_names: dict[str, dict[str, DreamberdValue]] = {}
    for filename, code in files:
        filename = filename or "__unnamed_file__"  # noqa: PLW2901
        tokens = tokenize(filename, code)
        statements = generate_syntax_tree(filename, tokens, code)

        # load variables and run the code
        namespaces: list[dict[str, Variable | Name]] = [KEYWORDS.copy()]  # pyright: ignore[reportAssignmentType]
        exported_names: list[tuple[str, str, DreamberdValue]] = []
        load_globals(
            filename,
            code,
            {},
            set(),
            exported_names,
            importable_names.get(filename, {}),
        )
        load_global_dreamberd_variables(namespaces)
        load_public_global_variables(namespaces)
        interpret_code_statements_main_wrapper(statements, namespaces, [], [{}])

        # take exported names and put them where they belong
        for target_filename, name, value in exported_names:
            if target_filename not in importable_names:
                importable_names[target_filename] = {}
            importable_names[target_filename][name] = value

    print(
        "\033[33mCode has finished executing. Press ^C once or twice to stop waiting for when-statements and after-statements.\033[039m",
        flush=True,
    )
    try:
        while True:
            sleep(1)  # just waiting for any clicks, when statements, etc
    except KeyboardInterrupt:
        sys.exit()  # quit silently


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "file",
        help="the file containing your DreamBerd code",
        nargs="?",
        default="",
        type=str,
    )
    parser.add_argument(
        "-s",
        "--show-traceback",
        help="show the full Python trackback upon errors",
        action="store_true",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.show_traceback:
        sys.tracebacklimit = 0
    if not args.file:
        run_repl()
    else:
        run_file(args.file)


if __name__ == "__main__":
    main()
