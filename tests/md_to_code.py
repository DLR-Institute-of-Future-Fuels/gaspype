import re
from typing import Generator, Iterable
from dataclasses import dataclass
import sys


@dataclass
class markdown_segment:
    code_block: bool
    language: str
    text: str


def convert_to(target_format: str, md_filename: str, out_filename: str, language: str = 'python'):
    with open(md_filename, "r") as f_in, open(out_filename, "w") as f_out:
        segments = segment_markdown(f_in)

        if target_format == 'test':
            f_out.write('\n'.join(segments_to_test(segments, language)))
        elif target_format == 'script':
            f_out.write('\n'.join(segments_to_script(segments, language)))
        elif target_format == 'striped_markdown':
            f_out.write('\n'.join(segments_to_striped_markdown(segments, language)))
        else:
            raise ValueError('Unknown target format')


def segment_markdown(markdown_file: Iterable[str]) -> Generator[markdown_segment, None, None]:
    regex = re.compile(r"(?:^```\s*(?P<language>(?:\w|-)*)$)", re.MULTILINE)

    block_language: str = ''
    code_block = False
    line_buffer: list[str] = []

    for line in markdown_file:
        match = regex.match(line)
        if match:
            if line_buffer:
                yield markdown_segment(code_block, block_language, ''.join(line_buffer))
                line_buffer.clear()
            block_language = match.group('language')
            code_block = not code_block
        else:
            line_buffer.append(line)

    if line_buffer:
        yield markdown_segment(code_block, block_language, '\n'.join(line_buffer))


def segments_to_script(segments: Iterable[markdown_segment], test_language: str = "python") -> Generator[str, None, None]:
    for segment in segments:
        if segment.code_block:
            if segment.language == test_language:
                yield segment.text

            else:
                for line in segment.text.splitlines():
                    yield '#   | ' + line
            yield ''

        else:
            for line in segment.text.strip(' \n').splitlines():
                yield '#  ' + line
            yield ''


def segments_to_striped_markdown(segments: Iterable[markdown_segment], test_language: str = "python") -> Generator[str, None, None]:
    for segment in segments:
        if segment.code_block:
            if segment.language == test_language:
                yield "``` " + test_language
                yield segment.text
                yield "```"

            elif segment.language:
                for line in segment.text.splitlines():
                    yield '#   | ' + line
            yield ''

        else:
            for line in segment.text.strip(' \n').splitlines():
                yield '#  ' + line
            yield ''


def segments_to_test(segments: Iterable[markdown_segment], script_language: str = "python") -> Generator[str, None, None]:

    ret_block_flag = False

    yield 'def run_test():'

    for segment in segments:
        if segment.code_block:
            if segment.language == script_language:
                lines = [line for line in segment.text.splitlines() if line.strip()]
                ret_block_flag = lines[-1] if not re.match(r'^[^(]*=', lines[-1]) and not lines[-1].startswith('import ') else None
                # print('Last line: ', ret_block_flag, '-----------', lines[-1])

                yield ''
                yield '    print("---------------------------------------------------------")'
                yield ''
                if ret_block_flag:
                    yield from ['    ' + str(line) for line in segment.text.splitlines()[:-1]]
                    yield f'    print("-- Result (({ret_block_flag})):")'
                    yield f'    print(({ret_block_flag}).__repr__().strip())'
                else:
                    yield from ['    ' + str(line) for line in segment.text.splitlines()]

            elif ret_block_flag:
                yield '    ref_str = r"""'
                yield from [str(line) for line in segment.text.splitlines()]
                yield '"""'
                yield f'    print("-- Reference (({ret_block_flag})):")'
                yield '    print(ref_str.strip())'
                yield f'    assert ({ret_block_flag}).__repr__().strip() == ref_str.strip()'
                ret_block_flag = False

    yield '\nif __name__ == "__main__":'
    yield '    run_test()'


if __name__ == "__main__":
    format = sys.argv[1]
    assert format in ['test', 'script']
    convert_to(sys.argv[1], sys.argv[2], sys.argv[3])
