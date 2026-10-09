import yaml
import struct
import sys
import os


def main():
    args = [a for a in sys.argv[1:] if a != '--double']
    # With --double the polynomial coefficients are stored as 64 bit floats
    # instead of 32 bit floats
    float_format = 'd' if '--double' in sys.argv[1:] else 'f'
    input_file = args[0]
    output_file = args[1]

    assert not output_file.endswith('.yml') and not output_file.endswith('.yaml'), 'Binary output file should not have yaml-extension'

    with open(input_file) as f:
        data = yaml.safe_load(f)

    data = list({v['name']: v for v in data['species']}.values())

    species_names = ' '.join(s['name'] for s in data)

    header_len = 8

    offset = 8 + len(species_names) + len(data) * header_len

    header_list = []
    body_list = []
    added_list = set()

    for dat in data:
        if dat['name'] not in added_list:

            composition_count = len(dat['composition'])
            model = {'NASA9': 9}[dat['thermo']['model']]
            temperatures = dat['thermo']['temperature-ranges']
            ref_string = dat['thermo']['note'].encode('utf-8')
            header = struct.pack('<I4B', offset, composition_count, model, len(temperatures), len(ref_string))

            assert len(header) == header_len

            header_list.append(header)

            composition = [el for k, v in dat['composition'].items() for el in [k.ljust(2).encode('ASCII'), v]]
            data_vals = [d for darr in dat['thermo']['data'] for d in darr[:model]]

            assert len(dat['thermo']['data']) == len(temperatures) - 1, f"Temperature data length mismatch for {dat['name']}."

            if any(len(d) != model for d in dat['thermo']['data']):
                print(f"Warning: Data length mismatch for {dat['name']}. Expected {model} coefficients, got {len(dat['thermo']['data'][0])}.")

            format_string = '<' + '2sB' * composition_count + f'{len(temperatures)}f{len(data_vals)}{float_format}{len(ref_string)}s'

            body = struct.pack(format_string, *(composition + temperatures + data_vals + [ref_string]))
            body_list.append(body)

            offset += len(body)
            added_list.add(dat['name'])

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'wb') as f:
        f.write(b'gapd' if float_format == 'd' else b'gapy')
        f.write(struct.pack('<I', len(species_names)))
        f.write(species_names.encode('ASCII'))
        for dat in header_list:
            f.write(dat)
        for dat in body_list:
            f.write(dat)


if __name__ == '__main__':
    main()
