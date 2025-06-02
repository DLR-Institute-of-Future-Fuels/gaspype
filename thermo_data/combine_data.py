import yaml
import xml.etree.ElementTree as ET
import glob
import sys


def main():
    output_file = sys.argv[1]
    input_files = sys.argv[2:]

    inp_therm_prop: list[dict] = []

    for glop_filter in input_files:
        for file_name in glob.glob(glop_filter):
            print(f'Processing file: {file_name}...')
            if file_name.endswith('.xml'):
                inp_therm_prop += get_xml_data(file_name)
            else:
                with open(file_name, 'r') as f:
                    inp_therm_prop += yaml.safe_load(f)['species']

    therm_prop = []
    added_species_names: set[str] = set()

    for species in inp_therm_prop:
        name = species['name']
        assert isinstance(name, str)
        for element in species['composition']:
            name = name.replace(element.upper(), element)
        if name not in added_species_names and 'E' not in species['composition']:
            species['name'] = name
            therm_prop.append(species)
            added_species_names.add(name)

    with open(output_file, 'w') as f:
        f.write('species:\n')
        for species in sorted(therm_prop, key=lambda s: s['name']):
            f.write('\n- ' + yaml.dump({'name': species['name']}, default_flow_style=False))
            f.write(f'  composition: {yaml.dump(species["composition"],default_flow_style=True)}')
            f.write('  thermo:\n')
            f.write(f'    model: {species["thermo"]["model"]}\n')
            f.write(f'    temperature-ranges: {yaml.dump(species["thermo"]["temperature-ranges"],default_flow_style=True)}')
            f.write('    data:\n')
            for d in species['thermo']['data']:
                f.write(f'    - {d}\n')
            f.write('    ' + yaml.dump({'note': species['thermo']['note']}, default_flow_style=False))

    print('Added: ', ', '.join(sorted(added_species_names)))


def get_xml_data(file_name: str) -> list[dict]:
    xml_data_list: list[dict] = []
    tree = ET.parse(file_name)
    root = tree.getroot()
    for i, child in enumerate(root):

        elements = {el[0]: int(el[1]) for c in child.find('elements').findall('element') for el in c.items()}

        values_temp = [tr for tr in child.findall('T_range')]
        t_ranges = [float(v.attrib['Tlow']) for v in values_temp] + [float(values_temp[-1].attrib['Thigh'])]
        data = [[float(v.text) for v in tr] for tr in values_temp]

        is_gas = child.find('condensed').text == 'False'

        if is_gas:
            xml_data_list.append({
                'name': child.attrib['inp_file_name'],
                'composition': elements,
                'thermo': {
                    'model': 'NASA9',
                    'temperature-ranges': t_ranges,
                    'data': data,
                    'note': child.find('comment').text
                }
            })
    return xml_data_list


if __name__ == "__main__":
    main()
