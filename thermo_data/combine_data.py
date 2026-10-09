import yaml
import xml.etree.ElementTree as ET
import glob
import sys


def main():
    args = [a for a in sys.argv[1:] if a != '--condensed']
    # With --condensed the condensed phase species (solids and liquids) of
    # the XML files are selected instead of the gas phase species
    condensed = '--condensed' in sys.argv[1:]
    output_file = args[0]
    input_files = args[1:]

    inp_therm_prop: list[dict] = []

    for glop_filter in input_files:
        for file_name in glob.glob(glop_filter):
            print(f'Processing file: {file_name}...')
            if file_name.endswith('.xml'):
                inp_therm_prop += get_xml_data(file_name, condensed)
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
            f.write(f'  composition: {yaml.dump(species["composition"], default_flow_style=True)}')
            f.write('  thermo:\n')
            f.write(f'    model: {species["thermo"]["model"]}\n')
            f.write(f'    temperature-ranges: {yaml.dump(species["thermo"]["temperature-ranges"], default_flow_style=True)}')
            f.write('    data:\n')
            for d in species['thermo']['data']:
                f.write(f'    - {d}\n')
            f.write('    ' + yaml.dump({'note': species['thermo']['note']}, default_flow_style=False, width=1000))

    print('Added: ', ', '.join(sorted(added_species_names)))


def get_xml_data(file_name: str, condensed: bool = False) -> list[dict]:
    xml_data_list: list[dict] = []
    xml_data_index: dict[str, dict] = {}
    tree = ET.parse(file_name)
    root = tree.getroot()
    for i, child in enumerate(root):

        elements = {el[0]: int(el[1]) for c in child.find('elements').findall('element') for el in c.items()}

        values_temp = [tr for tr in child.findall('T_range')]
        t_ranges = [float(v.attrib['Tlow']) for v in values_temp] + [float(values_temp[-1].attrib['Thigh'])]
        data = [[float(v.text) for v in tr] for tr in values_temp]

        is_gas = child.find('condensed').text == 'False'
        name = child.attrib['inp_file_name']
        note = child.find('comment').text

        if is_gas == condensed:
            continue

        if not all(elements.values()):
            # Non-stoichiometric species like Fe.947O can not be represented
            # with integer element counts
            print(f'Skipping {name}: element counts {elements}')
            continue

        if name in xml_data_index and xml_data_index[name]['temperature-ranges'][-1] == t_ranges[0]:
            # Some condensed species are split in multiple entries at a
            # lambda transition, these are merged to a single species
            thermo = xml_data_index[name]
            thermo['temperature-ranges'] += t_ranges[1:]
            thermo['data'] += data
            thermo['note'] += ' / ' + note
        else:
            xml_data_list.append({
                'name': name,
                'composition': elements,
                'thermo': {
                    'model': 'NASA9',
                    'temperature-ranges': t_ranges,
                    'data': data,
                    'note': note
                }
            })
            xml_data_index.setdefault(name, xml_data_list[-1]['thermo'])
    return xml_data_list


if __name__ == "__main__":
    main()
