import gaspype as gp
import numpy as np
import pytest
import pandas as pd


def outp(arr, decr=''):
    print(decr + '    '.join([f'{f*100:8.1f} %' for f in arr]))


def check_gas_data(composition, reference_file, rel_error=3e-3):
    test_data = np.genfromtxt(reference_file, delimiter='\t', skip_header=1)
    fl = gp.fluid(composition)

    h0, s0 = test_data[0, 3:5]
    t0 = test_data[0, 0]

    for t, p, roh, h, s, cp in test_data:
        print(f'T = {t} K')
        assert fl.get_v(t, p * 1e6) == pytest.approx(1e-3 / roh, rel=rel_error)
        assert fl.get_h(t) - fl.get_h(t0) == pytest.approx(h - h0, rel=rel_error)
        assert fl.get_s(t, p * 1e6) - fl.get_s(t0, p * 1e6) == pytest.approx(s - s0, rel=rel_error)
        assert fl.get_cp(t) == pytest.approx(cp, rel=rel_error)


def test_density_vs_volume():
    fl = gp.fluid({sp: 1 for sp in gp.species('C#H#(|OH)')})

    t = 858
    p = 2.56e5

    assert fl.get_density(t, p) == pytest.approx(fl.get_mass() / fl.get_v(t, p))


def test_volume():
    fl = gp.fluid({'O2': 1})  # 1 mol oxygen

    t = 273.15  # K
    p = 1e5  # Pa
    v_ref = 22.710954  # l

    assert fl.get_v(t, p) == pytest.approx(v_ref / 1000)


def test_h2_data():
    # Compare results to Refprop calculation
    check_gas_data({'H2': 1}, 'tests/test_data/test_data_h2.tsv')


def test_nh3_data():
    # Compare results to Refprop calculation
    check_gas_data({'NH3': 1}, 'tests/test_data/test_data_nh3_ht.tsv', 4e-2)


def test_equilibrium():
    # Compare equilibrium calculations to Cycle-Tempo results
    df = pd.read_csv('tests/test_data/cycle_temp_matlab_ref.csv', sep=';', decimal=',').fillna(0)
    fs = gp.fluid_system(['CH4', 'C2H6', 'C3H8', 'C4H10,n-butane', 'H2O', 'H2', 'CO2', 'CO'])

    for index, row in df.iterrows():
        compositions = {'CH4': row['m CH4 /g/s'],
                        'C2H6': row['m C2H6 /g/s'],
                        'C3H8': row['m C3H8 /g/s'],
                        'C4H10,n-butane': row['m C4H10 /g/s'],
                        'H2O': row['m H2O /g/s']
                        }

        reference_values = [v for v in row[
            ['x CH4.1', 'x C2H6', 'x C3H8', 'x C4H10',
             'x H2O.1', 'x H2', 'x CO2', 'x CO']]]

        t = row['T /°C'] + 273.15
        p = row['p /bar abs'] * 1e5
        carbon = row['x C(s)']

        # Compare only results without solid carbon in the equilibrium because
        # this code does not consider solids
        if carbon == 0:
            mass_comp = np.array([compositions[s] if s in compositions else 0 for s in fs.species])
            molar_comp = mass_comp / fs.array_molar_mass / np.sum(mass_comp / fs.array_molar_mass)

            fl = gp.fluid(molar_comp, fs)

            result_values = gp.equilibrium(fl, t, p).array_fractions

            print(index, gp.get_solver(), '----')
            print(molar_comp)
            outp(result_values, 'Under test: ')
            outp(reference_values, 'Reference:  ')

            assert result_values == pytest.approx(reference_values, abs=1e-3, rel=0.01)


def test_carbon():
    # Compare if solid carbon is in equilibrium present to Cycle-Tempo results
    df = pd.read_csv('tests/test_data/cycle_temp_matlab_ref.csv', sep=';', decimal=',').fillna(0)
    fs = gp.fluid_system(['CH4', 'C2H6', 'C3H8', 'C4H10,n-butane', 'H2O', 'H2', 'CO2', 'CO'])

    for index, row in df.iterrows():
        compositions = {'CH4': row['m CH4 /g/s'],
                        'C2H6': row['m C2H6 /g/s'],
                        'C3H8': row['m C3H8 /g/s'],
                        'C4H10,n-butane': row['m C4H10 /g/s'],
                        'H2O': row['m H2O /g/s']
                        }

        t = row['T /°C'] + 273.15
        p = row['p /bar abs'] * 1e5
        carbon = row['x C(s)']

        mass_comp = np.array([compositions[s] if s in compositions else 0 for s in fs.species])
        molar_comp = mass_comp / fs.array_molar_mass / np.sum(mass_comp / fs.array_molar_mass)

        fl = gp.fluid(molar_comp, fs)

        result_values = gp.carbon_activity(fl, t, p)

        print('----')
        print(f'Under test, carbon activity:     {result_values}')
        print(f'Reference carbon amount in mol:  {carbon}')

        if carbon > 0:
            assert result_values > 0.9
        else:
            assert result_values < 1.1
