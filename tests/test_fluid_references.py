import gaspype as gp


def test_fluid_references():
    fs = gp.fluid_system('Cl, CH4, H2O, C2H6, C3H8')

    reference_string = """Cl          : Hf:Cox,1989. Moore,1971. Moore,1970a. Gordon,1999. [g 7/97]
CH4         : Gurvich,1991 pt1 p44 pt2 p36. [g 8/99]
H2O         : Hf:Cox,1989. Woolley,1987. TRC(10/88) tuv25. [g 8/89]
C2H6        : Ethane. Pamidimukkala,1982. [g 7/00]
C3H8        : Hf:TRC(10/85) w1350. Chao,1973. [g 2/00]"""

    assert reference_string == fs.get_species_references(), 'fs.get_species_references() == ' + fs.get_species_references()
