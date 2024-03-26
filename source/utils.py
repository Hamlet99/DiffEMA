import gemmi
import os


def preprocess_pdb(
    structure_path,
    map_path,
    sampling_radius=7,
    output_directory=None,
):
    """
    Function to preprocess a PDB and its corresponding map file by extracting amino acids and map patches.

    :param structure_path: path to the PDB file
    :type structure_path: str
    :param map_path: path to the map file
    :type map_path: str
    :param sampling_radius: radius of the map patch to be extracted around each CA atom in angstroms
    :type sampling_radius: int
    :param output_directory: path to the output directory
    :type output_directory: str or None
    """
    # Checking the file formats
    if structure_path.split(".")[-1] not in [
        "pdb",
        "cif",
    ]:
        raise ValueError("PDB file must be in .pdb or .cif format")

    structure_gemmi = gemmi.read_structure(structure_path)

    if map_path.split(".")[-1] == "mtz":
        mtz_map = gemmi.read_mtz_file(map_path)

    else:
        raise ValueError("Map file must be in .mtz format")

    # Creating the output directory
    if output_directory is not None:
        output_path = os.path.join(
            output_directory,
            f"{structure_path.split('/')[-1].split('.')[0]}_preprocessed",
        )
    else:
        output_path = f"{structure_path.split('/')[-1].split('.')[0]}_preprocessed"

    os.mkdir(output_path)
    extracted_amino_dir = os.path.join(output_path, "extracted_amino")
    extracted_map_dir = os.path.join(output_path, "extracted_map_patches")
    os.mkdir(extracted_amino_dir)
    os.mkdir(extracted_map_dir)

    # Extracting amino acids and the map patches
    for model_id, model in enumerate(structure_gemmi):
        for cra in model.all():  # cra = chain, residue, atom
            if cra.atom.name == "CA" and (
                cra.atom.altloc == "A" or not cra.atom.has_altloc()
            ):
                cra_str = (
                    str(model_id)
                    + "_"
                    + cra.__str__().replace("/", "_").replace(" ", "_")
                )
                cra_list = cra_str.split("_")

                selection_criteria_ca = (
                    f"/{int(cra_list[0]) + 1}/{cra_list[1]}/{cra_list[3]}/CA[C]:,A"
                )
                selection_criteria_residue = (
                    f"/{int(cra_list[0]) + 1}/{cra_list[1]}/{cra_list[3]}/:,A"
                )

                sel_ca = gemmi.Selection(selection_criteria_ca)
                sel_residue = gemmi.Selection(selection_criteria_residue)

                ca_atom = sel_ca.copy_structure_selection(structure_gemmi)
                residue = sel_residue.copy_structure_selection(structure_gemmi)

                ccp_map = gemmi.Ccp4Map()
                ccp_map.grid = mtz_map.transform_f_phi_to_map(
                    "2FOFCWT", "PH2FOFCWT", sample_rate=3
                )
                ccp_map.update_ccp4_header()

                ccp_map.set_extent(
                    ca_atom.calculate_fractional_box(margin=sampling_radius)
                )
                ccp_map.write_ccp4_map(
                    os.path.join(extracted_map_dir, cra_str + ".ccp4")
                )

                residue.write_pdb(
                    os.path.join(extracted_amino_dir, cra_str + ".pdb"),
                    gemmi.PdbWriteOptions(minimal=True, numbered_ter=False),
                )

    print(
        f"Structure {structure_path.split('/')[-1].split('.')[0]}: Preprocessing done!"
    )
