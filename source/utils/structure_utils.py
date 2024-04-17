import os
import gemmi
import pandas as pd

# TODO: Add the arguments into the parser
# TODO: Add the unit tests


class StructurePreprocessing:

    def __init__(self, pdb_structure_path, ed_map_path):
        """
         Class to preprocess a PDB and its corresponding maps.

         :param pdb_structure_path: path to the PDB file
         :type pdb_structure_path: str
         :param ed_map_path: path to the map file
         :type ed_map_path: str
        """

        self.pdb_structure_path = pdb_structure_path
        self.ed_map_path = ed_map_path
        self.patch_sampling_radius = 7  # args.patch_sampling_radius
        self.output_directory = 'training_data_preprocessed'  # args.output_directory
        self.point_cloud_sigma_level = 1.0  # args.point_cloud_sigma_level

        # Checking the file formats
        if self.pdb_structure_path.split(".")[-1] not in ["pdb", "cif"]:
            raise ValueError("PDB file must be in .pdb or .cif format")

        self.structure_gemmi = gemmi.read_structure(self.pdb_structure_path)

        if self.ed_map_path.split(".")[-1] == "mtz":
            self.protein_mtz_map = gemmi.read_mtz_file(self.ed_map_path)
        else:
            raise ValueError("Map file must be in .mtz format")

        self.protein_ccp4 = gemmi.Ccp4Map()
        self.protein_ccp4.grid = self.protein_mtz_map.transform_f_phi_to_map("2FOFCWT", "PH2FOFCWT", sample_rate=3)

    def preprocess_pdb(self):
        """
        Method to preprocess a PDB and its corresponding map file by extracting amino acids and map patches.
        """

        # Creating the output directory
        if self.output_directory is not None:
            output_path = os.path.join(self.output_directory,
                                       f"{self.pdb_structure_path.split('/')[-1].split('.')[0]}_preprocessed")
        else:
            output_path = (
                f"{self.pdb_structure_path.split('/')[-1].split('.')[0]}_preprocessed")

        os.mkdir(output_path)
        extracted_amino_dir = os.path.join(output_path, "extracted_amino")
        extracted_map_dir = os.path.join(output_path, "extracted_map_patches")
        extracted_point_clouds = os.path.join(output_path, "extracted_point_clouds")
        os.mkdir(extracted_amino_dir)
        os.mkdir(extracted_map_dir)
        os.mkdir(extracted_point_clouds)

        # Extracting amino acids, map patches and point clouds
        for model_id, model in enumerate(self.structure_gemmi):
            for cra in model.all():  # cra = chain, residue, atom
                if cra.atom.name == "CA" and (cra.atom.altloc == "A" or not cra.atom.has_altloc()):

                    cra_str = str(model_id) + "_" + cra.__str__().replace("/", "_").replace(" ", "_")
                    cra_list = cra_str.split("_")

                    # Extracting the patch map
                    selection_criteria_ca = f"/{int(cra_list[0]) + 1}/{cra_list[1]}/{cra_list[3]}/CA[C]:,A"
                    sel_ca = gemmi.Selection(selection_criteria_ca)

                    ca_atom = sel_ca.copy_structure_selection(self.structure_gemmi)
                    ca_coords = [float(j) for j in ca_atom.make_minimal_pdb().split('\n')[1].split()[-6:-3]]

                    ccp_map = gemmi.Ccp4Map()
                    ccp_map.grid = self.protein_mtz_map.transform_f_phi_to_map("2FOFCWT", "PH2FOFCWT", sample_rate=3)

                    ccp_map.update_ccp4_header()
                    ccp_map.set_extent(ca_atom.calculate_fractional_box(margin=self.patch_sampling_radius))
                    ccp_map.grid.normalize()

                    # Extracting and saving the point cloud
                    self.get_point_cloud(ccp_map, os.path.join(extracted_point_clouds, cra_str + ".csv"), ca_coords)

                    # Saving the map patch
                    ccp_map.write_ccp4_map(os.path.join(extracted_map_dir, cra_str + ".ccp4"))

                    # Saving the amino acid as a PDB file
                    selection_criteria_residue = f"/{int(cra_list[0]) + 1}/{cra_list[1]}/{cra_list[3]}/:,A"
                    sel_residue = gemmi.Selection(selection_criteria_residue)
                    residue = sel_residue.copy_structure_selection(self.structure_gemmi)
                    residue.write_pdb(os.path.join(extracted_amino_dir, cra_str + ".pdb"),
                                      gemmi.PdbWriteOptions(minimal=True, numbered_ter=False))

        print(f"Structure {self.pdb_structure_path.split('/')[-1].split('.')[0]}: Preprocessing done!")

    def get_point_cloud(self, ed_map, output_path, ca_pos):
        """
        Function to extract the electron density point cloud with specified sigma level to a CSV file.

        :param ed_map: electron density map
        :type ed_map: gemmi.Ccp4Map
        :param output_path: path to save the CSV file
        :type output_path: str
        :param ca_pos: CA atom coordinates [x, y, z]
        :type ca_pos: list
        """

        # Difference between the center of the patch and the CA atom
        ca_gemmi_pos = gemmi.Position(ca_pos[0], ca_pos[1], ca_pos[2])
        nearest_to_the_center = self.protein_ccp4.grid.get_nearest_point(ca_gemmi_pos)
        ca_coords_in_unit_cell = self.protein_ccp4.grid.point_to_position(nearest_to_the_center)

        dx_centers = ca_pos[0] - ca_coords_in_unit_cell.x
        dy_centers = ca_pos[1] - ca_coords_in_unit_cell.y
        dz_centers = ca_pos[2] - ca_coords_in_unit_cell.z

        # Getting the patch center
        point_a = self.protein_ccp4.grid.get_position(0, 0, 0)  # diagonal corner
        point_b = self.protein_ccp4.grid.get_position(ed_map.grid.nu, ed_map.grid.nv, ed_map.grid.nw)  # opposite corner

        dx = ca_pos[0] - (point_a.x + point_b.x)*0.5 - dx_centers
        dy = ca_pos[1] - (point_a.y + point_b.y)*0.5 - dy_centers
        dz = ca_pos[2] - (point_a.z + point_b.z)*0.5 - dz_centers

        # Aligning the patch with the CA atom and extracting the point cloud

        x, y, z, intensities = [], [], [], []
        for point in ed_map.grid:
            if point.value >= self.point_cloud_sigma_level:
                intensities.append(point.value)
                point_position = self.protein_ccp4.grid.point_to_position(point)
                x.append(point_position.x + dx)
                y.append(point_position.y + dy)
                z.append(point_position.z + dz)

        data_coords = pd.DataFrame({"x": x, "y": y, "z": z, "i": intensities})
        data_coords.to_csv(output_path, index=False)
