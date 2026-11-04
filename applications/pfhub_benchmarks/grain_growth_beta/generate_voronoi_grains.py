import array
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.spatial import KDTree


def bbox2_2D(img, target):
    x = np.any(img == target, axis=1)
    y = np.any(img == target, axis=0)
    xmin, xmax = np.where(x)[0][[0, -1]]
    ymin, ymax = np.where(y)[0][[0, -1]]
    return xmin, xmax, ymin, ymax


def op_assignment(grain_id_array, ex_dist):
    print("Performing order parameter assignment.")
    # Calculate adjacency matrix
    Nx, Ny = grain_id_array.shape
    gid_list = np.unique(grain_id_array)
    adjacency_matrix = np.zeros((gid_list.size, gid_list.size))

    print("Number of grains: {}".format(gid_list.size))
    for i in range(gid_list.size):
        print("Calculating adjacency for grain ID {:.0f}".format(gid_list[i]))
        # Calculate a bounding box (not strictly necessary, but makes things faster)
        xmin, xmax, ymin, ymax = bbox2_2D(grain_id_array, gid_list[i])
        x0 = max(0, xmin - ex_dist*2)
        x1 = min(Nx, xmax + ex_dist*2 + 1)
        y0 = max(0, ymin - ex_dist*2)
        y1 = min(Ny, ymax + ex_dist*2 + 1)
        B = grain_id_array[x0:x1, y0:y1]
        C = B == gid_list[i]
        dist = distance_transform_edt(1 - C)
        D = dist < ex_dist
        E = D * B
        overlap_list = np.unique(E)
        for n in overlap_list:
            if n == i:
                continue
            adjacency_matrix[i, gid_list == n] = 1
    # end for - adjacency matrix is computed

    # Graph coloring algorithm to assign order parameters
    print("Assigning grains to order parameters...")
    node_colors = np.zeros(adjacency_matrix.shape[0])
    for node in range(adjacency_matrix.shape[0]):
        neighbors = np.where(adjacency_matrix[node, :] != 0)[0]
        neighboring_colors = node_colors[neighbors]

        unused_color = 1
        while np.any(neighboring_colors == unused_color):
            unused_color = unused_color + 1
        node_colors[node] = unused_color
    
    print("Number of order parameters: {:.0f}".format(np.max(node_colors)))
    gid_mapping = {}
    for i, color in zip(gid_list, node_colors):
        gid_mapping[i] = color

    return gid_mapping, int(np.max(node_colors))


def write_binary_vtk_header(filename, coords, shape):
    Nx, Ny = shape
    nodeno = Nx * Ny
    elmno = (Nx - 1) * (Ny - 1)

    # Write VTK header and grain ID field
    with open(filename, "wb") as f:
        f.write(b"# vtk DataFile Version 4.0\n")
        f.write(b"PFHub Hillert Grains\n")
        f.write(b"BINARY\n")
        f.write(b"DATASET UNSTRUCTURED_GRID\n")
        
        # Points
        f.write("POINTS {:d} double\n".format(nodeno).encode())
        # Loop over the points (nodes)
        for j, y in enumerate(coords[1]):
            for i, x in enumerate(coords[0]):
                f.write(array.array('d', (x, y, 0.0)))
        f.write(b"\n")

        # Cells
        f.write("CELLS {:d} {:d}\n".format(elmno, elmno*5).encode())
        # Loop over the elements (quads, a.k.a. type 9)
        for j in range(coords[1].size - 1):
            for i in range(coords[0].size - 1):
                n1 = i + j*Nx
                n2 = i + 1 + j*Nx
                n3 = i + 1 + (j+1)*Nx
                n4 = i + (j+1)*Nx
                elm = (4, n1, n2, n3, n4)
                f.write(array.array('i', elm))
        f.write(b"\n")

        # Cell types (all type 9)
        f.write("CELL_TYPES {:d}\n".format(elmno).encode())
        for n in range(elmno):
            f.write(array.array('i', (9,)))
        f.write(b"\n")
        f.write("POINT_DATA {:d}\n".format(nodeno).encode())
    print("VTK binary file header written to {}".format(filename))
    

def write_ascii_vtk_header(filename, coords, shape):
    Nx, Ny = shape
    nodeno = Nx * Ny
    elmno = (Nx - 1) * (Ny - 1)

    # Write VTK header and grain ID field
    with open(filename, "w") as f:
        f.write("# vtk DataFile Version 4.0\n")
        f.write("PFHub Hillert Grains\n")
        f.write("ASCII\n")
        f.write("DATASET UNSTRUCTURED_GRID\n")

        # Points
        f.write("POINTS {:d} double\n".format(nodeno))
        # Loop over the points (nodes)
        for j, y in enumerate(coords[1]):
            for i, x in enumerate(coords[0]):
                f.write("{:.8g} {:.8g} 0.0\n".format(x, y))

        # Cells
        f.write("CELLS {:d} {:d}\n".format(elmno, elmno*5))
        # Loop over the elements
        for j in range(coords[1].size - 1):
            for i in range(coords[0].size - 1):
                n1 = i + j*Nx
                n2 = i + 1 + j*Nx
                n3 = i + 1 + (j+1)*Nx
                n4 = i + (j+1)*Nx
                elm = (n1, n2, n3, n4)
                f.write("4 {:d} {:d} {:d} {:d}\n".format(*elm))

        # Cell types (all type 12)
        f.write("CELL_TYPES {:d}\n".format(elmno))
        for n in range(elmno):
            f.write("{:d}\n".format(9))
        f.write("POINT_DATA {:d}\n".format(nodeno))
    print("VTK file header written to {}".format(filename))
    

def append_vtk_scalar_field_binary(filename, coords, nodeno, data_type, field_name, field):
    with open(filename, "ab") as f:
        f.write("SCALARS {} {} 1\n".format(field_name, data_type).encode())
        f.write(b"LOOKUP_TABLE default\n")
        for j, y in enumerate(coords[1]):
            for i, x in enumerate(coords[0]):
                if data_type == "int":
                    f.write(array.array('i', (int(field[i, j]),)))
                elif data_type == "float":
                    f.write(array.array('f', (float(field[i, j]),)))
                elif data_type == "double":
                    f.write(array.array('d', (float(field[i, j]),)))
                else:
                    raise ValueError("Unsupported data type {} for binary VTK output".format(data_type))
        f.write(b"\n")
        
        
def append_vtk_scalar_field(filename, coords, nodeno, data_type, field_name, field):
    with open(filename, "a") as f:
        f.write("SCALARS {} {} 1\n".format(field_name, data_type))
        f.write("LOOKUP_TABLE default\n")
        for j, y in enumerate(coords[1]):
            for i, x in enumerate(coords[0]):
                if data_type == "int":
                    f.write("{:d}\n".format(int(field[i, j])))
                elif data_type == "float" or data_type == "double":
                    f.write("{:f}\n".format(float(field[i, j])))
                else:
                    f.write("{}\n".format(field[i, j]))
    print("Wrote field {} to {}".format(field_name, filename))
    
        
def write_order_parameters(grain_id_array, coords, gid_mapping, output_filename):
    print("Writing order parameters to {}".format(output_filename))
    # Create order parameters one at a time, save them to a file
    op_id_list = []
    for gid in gid_mapping.keys():
        op_id_list.append(gid_mapping[gid])
    op_id_list = np.unique(op_id_list)

    # Calculate mesh size and number of nodes
    Nx, Ny = grain_id_array.shape
    nodeno = Nx * Ny

    # Write VTK header and grain ID field
    write_ascii_vtk_header(output_filename, coords, grain_id_array.shape)
        
    append_vtk_scalar_field(output_filename, coords, nodeno,
                            "int", "grain_id", grain_id_array)

    # Iterate over the order parameters
    for n, op in enumerate(op_id_list):
        field_name = "n{}".format(n)
        
        print("Computing order parameter field {}".format(field_name))
        op_array = np.zeros(grain_id_array.shape)
        for gid in gid_mapping.keys():
            if op == gid_mapping[gid]:
                op_array[grain_id_array == gid] = 1.0
        
        append_vtk_scalar_field(output_filename, coords, nodeno,
                                "double", field_name, op_array)
    # end for
    print("All order parameters written to VTK.")


def write_array_flat_binary(filename, output_array):
    with open(filename, "wb") as f:
        f.write(output_array.tobytes())
    print("Wrote array to binary file {}".format(filename))


def write_fb_order_parameters(grain_id_array, gid_mapping, output_filename):
    print("Writing flat binary order parameters.")
    s = output_filename.split(".")[:-1]
    basename = ".".join(s)
    print("File name format: {}_n*.dat".format(basename))
    
    Nx, Ny = grain_id_array.shape
    
    # Create order parameters one at a time
    op_id_list = []
    for gid in gid_mapping.keys():
        op_id_list.append(gid_mapping[gid])
    op_id_list = np.unique(op_id_list)
    
    # Iterate over the order parameters
    for n, op in enumerate(op_id_list):
        field_name = "n{}".format(n + 1)
        
        print("Computing order parameter field {}".format(field_name))
        op_array = np.zeros(grain_id_array.shape)
        for gid in gid_mapping.keys():
            if op == gid_mapping[gid]:
                op_array[grain_id_array == gid] = 1.0
        
        subfilename = "{}_{}.dat".format(basename, field_name)
        write_array_flat_binary(subfilename, op_array)
    # end for
    print("All order parameters written to flat binary file series.")
    print("Copy the following for the PRISMS-PF input file:\n")
    for n, op in enumerate(op_id_list):
        field_name = "n{}".format(n + 1)
        subfilename = "{}_{}.dat".format(basename, field_name)
        print("subsection input file: {}".format(n))
        print("  set file name = {}".format(subfilename))
        print("  set file variables = {}".format(field_name))
        print("  set format = binary")
        print("  set simulation variables = {}".format(field_name))
        print("  set x data points = {:d}".format(Nx))
        print("  set y data points = {:d}".format(Ny))
        print("  set z data points = {:d}".format(1))
        print("end\n")
    print("# --- End Field Inputs for Parameters File ---")
    if len(op_id_list) > 10:
        print("\nWARNING!! You will be reading more than 10 files!")
        print("WARNING!! You must change the following in the PF application code (main.cc) and recompile:")
        print("UserInputParameters<dim>(cli_options.get_parameters_filename(), {});\n".format(len(op_id_list)))

    
if __name__ == "__main__":
    print("Loading seeds from grains_hillert.txt")
    seeds = np.loadtxt('grains_hillert.txt', skiprows=1)
    
    Lx = 3000.0
    Ly = 3000.0
    dx = 5.859375
    nx = int(Lx / dx) + 1
    ny = int(Ly / dx) + 1
    exclusion_dist = int(150.0 / dx) # in voxels
    output_filename = "polycrystal_input.vtk"
    write_flat_binary = True
    
    print("Calculating Voronoi tesselation.")
    tree = KDTree(seeds, boxsize=(Lx, Ly))
    gid = np.zeros((nx, ny))
    
    for i in range(nx):
        for j in range(ny):
            x = i*dx
            y = j*dx
            min_dist, min_index = tree.query(np.array([x,y]))
            gid[i,j] = min_index
    
    op_mapping, num_ops = op_assignment(gid, exclusion_dist)
    
    # Write VTK
    print("Writing PRISMS-PF input file (binary VTK).")
    
    x_coords = np.linspace(0, Lx, num=nx)
    y_coords = np.linspace(0, Ly, num=ny)
    if write_flat_binary:
        write_fb_order_parameters(gid, op_mapping, output_filename)
    else:
        write_order_parameters(gid, (x_coords, y_coords), op_mapping, output_filename)
        all_fields = []
        for i in range(num_ops):
            all_fields.append("n{}".format(i))
        all_fields_str = ",".join(all_fields)
        print("Add the following to the PRISMS-PF parameters file:")
        print("subsection input file: 0")
        print("  set file name = {}".format(output_filename))
        print("  set file variables = {}".format(all_fields_str))
        print("  set format = vtk")
        print("  set simulation variables = {}".format(all_fields_str))
        print("end\n")
    
    print("\nDone!\n\n")
    