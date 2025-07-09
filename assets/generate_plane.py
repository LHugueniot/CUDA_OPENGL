# generate_plane_with_slash.py

size = 20
spacing = 0.5
half = (size * spacing) / 2

with open("grid_plane.obj", "w") as f:
    f.write("# Simple flat plane grid with UVs and normals\n")

    # Write vertices
    for i in range(size + 1):
        for j in range(size + 1):
            x = -half + i * spacing
            z = -half + j * spacing
            f.write(f"v {x} 0 {z}\n")

    # Write faces
    for i in range(size):
        for j in range(size):
            v0 = i * (size + 1) + j + 1
            v1 = v0 + 1
            v2 = v0 + (size + 1)
            v3 = v2 + 1

            # All normals are index 1 here
            n = 1

            # Two triangles per quad
            f.write(f"f {v0}/{v0}/{n} {v1}/{v1}/{n} {v3}/{v3}/{n}\n")
            f.write(f"f {v0}/{v0}/{n} {v3}/{v3}/{n} {v2}/{v2}/{n}\n")
