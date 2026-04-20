import glob

files = glob.glob("datasets_korosi/labels/**/*.txt", recursive=True)

for file in files:
    with open(file, "r") as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        parts = line.split()
        if len(parts) > 0:
            parts[0] = "0"  # ubah class ke 0
            new_lines.append(" ".join(parts))

    with open(file, "w") as f:
        f.write("\n".join(new_lines))