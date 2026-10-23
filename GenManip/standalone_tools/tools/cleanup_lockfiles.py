import os

for root, dirs, files in os.walk("saved"):
    for file in files:
        if file.endswith(".lock"):
            os.remove(os.path.join(root, file))
            print(f"removed {os.path.join(root, file)}")
