import sys
from os.path import realpath

paths = [r"venv\Lib\packages", r"venv\Lib\site_packages", realpath(r"SmartGlasses(DBZ)"), realpath(r"venv\Scripts"),
         realpath("")]

print("D:\\")
sys.path.append("D:")

for i in paths:
    pth = realpath(i)
    print(pth)
    if pth not in paths:
        sys.path.append(realpath(i))
