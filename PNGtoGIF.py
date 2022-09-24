import imageio, os
GIF = []
filepath = "Examples/DEM3D/vtkDataTest5/Animation"
filenames = sorted((fn for fn in os.listdir(filepath) if fn.endswith('.png')))
for filename in filenames:
    GIF.append(imageio.imread(filepath + "/" + filename))
imageio.mimsave(filepath + "/" + 'result.gif', GIF, fps=10)
