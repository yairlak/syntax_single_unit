import matplotlib.pyplot as plt

x, y, ns = [], [], []
for side in [64, 32, 24, 16, 8, 4, 2, 1, 0.5]:
    fn = f'../../../Output/MNI_coords/n_multivariate_voxels_with_side_{side}.csv'
    print(f'Reading: {fn}')    
    try:
        lines = open(fn, 'r').readlines()
        empty_voxels, multi_voxels, n_cubes = lines[1].strip('\n').split(',')
        print(side, int(multi_voxels)/int(n_cubes))
        x.append(side)
        y.append(int(multi_voxels)/int(n_cubes))
        ns.append(n_cubes)
    except:
        print(f'No file found for side: {side}')

fig, ax = plt.subplots(figsize=(10, 10))
ax.plot(x, y)
ax.set_xticks(x)
ax.set_xlabel('Side [mm]')
ax.set_ylabel('Fraction of multivariate decodings')

ax_n_cubes = ax.twinx()
ax_n_cubes.plot(x, ns, 'r')
ax_n_cubes.set_ylabel('Total number of cubes', color='r')

# SAVE
fn_fig = '../../../Figures/MNI_coords/n_multivariate_voxels_vs_side.png'
fig.savefig(fn_fig)
print(f'Figure saved to: {fn_fig}')
