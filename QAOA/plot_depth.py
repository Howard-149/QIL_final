import matplotlib.pyplot as plt

qaoa_energies = [-0.4304667,-0.4094261,-0.59543884,-0.8446765,-0.79575336,-0.8373448,-0.94964534,-0.86077106,-1.0289092,-0.96008223,-1.0505481,-1.0128074,-1.0857273,-1.096443,-1.0077829,]
qaoa_energies = [1.1 + i for i in qaoa_energies]
qaoa_depth = [14, 24, 34, 44, 54, 64, 74, 84, 94, 104, 114, 124, 134, 144, 154]


adapt_single = [0.8132826685905457, 0.8271166086196899, 0.967273473739624, 0.9673418402671814, 1.0983492136001587, 0.8973710536956787, 0.777545154094696, 0.9288449287414551, 1.0633704662322998, 0.9558559060096741, 1.0274100303649902, 0.9955502152442932, 1.0829724073410034, 1.0276131629943848, 1.0193370580673218]
adapt_single = [1.1 - i for i in adapt_single]
adapt_single_depth = [19, 28, 43, 55, 67, 91, 100, 115, 124, 142, 154, 169, 181, 190, 202]

adapt_multi = [0.9225035905838013, 0.9427827000617981, 1.0063464641571045, 1.0175193548202515, 1.0999062061309814, 1.0464084148406982, 1.08549165725708, 1.0247260332107544, 1.0443050861358643, 1.0627113580703735, 1.0905680656433105, 1.0958552360534668, 1.012535810470581, 1.099886178970337, 1.0996888875961304]
adapt_multi = [1.1 - i for i in adapt_multi]
adapt_multi_depth = [29, 46, 61, 81, 103, 123, 140, 157, 172, 186, 198, 210, 225, 239, 256]

# Read from log
# energy_list = []
# depth_list=[]
# with open('./results/5_cycle_wn_1.log', 'r') as file:
#     lines = file.readlines()

# for i,line in enumerate(lines):
#     if 'depth' in line:
#         try:
#             depth = int(line.split(':')[-1].strip())
#             depth_list.append(depth)
#             energy = float(lines[i-2].split('-')[-1].split()[0].strip())
#             energy_list.append(energy)
#         except ValueError:
#             pass
# print("Energy List:", energy_list)
# print("Depth List:", depth_list)
# print(len(energy_list))
# print(len(depth_list))

experments = {
    'QAOA':qaoa_energies,
    'ADAPT-single-QAOA': adapt_single,
    'ADAPT-double-QAOA':adapt_multi
}

experments_depth = {
    'QAOA':qaoa_depth,
    'ADAPT-single-QAOA': adapt_single_depth,
    'ADAPT-double-QAOA':adapt_multi_depth
}

# Create a horizontal line plot
for name, y in experments.items():
    plt.semilogy(experments_depth[name], y, marker='o', linestyle='-', label=name)

# plt.title('Absolute Error of Extracted Energy Values (Log Scale)')
plt.xlabel('depth')
plt.ylabel('QAOA Energy Error(log scale)')
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)  # Add a grid with dashed lines

# Adjust the appearance of the legend
legend = plt.legend()
legend.get_frame().set_linewidth(0.7)

# Customize the appearance of the plot
plt.tight_layout()  # Adjust layout for better appearance

# Save the figure as an image (optional)
plt.savefig('./results/5_cycle_wn_depth_1.png', dpi=300, bbox_inches='tight')

plt.show()