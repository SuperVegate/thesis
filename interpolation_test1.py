import numpy as np
from scipy.interpolate import RegularGridInterpolator
import pandas as pd
import matplotlib.pyplot as plt
from skimage import exposure
import pydicom as dicom
import os
import glob
import pickle
from mpl_toolkits.mplot3d import Axes3D

def sort(array1,array2):
    array1=np.array(array1)
    array2=np.array(array2)


    sorted_indices= np.argsort(array1)

    sorted_array1=array1[sorted_indices]
    sorted_array2=array2[sorted_indices]

    

    return sorted_array1.tolist(), sorted_array2.tolist()




def get_position_from_NBvoxel(NBvoxel,number_x_voxel,number_y_voxel,d_corner_x,d_corner_y,d_corner_z):
    z = (NBvoxel // (number_x_voxel*number_y_voxel))*3+(d_corner_z+1.5)
    y = ((NBvoxel % (number_x_voxel*number_y_voxel)) // number_x_voxel)*3+(d_corner_y+1.5)
    x = (NBvoxel % number_x_voxel)*3+(d_corner_x+1.5)
    return z,y,x

def apply_window(hu_matrix, window_center, window_width):

    window_min = window_center - window_width // 2
    window_max = window_center + window_width // 2
    

    hu_matrix_windowed = np.clip(hu_matrix, window_min, window_max)
    
    
    hu_matrix_windowed = (hu_matrix_windowed - window_min) / (window_max - window_min)
    
    return hu_matrix_windowed


 

def dose_image(x_voxels,y_voxels,z_voxels,dose):
    
    slice_num=1
    voxel_number_slice=x_voxels*y_voxels
    slice=[]
    slice_out=[]
    x_array=[]
    y_array=[]
    for z in range(0,z_voxels):          #for z in range(layer_begin,z_voxels): 
        for y in range(0,y_voxels):
            for x in range(0,x_voxels):
                x_array.append(dose[z*voxel_number_slice+y*x_voxels+x])
            y_array.append(x_array)
            x_array=[]
        y_array.reverse()
        slice_out.append(y_array)
        y_array=[]

    return np.array(slice_out)


def get_file_names(directory_path):
    # 获取指定路径下的所有文件
    file_paths = glob.glob(os.path.join(directory_path, '*'))
    
    # 用于存储所有文件名的数组
    file_names = []

    for file_path in file_paths:
        # 确保只处理文件，而不是目录
        if os.path.isfile(file_path):
            # 获取文件名并添加到数组中
            file_name = os.path.basename(file_path)
            file_names.append(file_name)
    
    return file_names




folder_name = 'test'
if not os.path.exists(folder_name):
    os.makedirs(folder_name)








# 示例用法
directory_path = 'C:/Users/junji/OneDrive/Desktop/thesis/dicom_files_CT_1'
file_names = get_file_names(directory_path)

zvalue=[]
slice=[]
i=0
# 输出文件名
for file_name in file_names:
    ds = dicom.dcmread('C:/Users/junji/OneDrive/Desktop/thesis/dicom_files_CT_1/'+file_name)
    if hasattr(ds, 'ImagePositionPatient'):
        zvalue.append(float(ds.ImagePositionPatient[2]))
    else:
        print(f"Attribute 'ImagePositionPatient' not found in {file_name}.")
    
    slice.append(ds.pixel_array)

   
#print(len(zvalue))
#print(len(slice))

sorted_zvalue, soreted_slice=sort(zvalue,slice)





print(sorted_zvalue)
#soreted_slice=np.array(soreted_slice)
#print(sorted_zvalue)
ct_x_number=ct_y_number=512
slice_number=114
spacing_ct=0.976562
spacing_slice=2.5
ct_x_corner,ct_y_corner,ct_z_corner=-250,-250,sorted_zvalue[0]
window_center = 1000 
window_width = 300  
ct_x=np.linspace(ct_x_corner,ct_x_corner+spacing_ct*(ct_x_number-1),ct_x_number)
ct_y=np.linspace(ct_y_corner,ct_y_corner+spacing_ct*(ct_y_number-1),ct_y_number)
ct_z=np.linspace(ct_z_corner,ct_z_corner+spacing_slice*(slice_number-1),slice_number)
#print(sorted_zvalue)




















d_corner_x,d_corner_y,d_corner_z=-172.2541,-140.4241,-141

d_voxel_x,d_voxel_y,d_voxel_z=120,83,99

Nmuber_voxels=d_voxel_x*d_voxel_y*d_voxel_z

d_spacing=3


#zero_array=np.zeros((d_voxel_x,d_voxel_y,d_voxel_z))

#d_x=np.linspace(d_corner_x+d_spacing/2,d_spacing*d_voxel_x+d_corner_x-d_spacing/2,d_voxel_x)

#d_y=np.linspace(d_corner_y+d_spacing/2,d_spacing*d_voxel_y+d_corner_y-d_spacing/2,d_voxel_y)

#d_z=np.linspace(d_corner_z+d_spacing/2,d_spacing*d_voxel_z+d_corner_z-d_spacing/2,d_voxel_z)


x = np.linspace(d_corner_x + d_spacing / 2, d_corner_x + (d_voxel_x - 0.5) * d_spacing, d_voxel_x)
y = np.linspace(d_corner_y + d_spacing / 2, d_corner_y + (d_voxel_y - 0.5) * d_spacing, d_voxel_y)
z = np.linspace(d_corner_z + d_spacing / 2, d_corner_z + (d_voxel_z - 0.5) * d_spacing, d_voxel_z)

#print(x)


#output_dose_av_LET_grid3.0mm_Prostate_1_IMPT.txt
file_path = 'C:/Users/junji/OneDrive/Desktop/thesis/output_dose_grid3.0mm_Prostate_1_IMPT.txt'
start_line = 19

df = pd.read_csv(file_path, delim_whitespace=True, skiprows=start_line, names=['VOXEL_NUMBER', 'DOSE'])

df['VOXEL_NUMBER'] = df['VOXEL_NUMBER'].astype(int)
df['DOSE'] = df['DOSE'].astype(float)


voxel_numbers = df['VOXEL_NUMBER'].tolist()
doses = df['DOSE'].tolist()

#dose_matrix = np.zeros((d_voxel_z, d_voxel_y, d_voxel_x))
#for n, voxel in enumerate(voxel_numbers):
#    z = voxel // (d_voxel_x * d_voxel_y)
#    y = (voxel % (d_voxel_x * d_voxel_y)) // d_voxel_x
#    x = (voxel % d_voxel_x)
#    dose_matrix[z, y, x] = doses[n]


dose_array=[0]*Nmuber_voxels

for n,voxel in enumerate(voxel_numbers):
    dose_array[voxel]=doses[n]

#print(len(dose_array))
#print(dose_array[0])
#dose_3D=[]
#dose_3D=dose_image(d_voxel_x,d_voxel_y,d_voxel_z,dose_array)
dose_array=np.array(dose_array)
dose_3D=dose_array.reshape((99,83,120))

interpolator = RegularGridInterpolator((z,y,x), dose_3D, bounds_error=False, fill_value=0)



test=get_position_from_NBvoxel(21385,d_voxel_x,d_voxel_y,d_corner_x,d_corner_y,d_corner_z)

#point = np.array(test)

#print(np.shape(point))
#dose_at_point = interpolator(point)
#print(test)
#print('NB:',(test[0]-d_corner_z)/3*120*83+(test[1]-d_corner_y)/3*120+(test[2]-d_corner_x)/3)
#print(dose_at_point)

'''
if (x[0] <= test[2] <= x[-1]) and (y[0] <= test[1] <= y[-1]) and (z[0] <= test[0] <= z[-1]):
    point = np.array(test)
    print(point)
    dose_at_point = interpolator(point)
    #print("Dose at point:", dose_at_point)
else:
    #print("Test point is out of interpolation range.")
'''


Z,Y,X = np.meshgrid(ct_z, ct_y, ct_x, indexing='ij')

#X,Y,Z = np.meshgrid(ct_x, ct_y, ct_z,indexing='ij')
# points = np.array([Z.flatten(), Y.flatten(), X.flatten()]).T
# dose_final = interpolator(points).reshape((len(ct_x), len(ct_y), len(ct_z)))
print('test:',np.shape(X))
dose_final=interpolator((Z,Y,X))
#print(np.shape(dose_final))
dose_final=np.array(dose_final)
dose_final=20*dose_final/100

''''
rotated_dose_final = np.empty_like(dose_final)

# 对每个切片进行旋转，并存储在新矩阵中
for i in range(dose_final.shape[2]):
    rotated_dose_final[:, :, i] = np.rot90(dose_final[:, :, i], k=-1)

dose_final=rotated_dose_final
'''
cmap = plt.cm.jet
cmap.set_bad(color=(0, 0, 0, 0))  # 设置透明度

# 创建一个带有透明度的 dose_matrix


soreted_slice=np.array(soreted_slice)
print(np.shape(dose_final))
print(np.shape(soreted_slice))


for n in range(soreted_slice.shape[1]):
    if -150<ct_y[n]<150:
        ct_image=soreted_slice[:,n,:]
        ct_image=np.array(ct_image)
        #print(np.shape(ct_image))

        ct_image = apply_window(ct_image, window_center, window_width)

        dose_matrix_masked = dose_final[:,n,:]  

        #dose_matrix_masked= np.rot90(dose_matrix_masked,k=-1)
        #print(np.shape(dose_matrix_masked))
        #np.ma.masked_where(dose_final[:,:,n] < 0.00000001, dose_final[:,:,n])
        #ct_image[ct_image<0]=0

        dose_min = dose_matrix_masked.min()
        dose_max = dose_matrix_masked.max()
        alpha_matrix = (dose_matrix_masked - dose_min) / (dose_max - dose_min)

        alpha_matrix = np.nan_to_num(alpha_matrix, nan=0.0, posinf=1.0, neginf=0.0)

        plt.figure(figsize=(6, 6))
        plt.imshow(ct_image, cmap='gray', extent=[ct_y[0], ct_y[-1], ct_z[0], ct_z[-1]], aspect='equal')
        plt.imshow(dose_matrix_masked, cmap='jet',alpha=alpha_matrix, extent=[ct_y[0], ct_y[-1], ct_z[0], ct_z[-1]], aspect='equal')
        plt.colorbar(label='Total Dose(Gy)')  

        plt.xlabel('X Coordinates/mm')
        plt.ylabel('Y Coordinates/mm')
        plt.title(f'CT-Dose Image,Slice Number:{n}')
        #plt.show()
        #plt.xlim(-200,200)
        #plt.ylim(-200,200)
        output_path = os.path.join(folder_name, f'CT_Dose_Image_Slice_{n}.png')
        plt.savefig(output_path)
        plt.close()

        print(f'Image saved: {output_path}')


















