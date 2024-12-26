import cortex
import SimpleITK as sitk
import numpy as np



def get_roi_index(subject, roi_list=['AC'], full_head=False):
    # subject = "UTS03"
    # xfm = "UTS03_auto"
    # mask_dir = "/nfs/diskstation/DataStation/public_dataset/Auditory_Brain/deep-fMRI-dataset/derivative/pycortex-db/UTS03/transforms/UTS03_auto/mask_thick.nii.gz"

    xfm = subject + '_auto'
    pycortex_path = 'path to the pycortex-db in your dataset'
    mask_dir = pycortex_path + subject + \
               '/transforms/' + xfm + '/mask_thick.nii.gz'

    # Load mask ------------------------------------------------------------------
    image = sitk.ReadImage(mask_dir)
    # sitk image to numpy
    mask = sitk.GetArrayFromImage(image)
    print("mask size:", mask.shape)  # np_array size: (54, 84, 84)

    voxel_coord_total = np.where(mask == 1)
    voxel_num = len(voxel_coord_total[0])
    # assert voxel_num == data.shape[-1]
    print("Total voxel num:", voxel_num)

    if full_head:
        return list(range(voxel_num))

    # Get the map of which voxels are inside of our ROI
    index_volume, index_keys = cortex.utils.get_roi_masks(subject, xfm,
                                                          roi_list=roi_list,
                                                          # Default (None) gives all available ROIs in overlays.svg
                                                          gm_sampler='cortical-conservative',
                                                          # Select only voxels mostly within cortex
                                                          split_lr=True,
                                                          # Separate left/right ROIs (this occurs anyway with index volumes)
                                                          threshold=0.9,
                                                          # convert probability values to boolean mask for each ROI
                                                          return_dict=False  # return index volume, not dict of masks
                                                          )
    roi_index_list = [index_keys[roi] for roi in roi_list]
    voxel_coord_roi = np.where(np.logical_or.reduce([np.abs(index_volume) == roi_index for roi_index in roi_index_list]))
    print("Select voxel num:", len(voxel_coord_roi[0]))

    def search_index_fast(A, B):
        indices = []
        for a in A:
            index = np.where(np.all(B == a, axis=1))
            if index[0].size > 0:
                indices.append(index[0][0])
        return indices

    def search_index(A, B):
        diff = B[np.newaxis, :, :] - A[:, np.newaxis, :]

        is_equal = np.all(diff == 0, axis=2)

        indices = np.argmax(is_equal, axis=1)
        indices = np.where(np.any(is_equal, axis=1), indices, None)
        return indices

    voxel_coord_total = np.array(voxel_coord_total).transpose(1, 0)
    voxel_coord_roi = np.array(voxel_coord_roi).transpose(1, 0)
    roi_index_list = search_index_fast(voxel_coord_roi, voxel_coord_total)
    return roi_index_list


if __name__=='__main__':
    get_roi_index("UTS02", roi_list=['AC'], full_head=False)