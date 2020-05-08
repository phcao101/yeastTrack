"""
Mask R-CNN
Pre processing for using OME-TIFF files from microscopy acquisition with micromanager with CVAT (labeling) and Mask R-CNN
(training and segmentation).


------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Preprocesses a directory of OME-TIFF files
    python3 tiff-reader.py --inputdir=/path/to/ometif/dir --output=--BFchannel=1 --weights=imagenet

    for me: python tiff-reader.py -i/rine_lab/local_track/sample_tiff --output=out_dir --BFchannel=1

"""

# Information about tifffile package
# https://scikit-image.org/docs/dev/api/skimage.external.tifffile.html
# https://pypi.org/project/tifffile/

import os
import sys
import matplotlib.pyplot as plt
import skimage.external.tifffile as tfile
from pystackreg import StackReg
from skimage import io, img_as_ubyte
import numpy as np

# Root directory of the project
ROOT_DIR = os.path.abspath("./")
DEFAULT_OUT_DIR = os.path.join(ROOT_DIR, "output")

############################################################
#  Pre processing
############################################################

def preprocess(input_dir, output_dir, bfchannel, meta_export, register, png_export):
    """Run preprocessing on OME-TIFF images in the given directory."""
    print("Running on {}".format(input_dir))
#    bfchannel = int(bfchannel)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file in os.listdir(input_dir):
        if file.endswith(".ome.tif"):
            print("Opening: ", os.path.join(input_dir, file))
            # Open ome-tiff file with tifffile.TiffFile
            with tfile.TiffFile(os.path.join(input_dir, file), is_ome=False) as tif:
                print(tif.series[0].shape)
                imagej_hyperstack = tif.asarray()
                #tfile.imshow(imagej_hyperstack[:,bfchannel,:,:])
                #plt.show()
                print(imagej_hyperstack.shape)
                #imagej_metadata = tif.imagej_metadata
                #micromanager_metadata = tif.micromanager_metadata
                #ome_metadata = tif.ome_metadata
                nbchannels = 1

                if register:
                    registration(imagej_hyperstack, bfchannel, nbchannels, file, output_dir)

                if png_export:
                    io.imsave(os.path.join(output_dir,file)+'.png',imagej_hyperstack[bfchannel,:,:])

# with TiffFile('/media/microjasper/Data/Marc/2018.12.10 - Fish again/Sample8_1/Sample8_1_MMStack_3-Pos_000_000.ome.tif') as tif:
#     imagej_hyperstack = tif.asarray()
#     imagej_metadata = tif.imagej_metadata
#     micromanager_metadata = tif.micromanager_metadata
#     ome_metadata = tif.ome_metadata
# imagej_hyperstack.shape
# imagej_metadata['slices']
#
#
# with ('/media/microjasper/Data/Marc/2019.07.10 - PLT3 - JRy9103 Ground truth/2019.07.10-PLT3-JRy9103_1/2019.07.10-PLT3-JRy9103_1_MMStack_1-1.ome.tif') as tif:
#     imagej_hyperstack = tif.asarray()
#     for page in tif.pages:
#         for tag in page.tags.values():
#             tag_name, tag_value = tag.name, tag.value
#             print(tag_name, tag_value)
#         image = page.asarray()

############################################################
#  Registration
############################################################
# https://pypi.org/project/pystackreg/#description
def registration(hstack, bfchannel, nbchannels, file, output_dir):
    print("This is the bfchannel: " + bfchannel)
    hstackreg = []
    sr = StackReg(StackReg.RIGID_BODY)

    # register each frame to the previous (already registered) one
    # this is what the original StackReg ImageJ plugin uses

    tmats = sr.register_stack(hstack[:,bfchannel,:,:], reference='previous')
    for channel in range(nbchannels):
        hstackreg[:,channel,:,:] = sr.transform_stack(hstack[:,channel,:,:])

    # tmats contains the transformation matrices -> they can be saved
    # and loaded at another time
    np.save(os.path.join(output_dir,file,'tmats.npy'), tmats)

    return hstackreg

############################################################
#  Command Line
############################################################

if __name__ == '__main__':
    import argparse
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Channel selection, Metadata extraction, Registration and Export of OME-TIFF files')
    parser.add_argument('-i', '--input', required=True,
                        metavar="/path/to/images-directory/",
                        help='Directory containing the OME-TIFF files')
    parser.add_argument('-o', '--output', required=False,
                        metavar="/path/to/output/",
                        help='Output images and metadata directory (default=/images-directory/output/)')
    parser.add_argument('-c', '--BFchannel', required=False,
                        default=0,
                        metavar="Bright Field Channel",
                        help='Indicates which channel is the bright field (default=0)')
    parser.add_argument('-m', '--metadata', required=False, action='store_true',
                        default=False,
                        help='Exports metadatas in a separate file (default=False)')
    parser.add_argument('-r', '--reg', required=False, action='store_true',
                        default=False,
                        help='Enables registration with first image as ref (default=False)')
    parser.add_argument('-p', '--png', required=False, action='store_true',
                        default=False,
                        help='Exports bright field channel (registered if required) to 8-bits PNGs (default=False)')

    args = parser.parse_args()

    # Validate arguments
    print("Input directory: ", args.input)
    if not args.output:
        args.output = os.path.join(args.input, "output")
    print("Output directory: ", args.output)
    if args.BFchannel:
        print("Bright field channel: ", args.BFchannel)
    if args.reg:
        args.reg = True
        print("Registration: ", args.reg)
    if args.metadata:
        print("Saving metadatas")
    if args.png:
        print("Exporting images to 8-bit PNG")

preprocess(args.input, args.output, args.BFchannel, args.metadata, args.reg, args.png)

# To access metadata:
# tif.filename
# tif.ome_metadata.
