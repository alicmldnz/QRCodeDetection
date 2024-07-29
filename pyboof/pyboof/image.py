import numpy as np
import os
import struct

import py4j.java_gateway as jg
import pyboof
from pyboof import pbg

import pyboof.common
from pyboof.common import JavaWrapper


class Family:
    """
    Enum for the family which the image data type belongs in.

    Equivalent to boofcv.struct.image.ImageType.Family
    """
    SINGLE_BAND = 0
    PLANAR = 1
    INTERLEAVED = 2


class BImage(JavaWrapper):
    """
    Wrapper around a BoofCV image.  Provide a slow but pythonic way to interact with the images.
    """
    def __init__(self, java_image):
        JavaWrapper.__init__(self,java_image)
        self.family = self.java_obj.getImageType().getFamily().ordinal()
        self.dtype = get_dtype(java_image)

    def get_image_type(self):
        """
        Returns the type of image it is
        :return: ImageType
        :rtype: ImageType
        """
        return ImageType(self.java_obj.getImageType())

    def ndarray(self):
        """
        Creates a new ndarray which will have the same values as this image
        :return: equivalent ndarray
        :rtype: numpy.ndarray
        """
        return boof_to_ndarray(self.java_obj)

    def create_same_shape(self):
        """
        Return an image of the same type and shape
        :return: image
        :rtype: BImage
        """
        java_image = self.java_obj.createSameShape()
        return BImage(java_image)

    def __getitem__(self, key):
        if isinstance(key, (list, tuple)):
            if len(key) == 2:
                if self.family is Family.SINGLE_BAND:
                    return self.java_obj.get(key[1], key[0])
                else:
                    raise RuntimeError("2D array like access is only valid for gray scale images")
            elif len(key) == 3:
                if self.family is Family.PLANAR:
                    return self.java_obj.getBand(key[2]).get(key[1], key[0])
                elif self.family is Family.INTERLEAVED:
                    return self.java_obj.getBand(key[2], key[1], key[0])
                else:
                    raise RuntimeError("Must be a multi band image")
            else:
                raise RuntimeError("Unexpected argument length")
        else:
            raise RuntimeError("Unexpected argument type")

    def __setitem__(self, key, value):
        if isinstance(key, (list, tuple)):
            if len(key) == 2:
                if self.family == Family.SINGLE_BAND:
                    if self.dtype == np.uint8:
                        return self.java_obj.set(key[1], key[0], int(value))
                    elif self.dtype == np.float32:
                        return self.java_obj.set(key[1], key[0], float(value))
                else:
                    raise RuntimeError("2D array like access is only valid for gray scale images")
            elif len(key) == 3:
                if self.family == Family.PLANAR:
                    if self.dtype == np.uint8:
                        return self.java_obj.getBand(key[2]).set(key[1], key[0], int(value))
                    elif self.dtype == np.float32:
                        return self.java_obj.getBand(key[2]).set(key[1], key[0], float(value))
                elif self.family == Family.INTERLEAVED:
                    if self.dtype == np.uint8:
                        return self.java_obj.setBand(key[1], key[0], key[2], int(value))
                    elif self.dtype == np.float32:
                        return self.java_obj.setBand(key[1], key[0], key[2], float(value))
                else:
                    raise RuntimeError("Must be a multi band image")
            else:
                raise RuntimeError("Unexpected argument length")
        else:
            raise RuntimeError("Unexpected argument type")

    def __getattr__(self, item):
        if item == 'shape':
            if self.family == Family.SINGLE_BAND:
                return self.java_obj.getHeight(), self.java_obj.getWidth()
            elif self.family == Family.PLANAR or self.family == Family.INTERLEAVED:
                return self.java_obj.getHeight(), self.java_obj.getWidth(), self.java_obj.getNumBands()
        return JavaWrapper.__getattr__(self, item)


class ImageType(JavaWrapper):
    """
    Description on the image data format.

    Equivalent to boofcv.struct.image.ImageType
    """
    def __init__(self, jImageType):
        JavaWrapper.__init__(self, jImageType)

    def create_boof_image(self, width, height):
        return self.java_obj.createImage(width, height)

    def get_family(self):
        return self.java_obj.getFamily().ordinal()

    def get_num_bands(self):
        return self.java_obj.getNumBands()

    def get_dtype(self):
        return JImageDataType_to_dtype(self.java_obj)


def create_ImageType(family, dtype, num_bands=1):
    """
    Returns an ImageType from the specified parameters

    :param family: Image data structure
    :type family: Family
    :param dtype: primitive data type of pixel
    :type dtype: int
    :param num_bands: Number of bands in the image.
    :type num_bands: int
    :return: Returns the ImageType
    :rtype: ImageType
    """
    if family == Family.SINGLE_BAND and num_bands != 1:
        raise Exception("SingleBand images must have only one band")

    j_image_class = dtype_to_Class_SingleBand(dtype)
    if family == Family.SINGLE_BAND:
        j_image_type = pbg.gateway.jvm.boofcv.struct.image.ImageType.single(j_image_class)
    elif family == Family.PLANAR:
        j_image_type = pbg.gateway.jvm.boofcv.struct.image.ImageType.pl(num_bands, j_image_class)
    elif family == Family.INTERLEAVED:
        j_image_type = pbg.gateway.jvm.boofcv.struct.image.ImageType.interleaved(num_bands, j_image_class)
    else:
        raise Exception("Unknown family = "+str(family))

    return ImageType(j_image_type)


def load_single_band( path , dtype ):
    """
    Loads a singe band BoofCV image
    :param path: File path to image
    :param dtype: The data type of the image
    :return: BoofCV image
    """
    file_path = os.path.abspath(path)

    boof_type = dtype_to_Class_SingleBand(dtype)
    return pbg.gateway.jvm.boofcv.io.image.UtilImageIO.loadImage(file_path,boof_type)


def load_planar( path , dtype ):
    file_path = os.path.abspath(path)

    buffered_image =  pbg.gateway.jvm.boofcv.io.image.UtilImageIO.loadImage(file_path)
    if not buffered_image:
        return None
    num_bands = buffered_image.getRaster().getNumBands()
    PLANAR = create_planar(buffered_image.getWidth(), buffered_image.getHeight(),num_bands, dtype)
    pbg.gateway.jvm.boofcv.io.image.ConvertBufferedImage.convertFrom(buffered_image,PLANAR, True)

    return PLANAR


def convert_boof_image( input , output ):
    """
    Converts between two different boofcv images.  Essentially performs a typecast and doesn't change
    pixel values.  Except if the input has more than one band and the output is gray scale.  It then
    averages the bands.

    :param input: BoofCV image
    :param output:  BoofCV image
    :return:
    """
    pbg.gateway.jvm.boofcv.core.image.GConvertImage.convert(input,output)


def ndarray_to_boof( npimg , boof_img=None):
    """
    Converts an image in ndarray format into a BoofCV image

    :param npimg: numpy image
    :param boof_img: Optional storage for BoofCV image.  None to declare a new image
    :return: Converted BoofCV image
    """
    if npimg is None:
        raise Exception("Input image is None")

    if pbg.mmap_file:
        if len(npimg.shape) == 2:
            if npimg.dtype == np.uint8:
                return mmap_numpy_to_boof_U8(npimg, boof_img)
            elif npimg.dtype == np.float32:
                return mmap_numpy_to_boof_F32(npimg, boof_img)
            else:
                raise Exception("Image type not supported yet")
        else:
            if npimg.dtype == np.uint8:
                return mmap_numpy_to_boof_IU8(npimg)
            elif npimg.dtype == np.float32:
                raise Exception("Need to add support for float images")
            else:
                raise Exception("Image type not supported yet")
    else:
        if len(npimg.shape) == 2:
            if npimg.dtype == np.uint8:
                if boof_img is None:
                    b = pbg.gateway.jvm.boofcv.struct.image.GrayU8()
                else:
                    b = boof_img
                b.setData( bytearray(npimg.data))
            elif npimg.dtype == np.float32:
                if boof_img is None:
                    b = pbg.gateway.jvm.boofcv.struct.image.GrayF32()
                else:
                    b = boof_img
                b.setData(npimg.data)
            else:
                raise Exception("Image type not supported yet")
        else:
            if boof_img is not None:
                raise RuntimeError("multiband images doesn't yet support predeclared storage")

            # TODO change this to interleaved images since that's the closest equivalent
            num_bands = npimg.shape[2]

            bands = []
            for i in range(num_bands):
                bands.append(npimg[:, :, i])

            class_type = dtype_to_Class_SingleBand(npimg.dtype)

            b = pbg.gateway.jvm.boofcv.struct.image.Planar(class_type,num_bands)

            for i in range(num_bands):
                if npimg.dtype == np.uint8:
                    band = pbg.gateway.jvm.boofcv.struct.image.GrayU8()
                    band.setData( bytearray(bands[i]) )
                elif npimg.dtype == np.float32:
                    band = pbg.gateway.jvm.boofcv.struct.image.GrayF32()
                    band.setData( bands[i] )
                band.setWidth(npimg.shape[1])
                band.setHeight(npimg.shape[0])
                band.setStride(npimg.shape[1])
                b.setBand(i,band)

    b.setWidth(npimg.shape[1])
    b.setHeight(npimg.shape[0])
    b.setStride(npimg.shape[1])

    return b


def boof_to_ndarray( boof ):
    width = boof.getWidth()
    height = boof.getHeight()
    if jg.is_instance_of(pbg.gateway, boof, pbg.gateway.jvm.boofcv.struct.image.ImageGray):
        nptype = JImageDataType_to_dtype(boof.getImageType().getDataType())
        boof_data = boof.getData()

        if pbg.mmap_file:
            if nptype == np.uint8:
                return mmap_boof_to_numpy_U8(boof)
            elif nptype == np.float32:
                return mmap_boof_to_numpy_F32(boof)
            else:
                raise RuntimeError("Unsupported image type")
        else:
            # create copy the very slow way
            N = len(boof_data)
            print("Before painful copy {}".format(N))
            data = [0]*N
            for i in range(N):
                data[i] = boof_data[i]
            print("After painful copy")
            return np.ndarray(shape=(height,width), dtype=nptype, buffer=np.array(data))
    elif jg.is_instance_of(pbg.gateway, boof, pbg.gateway.jvm.boofcv.struct.image.Planar):
        nptype = JImageDataType_to_dtype(boof.getImageType().getDataType())
        if pbg.mmap_file:
            if nptype == np.uint8:
                return mmap_boof_PU8_to_numpy_IU8(boof)
            else:
                raise RuntimeError("Unsupported image type.  Only U8 currently supported")
        else:
            raise RuntimeError("Must have mmap turned on for this image type")
    else:
        raise Exception("Boof image type not yet supported")


def gradient_dtype(dtype):
    """
    Returns the appropriate image dtype of the provided image dtype
    :param dtype: Type of input image data
    :type dtype: np.dtype
    :return: Appropriate image type to store gradient data
    :rtype: np.dtype
    """
    if dtype == np.uint8:
        return np.int16
    elif dtype == np.int8:
        return np.int16
    elif dtype == np.uint16:
        return np.int32
    elif dtype == np.int16:
        return np.int32
    elif dtype == np.float32:
        return np.float32
    elif dtype == np.float64:
        return np.float64
    else:
        raise Exception("Unknown type: "+str(dtype))


def create_single_band(width, height, dtype):
    """
    Creates a single band BoofCV image.

    :param width: Image width
    :param height: Image height
    :param dtype: data type
    :return: New instance of a BoofCV single band image
    """
    if dtype == np.uint8:
        return pbg.gateway.jvm.boofcv.struct.image.GrayU8(width, height)
    elif dtype == np.int8:
        return pbg.gateway.jvm.boofcv.struct.image.GrayS8(width, height)
    elif dtype == np.uint16:
        return pbg.gateway.jvm.boofcv.struct.image.GrayU16(width, height)
    elif dtype == np.int16:
        return pbg.gateway.jvm.boofcv.struct.image.GrayS16(width, height)
    elif dtype == np.int32:
        return pbg.gateway.jvm.boofcv.struct.image.GrayS32(width, height)
    elif dtype == np.int64:
        return pbg.gateway.jvm.boofcv.struct.image.GrayS64(width, height)
    elif dtype == np.float32:
        return pbg.gateway.jvm.boofcv.struct.image.GrayF32(width, height)
    elif dtype == np.float64:
        return pbg.gateway.jvm.boofcv.struct.image.GrayF64(width, height)
    else:
        raise Exception("Unsupported type")


def create_planar( width , height , num_bands , dtype):
    """
    Creates a Planar BoofCV image.

    :param width: Image width
    :param height: Image height
    :param num_bands: Number of bands in the image
    :param dtype: data type
    :return: New instance of a BoofCV planar multi-bandimage
    """

    jImageClass = dtype_to_Class_SingleBand(dtype)

    return pbg.gateway.jvm.boofcv.struct.image.Planar(jImageClass,width,height,num_bands)


def create_interleaved(width, height, num_bands, dtype):
    """
    Creates a interleaved BoofCV image.

    :param width: Image width
    :param height: Image height
    :param num_bands: Number of bands/channels in the image
    :param dtype: data type
    :return: New instance of a BoofCV single band image
    """
    if dtype == np.uint8:
        return pbg.gateway.jvm.boofcv.struct.image.InterleavedU8(width, height, num_bands)
    elif dtype == np.int8:
        return pbg.gateway.jvm.boofcv.struct.image.InterleavedS8(width, height, num_bands)
    elif dtype == np.uint16:
        return pbg.gateway.jvm.boofcv.struct.image.InterleavedU16(width, height, num_bands)
    elif dtype == np.int16:
        return pbg.gateway.jvm.boofcv.struct.image.InterleavedS16(width, height, num_bands)
    elif dtype == np.int32:
        return pbg.gateway.jvm.boofcv.struct.image.InterleavedS32(width, height, num_bands)
    elif dtype == np.int64:
        return pbg.gateway.jvm.boofcv.struct.image.InterleavedS64(width, height, num_bands)
    elif dtype == np.float32:
        return pbg.gateway.jvm.boofcv.struct.image.InterleavedF32(width, height, num_bands)
    elif dtype == np.float64:
        return pbg.gateway.jvm.boofcv.struct.image.InterleavedF64(width, height, num_bands)
    else:
        raise Exception("Unsupported type")

def get_dtype( boof_image ):
    """
    Given a BoofCV image return the dtype which matches the storage format of its pixels
    :param boof_image: A BoofCV Java image
    :return: The NumPy dtype
    """
    return ImageDataType_to_dtype(boof_image.getImageType().getDataType())


def JImageDataType_to_dtype( ImageDataType ):
    """
    Given an instance of the BoofCV ImageType object return the dtype which represents its
    internal data
    :param ImageType:
    :return:
    """
    if ImageDataType.isInteger():
        if ImageDataType.getNumBits() == 8:
            if ImageDataType.isSigned():
                return np.int8
            else:
                return np.uint8
        elif ImageDataType.getNumBits() == 16:
            if ImageDataType.isSigned():
                return np.int16
            else:
                return np.uint16
        elif ImageDataType.getNumBits() == 32:
            if ImageDataType.isSigned():
                return np.int32
            else:
                raise Exception("Unsigned 32bit data isn't supported")
        elif ImageDataType.getNumBits() == 64:
            if ImageDataType.isSigned():
                return np.int64
            else:
                raise Exception("Unsigned 64bit data isn't supported")
        else:
            raise Exception("Number of bits not supported for ints. "+str(ImageDataType.getNumBits()))
    else:
       if ImageDataType.getNumBits() == 32:
           return np.float32
       elif ImageDataType.getNumBits() == 64:
           return np.float64
       else:
           raise Exception("Number of bits not supported for floats. "+str(ImageDataType.getNumBits()))


def dtype_to_ImageDataType( dtype ):
    if dtype == np.uint8:
        return pbg.gateway.jvm.boofcv.struct.image.ImageDataType.U8
    elif dtype == np.int8:
        return pbg.gateway.jvm.boofcv.struct.image.ImageDataType.S8
    elif dtype == np.uint16:
        return pbg.gateway.jvm.boofcv.struct.image.ImageDataType.U16
    elif dtype == np.int16:
        return pbg.gateway.jvm.boofcv.struct.image.ImageDataType.S16
    elif dtype == np.int32:
        return pbg.gateway.jvm.boofcv.struct.image.ImageDataType.S32
    elif dtype == np.int64:
        return pbg.gateway.jvm.boofcv.struct.image.ImageDataType.S64
    elif dtype == np.float32:
        return pbg.gateway.jvm.boofcv.struct.image.ImageDataType.F32
    elif dtype == np.float64:
        return pbg.gateway.jvm.boofcv.struct.image.ImageDataType.F64
    else:
        raise Exception("No BoofCV equivalent")


def family_to_Java_Family( family ):
    if family == Family.SINGLE_BAND:
        return pbg.gateway.jvm.boofcv.struct.image.ImageType.Family.SINGLE_BAND
    elif family == Family.PLANAR:
        return pbg.gateway.jvm.boofcv.struct.image.ImageType.Family.PLANAR
    elif family == Family.INTERLEAVED:
        return pbg.gateway.jvm.boofcv.struct.image.ImageType.Family.INTERLEAVED
    else:
        raise Exception("Unknown family. "+str(family))


def dtype_to_Class_SingleBand( dtype ):
    if dtype == np.uint8:
        class_path = "boofcv.struct.image.GrayU8"
    elif dtype == np.int8:
        class_path = "boofcv.struct.image.GrayS8"
    elif dtype == np.uint16:
        class_path = "boofcv.struct.image.GrayU16"
    elif dtype == np.int16:
        class_path = "boofcv.struct.image.GrayS16"
    elif dtype == np.int32:
        class_path = "boofcv.struct.image.GrayS32"
    elif dtype == np.int64:
        class_path = "boofcv.struct.image.GrayS64"
    elif dtype == np.float32:
        class_path = "boofcv.struct.image.GrayF32"
    elif dtype == np.float64:
        class_path = "boofcv.struct.image.GrayF64"
    else:
        raise Exception("No BoofCV equivalent. "+str(dtype))

    return pbg.gateway.jvm.java.lang.Class.forName(class_path)


def ImageDataType_to_dtype( jdatatype ):
    if jdatatype == pbg.gateway.jvm.boofcv.struct.image.ImageDataType.U8:
        return np.uint8
    elif jdatatype == pbg.gateway.jvm.boofcv.struct.image.ImageDataType.S8:
        return np.int8
    elif jdatatype == pbg.gateway.jvm.boofcv.struct.image.ImageDataType.U16:
        return np.uint16
    elif jdatatype == pbg.gateway.jvm.boofcv.struct.image.ImageDataType.S16:
        return np.int16
    elif jdatatype == pbg.gateway.jvm.boofcv.struct.image.ImageDataType.S32:
        return np.int32
    elif jdatatype == pbg.gateway.jvm.boofcv.struct.image.ImageDataType.S64:
        return np.int64
    elif jdatatype == pbg.gateway.jvm.boofcv.struct.image.ImageDataType.F32:
        return np.float32
    elif jdatatype == pbg.gateway.jvm.boofcv.struct.image.ImageDataType.F64:
        return np.float64
    else:
        raise Exception("Unknown ImageDataType. "+str(jdatatype))


def ClassSingleBand_to_dtype( jclass ):
    jdatatype = pbg.gateway.jvm.boofcv.struct.image.ImageDataType.classToType(jclass)
    return ImageDataType_to_dtype(jdatatype)


def dtype_to_ImageType( dtype ):
    java_class = dtype_to_Class_SingleBand(dtype)
    return pbg.gateway.jvm.boofcv.struct.image.ImageType.single(java_class)


def fill_uniform(image, min_value, max_value):
    java_random = pbg.gateway.jvm.java.util.Random()
    pbg.gateway.jvm.boofcv.alg.misc.GImageMiscOps.fillUniform(image, java_random, float(min_value), float(max_value))


# ================================================================
#        Functions for converting images using mmap files


def mmap_numpy_to_boof_U8(numpy_image, boof_img = None):
    width = numpy_image.shape[1]
    height = numpy_image.shape[0]
    num_bands = 1

    mm = pbg.mmap_file
    mm.seek(0)
    mm.write(struct.pack('>HIII', pyboof.MmapType.IMAGE_U8,width,height,num_bands))
    mm.write(numpy_image.data)

    return pbg.gateway.jvm.pyboof.PyBoofEntryPoint.mmap.readImage_U8(boof_img)


def mmap_numpy_to_boof_F32(numpy_image, boof_img = None):
    width = numpy_image.shape[1]
    height = numpy_image.shape[0]
    num_bands = 1

    mm = pbg.mmap_file
    mm.seek(0)
    mm.write(struct.pack('>HIII', pyboof.MmapType.IMAGE_F32, width, height, num_bands))
    mm.write(numpy_image.data)

    return pbg.gateway.jvm.pyboof.PyBoofEntryPoint.mmap.readImage_F32(boof_img)


def mmap_numpy_to_boof_IU8( numpy_image , boof_img=None):
    width = numpy_image.shape[1]
    height = numpy_image.shape[0]
    num_bands = numpy_image.shape[2]

    # The image write takes less than a millisecond
    mm = pbg.mmap_file
    mm.seek(0)
    mm.write(struct.pack('>HIII', pyboof.MmapType.IMAGE_U8, width, height, num_bands))
    mm.write(numpy_image.data)

    # TODO again just invoking the java function appears to take 2 to 3 ms even if the function does nothing
    #      Hard to tell how load the actual read takes in the MMAP file.  Probably around 1ms
    return pbg.gateway.jvm.pyboof.PyBoofEntryPoint.mmap.readImage_IU8(boof_img)


def mmap_boof_to_numpy_U8(boof_image):
    pbg.gateway.jvm.pyboof.PyBoofEntryPoint.mmap.writeImage_U8(boof_image)

    mm = pbg.mmap_file
    mm.seek(0)
    header_bytes = mm.read(14) # speed it up significantly by minimizing disk access
    data_type, width, height, num_bands = struct.unpack('>hiii', header_bytes)

    if data_type is not pyboof.MmapType.IMAGE_U8:
        raise RuntimeError("Expected IMAGE_U8 in mmap file")
    if num_bands != 1:
        raise RuntimeError("Expected single band image. Found {}".format(num_bands))

    data = mm.read(width*height)
    return np.ndarray(shape=(height, width), dtype=np.uint8, buffer=np.array(data))


def mmap_boof_to_numpy_F32(boof_image):
    # PERFORMANCE NOTE: Surprisingly this executes very fast.  The python code below is by far the slowest part
    pbg.gateway.jvm.pyboof.PyBoofEntryPoint.mmap.writeImage_F32(boof_image)

    mm = pbg.mmap_file
    mm.seek(0)
    header_bytes = mm.read(14) # speed it up significantly by minimizing disk access
    data_type, width, height, num_bands = struct.unpack('>hiii', header_bytes)

    if data_type is not pyboof.MmapType.IMAGE_F32:
        raise RuntimeError("Expected IMAGE_F32 in mmap file")
    if num_bands != 1:
        raise RuntimeError("Expected single band image. Found {}".format(num_bands))

    raw_data = mm.read(width * height * 4)
    data = struct.unpack('>{}f'.format(width*height), raw_data)

    if len(data) != width*height:
        print("Unexpected data length. {}".format(len(data)))

    # create array in java format then convert into native format
    tmp = np.ndarray(shape=(height, width), dtype='>f4', order='C', buffer=raw_data)
    return tmp.astype(dtype=np.float32, copy=False)


def mmap_boof_PU8_to_numpy_IU8(boof_image):
    pbg.gateway.jvm.pyboof.PyBoofEntryPoint.mmap.writeImage_PU8_as_IU8(boof_image)

    mm = pbg.mmap_file
    mm.seek(0)
    header_bytes = mm.read(14) # speed it up significantly by minimizing disk access
    data_type, width, height, num_bands = struct.unpack('>hiii', header_bytes)

    if data_type is not pyboof.MmapType.IMAGE_U8:
        raise RuntimeError("Expected IMAGE_U8 in mmap file")

    data = mm.read(width*height*num_bands)
    return np.ndarray(shape=(height, width, num_bands), dtype=np.uint8, buffer=np.array(data))