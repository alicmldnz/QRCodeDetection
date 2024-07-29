import os

import pyboof
from pyboof import ClassSingleBand_to_dtype
from pyboof import JavaConfig
from pyboof import JavaWrapper
from pyboof import dtype_to_Class_SingleBand
from pyboof import pbg
import numpy as np


class StereoParameters:
    """
    Calibration parameters for a stereo camera system.
    """

    def __init__(self, java_object=None):
        self.left = pyboof.CameraPinhole()
        self.right = pyboof.CameraPinhole()
        self.right_to_left = pyboof.Se3_F64()

        if java_object is not None:
            self.set_from_boof(java_object)

    def load(self, path_to_file):
        abs_path_to_file = os.path.abspath(path_to_file)
        boof_stereo = pbg.gateway.jvm.boofcv.io.calibration.CalibrationIO.load(abs_path_to_file)

        if boof_stereo is None:
            raise RuntimeError("Can't load stereo parameters")

        self.set_from_boof(boof_stereo)

    def set_from_boof(self, boof_stereo_param):
        self.left.set_from_boof(boof_stereo_param.getLeft())
        self.right.set_from_boof(boof_stereo_param.getRight())
        self.right_to_left = pyboof.Se3_F64(boof_stereo_param.getRightToLeft())

    def convert_to_boof(self):
        boof = pbg.gateway.jvm.boofcv.struct.calib.StereoParameters()
        boof.getLeft().set(self.left.convert_to_boof())
        boof.getRight().set(self.right.convert_to_boof())
        boof.getRightToLeft().set(self.right_to_left.java_obj)


class ConfigDisparityBM(JavaConfig):
    def __init__(self):
        JavaConfig.__init__(self, "boofcv.factory.disparity.ConfigDisparityBM")


class ConfigDisparityBMBest5(JavaConfig):
    def __init__(self):
        JavaConfig.__init__(self, "boofcv.factory.disparity.ConfigDisparityBMBest5")


class ConfigDisparitySGM(JavaConfig):
    def __init__(self):
        JavaConfig.__init__(self, "boofcv.factory.disparity.ConfigDisparitySGM")


class DisparityError:
    """
    Error functions which can be used with block-matching stereo disparity approaches
    """
    SAD = pbg.gateway.jvm.boofcv.factory.disparity.DisparityError.SAD
    CENSUS = pbg.gateway.jvm.boofcv.factory.disparity.DisparityError.CENSUS
    NCC = pbg.gateway.jvm.boofcv.factory.disparity.DisparityError.NCC
    values = pbg.gateway.jvm.boofcv.factory.disparity.DisparityError.values()


class DisparitySgmError:
    """
    Error functions which can be used with SGM stereo disparity
    """
    ABSOLUTE_DIFFERENCE = pbg.gateway.jvm.boofcv.factory.disparity.DisparitySgmError.ABSOLUTE_DIFFERENCE
    CENSUS = pbg.gateway.jvm.boofcv.factory.disparity.DisparitySgmError.CENSUS
    MUTUAL_INFORMATION = pbg.gateway.jvm.boofcv.factory.disparity.DisparitySgmError.MUTUAL_INFORMATION
    values = pbg.gateway.jvm.boofcv.factory.disparity.DisparitySgmError.values()


class StereoRectification:
    """
    Used to compute distortion for rectified stereo images
    """

    def __init__(self, intrinsic_left, intrinsic_right, right_to_left):
        """
        Configures rectification

        :param intrinsic_left:  Intrinsic parameters for left camera
        :type intrinsic_left: pyboof.CameraPinhole
        :param intrinsic_right: Intrinsic parameters for right camera
        :type intrinsic_right: pyboof.CameraPinhole
        :param right_to_left: Extrinsic parameters for right to left camera
        :type right_to_left: pyboof.Se3_F64
        """

        boof_left = intrinsic_left.convert_to_boof()
        boof_right = intrinsic_right.convert_to_boof()

        K1 = pbg.gateway.jvm.org.ejml.data.DMatrixRMaj(3, 3)
        K2 = pbg.gateway.jvm.org.ejml.data.DMatrixRMaj(3, 3)
        pbg.gateway.jvm.boofcv.alg.geo.PerspectiveOps.pinholeToMatrix(boof_left, K1)
        pbg.gateway.jvm.boofcv.alg.geo.PerspectiveOps.pinholeToMatrix(boof_right, K2)
        left_to_right = right_to_left.invert()

        rectify_alg = pbg.gateway.jvm.boofcv.alg.geo.RectifyImageOps.createCalibrated()

        rectify_alg.process(K1, pyboof.Se3_F64().java_obj, K2, left_to_right.get_java_object())

        self.intrinsic_left = intrinsic_left
        self.intrinsic_right = intrinsic_right

        self.orig_rect1 = rectify_alg.getUndistToRectPixels1()
        self.orig_rect2 = rectify_alg.getUndistToRectPixels2()
        self.orig_rectK = rectify_alg.getCalibrationMatrix()

        self.rect1 = self.orig_rect1.copy()
        self.rect2 = self.orig_rect2.copy()
        self.rectK = self.orig_rectK.copy()

    def all_inside_left(self):
        """
        Adjusts the rectification to ensure that there are no dead regions with no pixels.
        """
        self.rect1.set(self.orig_rect1)
        self.rect2.set(self.orig_rect2)
        self.rectK.set(self.orig_rectK)

        boof_left = self.intrinsic_left.convert_to_boof()
        pbg.gateway.jvm.boofcv.alg.geo.RectifyImageOps.allInsideLeft(boof_left, self.rect1, self.rect2, self.rectK, None)

    def full_view_left(self):
        """
        Adjusts the rectification to ensure that the full view (every single pixel) is inside the left camera view
        """
        self.rect1.set(self.orig_rect1)
        self.rect2.set(self.orig_rect2)
        self.rectK.set(self.orig_rectK)

        boof_left = self.intrinsic_left.convert_to_boof()
        pbg.gateway.jvm.boofcv.alg.geo.RectifyImageOps.fullViewLeft(boof_left, self.rect1, self.rect2, self.rectK, None)

    def create_distortion(self, image_type, is_left_image):
        """
        Creates and returns a class for distorting the left image and rectifying it

        :param image_type: Type of image the distortion will process
        :type image_type: pyboof.ImageType
        :param is_left_image: If true the distortion is for the left image if false then the right image
        :type is_left_image: bool
        :return: ImageDistort class
        :rtype: pyboof.ImageDistort
        """
        boof_image_type = image_type.java_obj
        boof_border = pyboof.border_to_java(pyboof.Border.SKIP)

        if is_left_image:
            boof_distorter = pbg.gateway.jvm.boofcv.alg.geo.RectifyDistortImageOps. \
                rectifyImage(self.intrinsic_left.convert_to_boof(), pyboof.ejml_matrix_d_to_f(self.rect1), boof_border,
                             boof_image_type)
        else:
            boof_distorter = pbg.gateway.jvm.boofcv.alg.geo.RectifyDistortImageOps. \
                rectifyImage(self.intrinsic_right.convert_to_boof(), pyboof.ejml_matrix_d_to_f(self.rect2), boof_border,
                             boof_image_type)
        return pyboof.ImageDistort(boof_distorter)


class StereoDisparity(JavaWrapper):
    """
    Class which computes the disparity between two stereo images.  Input images are assumed to be already
    rectified.
    """

    def __init__(self, java_object):
        JavaWrapper.__init__(self, java_object)

    def process(self, image_left, image_right):
        """
        Computes disparity from two images in BoofCV format.  To get results call
        :param image_left: BoofCV image rectified from left camera
        :param image_right: BoofCV image rectified from right camera
        """
        self.java_obj.process(image_left, image_right)

    def get_disparity_image(self):
        """
        Returns the disparity image.

        For pixel level precision a GrayU8 image is returned.  For sub-pixel a GrayF32 is returned.  Disparity
        values have a range of 0 to max-min-1 disparity.  Invalid values are any value above max-min.

        :return: BoofCV GrayU8 or GrayF32
        """
        return self.java_obj.getDisparity()

    def get_border_x(self):
        return self.java_obj.get_border_x()

    def get_border_y(self):
        return self.java_obj.get_border_y()

    def get_input_type(self):
        return ClassSingleBand_to_dtype(self.java_obj.get_input_type())

    def get_disparity_type(self):
        return ClassSingleBand_to_dtype(self.java_obj.get_disparity_type())


class FactoryStereoDisparity:
    def __init__(self, dtype):
        self.boof_image_type = dtype_to_Class_SingleBand(dtype)

    def block_match(self, config: ConfigDisparityBM):
        disp_type = dtype_to_Class_SingleBand(np.float32)
        if config and not config.subpixel:
            disp_type = dtype_to_Class_SingleBand(np.uint8)

        java_obj = pbg.gateway.jvm.boofcv.factory.disparity.FactoryStereoDisparity. \
            blockMatch(config.java_obj, self.boof_image_type, disp_type)
        return StereoDisparity(java_obj)

    def block_match_best5(self, config: ConfigDisparityBMBest5):
        disp_type = dtype_to_Class_SingleBand(np.float32)
        if config and not config.subpixel:
            disp_type = dtype_to_Class_SingleBand(np.uint8)

        java_obj = pbg.gateway.jvm.boofcv.factory.disparity.FactoryStereoDisparity. \
            blockMatchBest5(config.java_obj, self.boof_image_type, disp_type)
        return StereoDisparity(java_obj)

    def sgm(self, config: ConfigDisparitySGM):
        disp_type = dtype_to_Class_SingleBand(np.float32)
        if config and not config.subpixel:
            disp_type = dtype_to_Class_SingleBand(np.uint8)

        java_obj = pbg.gateway.jvm.boofcv.factory.disparity.FactoryStereoDisparity. \
            sgm(config.java_obj, self.boof_image_type, disp_type)
        return StereoDisparity(java_obj)
